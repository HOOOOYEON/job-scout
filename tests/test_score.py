#!/usr/bin/env python3
"""score.py 的回归测试。

运行：
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from score import (  # noqa: E402
    Job,
    WEIGHTS,
    UNKNOWN_REPUTATION_SCORE,
    check_hard_gates,
    dedup_key,
    evaluate,
    filter_jobs,
    score_dimensions,
)


def make_job(**overrides) -> Job:
    """造一个默认全部达标的岗位，再按需覆盖字段。"""
    base = dict(
        company="示例科技有限公司",
        title="AI 视觉设计师",
        city="成都",
        district="高新区",
        url="https://example.com/job",
        email="hr@example.com",
        salary="8-12K",
        commute_minutes=30,
        weekend="double",
        insurance="full",
        scores={
            "match": 80,
            "company": 80,
            "work": 80,
            "welfare": 80,
            "fresh": 80,
            "reputation": 80,
        },
    )
    base.update(overrides)
    return Job(**base)


class TestWeights(unittest.TestCase):
    """权重配置的完整性。"""

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0, places=6)

    def test_all_six_dimensions_present(self):
        self.assertEqual(
            set(WEIGHTS),
            {"match", "company", "work", "welfare", "fresh", "reputation"},
        )


class TestScoring(unittest.TestCase):
    """综合分合成。"""

    def test_all_eighty_gives_eighty(self):
        total, _ = score_dimensions({k: 80 for k in WEIGHTS})
        self.assertEqual(total, 80)

    def test_weighted_correctly(self):
        # 匹配分 100，其余 0 → 应为 25
        scores = {k: 0 for k in WEIGHTS}
        scores["match"] = 100
        total, _ = score_dimensions(scores)
        self.assertEqual(total, 25)

    def test_missing_dimension_falls_back_to_median(self):
        # 只给匹配分，其余缺失按 50 补
        total, normalized = score_dimensions({"match": 100})
        self.assertEqual(normalized["company"], 50)
        self.assertEqual(total, round(100 * 0.25 + 50 * 0.75))

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            score_dimensions({"match": 150})


class TestHardGates(unittest.TestCase):
    """硬门槛：双休与通勤是一票否决。"""

    def test_double_weekend_passes(self):
        passed, rejected, _, _ = check_hard_gates(make_job(weekend="double"))
        self.assertTrue(passed)
        self.assertEqual(rejected, [])

    def test_single_weekend_rejected(self):
        passed, rejected, _, _ = check_hard_gates(make_job(weekend="single"))
        self.assertFalse(passed)
        self.assertTrue(any("双休不达标" in r for r in rejected))

    def test_big_small_weekend_rejected(self):
        passed, rejected, _, _ = check_hard_gates(make_job(weekend="big_small"))
        self.assertFalse(passed)
        self.assertTrue(any("大小周" in r for r in rejected))

    def test_unknown_weekend_not_rejected_but_flagged(self):
        """未提及双休不等于没有，不能淘汰，但必须标待核实。"""
        passed, rejected, unknown, _ = check_hard_gates(make_job(weekend="unknown"))
        self.assertTrue(passed)
        self.assertEqual(rejected, [])
        self.assertIn("双休待核实", unknown)

    def test_commute_over_threshold_rejected(self):
        passed, rejected, _, _ = check_hard_gates(make_job(commute_minutes=60))
        self.assertFalse(passed)
        self.assertTrue(any("通勤超时" in r for r in rejected))

    def test_commute_at_threshold_passes(self):
        passed, _, _, _ = check_hard_gates(make_job(commute_minutes=45))
        self.assertTrue(passed)

    def test_commute_none_not_rejected_but_flagged(self):
        passed, rejected, unknown, _ = check_hard_gates(make_job(commute_minutes=None))
        self.assertTrue(passed)
        self.assertIn("通勤待核实", unknown)

    def test_missing_insurance_is_not_a_veto(self):
        """社保是加分项不是否决项，只在明确写「无」时给红线提示。"""
        passed, rejected, _, warnings = check_hard_gates(make_job(insurance="none"))
        self.assertTrue(passed)
        self.assertEqual(rejected, [])
        self.assertTrue(any("红线" in w for w in warnings))

    def test_missing_salary_flagged_not_rejected(self):
        passed, rejected, unknown, _ = check_hard_gates(make_job(salary=""))
        self.assertTrue(passed)
        self.assertIn("薪资待核实", unknown)


class TestEvaluate(unittest.TestCase):
    """完整判定。"""

    def test_high_score_but_rejected_stays_unqualified(self):
        """分数再高，硬门槛没过也不能入围 —— 这是本模型的核心纪律。"""
        job = make_job(
            weekend="single",
            scores={k: 100 for k in WEIGHTS},
        )
        v = evaluate(job)
        self.assertFalse(v.passed)
        self.assertFalse(v.qualified)
        self.assertEqual(v.total, 100)

    def test_below_threshold_not_qualified(self):
        job = make_job(scores={k: 60 for k in WEIGHTS})
        v = evaluate(job)
        self.assertTrue(v.passed)
        self.assertFalse(v.qualified)

    def test_missing_reputation_uses_median(self):
        scores = {k: 80 for k in WEIGHTS if k != "reputation"}
        job = make_job(scores=scores)
        v = evaluate(job)
        self.assertEqual(job.scores.get("reputation"), None)
        self.assertIn("口碑未查到可作为参考的评价", v.unknown_fields)
        expected, _ = score_dimensions({**scores, "reputation": UNKNOWN_REPUTATION_SCORE})
        self.assertEqual(v.total, expected)


class TestDedupKey(unittest.TestCase):
    """去重键的归一化。"""

    def test_same_job_across_platforms_same_key(self):
        a = dedup_key("示例科技有限公司", "AI 视觉设计师", "成都")
        b = dedup_key("示例科技有限责任公司", "急聘 AI 视觉设计师（双休）", "成都市")
        self.assertEqual(a, b)

    def test_city_suffix_normalized(self):
        """「成都」和「成都市」必须判为同一个地方。"""
        self.assertEqual(
            dedup_key("示例科技有限公司", "设计师", "成都"),
            dedup_key("示例科技有限公司", "设计师", "成都市"),
        )

    def test_different_business_words_not_merged(self):
        """「XX科技」和「XX网络」是两家公司，不能合并。"""
        self.assertNotEqual(
            dedup_key("示例科技有限公司", "设计师", "成都"),
            dedup_key("示例网络有限公司", "设计师", "成都"),
        )

    def test_different_city_not_merged(self):
        self.assertNotEqual(
            dedup_key("示例科技有限公司", "设计师", "成都"),
            dedup_key("示例科技有限公司", "设计师", "杭州"),
        )

    def test_fullwidth_and_case_insensitive(self):
        self.assertEqual(
            dedup_key("Example Tech Ltd", "AI Designer", "Chengdu"),
            dedup_key("example tech ltd", "ai designer", "chengdu"),
        )


class TestRanking(unittest.TestCase):
    """排序规则：确定优先，双休待核实的排后面。"""

    def test_unverified_weekend_ranks_below_confirmed(self):
        confirmed = make_job(
            company="已确认双休公司",
            weekend="double",
            scores={k: 82 for k in WEIGHTS},
        )
        unverified = make_job(
            company="待核实双休公司",
            weekend="unknown",
            scores={k: 95 for k in WEIGHTS},   # 分数明显更高
        )
        result = filter_jobs([unverified, confirmed], threshold=80)
        order = [item["job"]["company"] for item in result["qualified"]]
        self.assertEqual(order[0], "已确认双休公司")

    def test_rejected_excluded_from_qualified(self):
        good = make_job(company="好公司")
        bad = make_job(company="单休公司", weekend="single")
        result = filter_jobs([good, bad], threshold=80)
        self.assertEqual(result["qualified_count"], 1)
        self.assertEqual(result["rejected_count"], 1)
        self.assertEqual(result["qualified"][0]["job"]["company"], "好公司")

    def test_rejection_breakdown_counts(self):
        jobs = [
            make_job(company="A", weekend="single"),
            make_job(company="B", weekend="big_small"),
            make_job(company="C", commute_minutes=90),
        ]
        result = filter_jobs(jobs, threshold=80)
        self.assertEqual(result["rejected_count"], 3)
        self.assertEqual(result["rejection_breakdown"].get("双休不达标"), 2)
        self.assertEqual(result["rejection_breakdown"].get("通勤超时"), 1)

    def test_quota_is_upper_bound_not_target(self):
        """不足数时如实返回，不凑数。"""
        jobs = [make_job(company=f"C{i}") for i in range(2)]
        result = filter_jobs(jobs, threshold=80)
        self.assertEqual(result["qualified_count"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
