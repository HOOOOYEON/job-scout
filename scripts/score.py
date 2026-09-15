#!/usr/bin/env python3
"""job-scout 评分与筛选引擎。

把岗位的六个维度分合成为综合分，执行硬门槛过滤，并按分数排序。

只依赖 Python 标准库，不装任何第三方包。

用法：
    # 单个岗位评分
    python score.py score --input job.json

    # 批量筛选（执行硬门槛 + 排序 + 入围判定）
    python score.py filter --input jobs.json

    # 批量筛选，写出结果文件
    python score.py filter --input jobs.json --output result.json

    # 生成去重键
    python score.py key --company "示例科技有限公司" --title "急聘AI视觉设计师" --city "成都"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# Windows 控制台默认编码可能不是 UTF-8，中文会乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------- 配置

# 综合分权重，六项之和必须为 1.0
WEIGHTS: dict[str, float] = {
    "match": 0.25,       # 与候选人真实经历的贴合度
    "company": 0.20,     # 公司正规程度
    "work": 0.20,        # 工作内容价值与成长空间
    "welfare": 0.15,     # 五险一金、双休、补贴
    "fresh": 0.10,       # 对应届生的友好程度
    "reputation": 0.10,  # 员工口碑
}

DIMENSION_LABELS: dict[str, str] = {
    "match": "匹配",
    "company": "公司",
    "work": "工作",
    "welfare": "福利",
    "fresh": "应届",
    "reputation": "口碑",
}

# 默认及格线，低于此分数不进入当日清单
DEFAULT_THRESHOLD = 80

# 通勤默认阈值（分钟）
DEFAULT_MAX_COMMUTE = 45

# 无法确认口碑时的中位数分，避免瞎猜
UNKNOWN_REPUTATION_SCORE = 60

# 双休字段的合法取值
# double     = 明确双休
# single     = 单休
# big_small  = 大小周
# rotating   = 轮休 / 月休 N 天
# unknown    = 招聘信息未提及
WEEKEND_OK = {"double"}
WEEKEND_REJECT = {"single", "big_small", "rotating"}
WEEKEND_UNKNOWN = {"unknown", ""}


# ---------------------------------------------------------------- 数据结构


@dataclass
class Job:
    """一个待评估的岗位。"""

    id: str = ""
    company: str = ""
    title: str = ""
    city: str = ""
    district: str = ""
    url: str = ""
    email: str = ""
    salary: str = ""
    commute_minutes: int | None = None
    commute_note: str = ""
    weekend: str = "unknown"
    insurance: str = "unknown"
    scores: dict[str, int] = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Job":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class Verdict:
    """对一个岗位的完整判定结果。"""

    job: Job
    passed: bool                     # 是否通过硬门槛
    rejected_by: list[str]           # 未通过的原因
    total: int                       # 综合分
    threshold: int                   # 使用的及格线
    qualified: bool                  # 是否达到及格线
    unknown_fields: list[str]        # 待核实的字段
    warnings: list[str]              # 需要留意的提示

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------- 去重键


# 公司名里可以安全忽略的组织形式后缀。
# 注意：只去掉组织形式，不要去掉「科技」「网络」这类业务描述词 ——
# 「XX科技」和「XX网络」很可能是两家不同的公司，去掉就会误判成重复。
_COMPANY_NOISE = re.compile(
    r"(有限责任公司|股份有限公司|有限公司|集团有限公司|公司|集团)"
)

# 城市名的行政区划后缀，去掉后「成都」和「成都市」才能正确判为同一个地方
_CITY_NOISE = re.compile(r"(特别行政区|自治州|地区|市|县|区)$")

# 岗位名里的营销性修饰词，去掉后不影响岗位本身的识别
_TITLE_NOISE = re.compile(
    r"(急聘|急招|高薪|双休|五险一金|包吃住|可实习|长期招聘|热招|诚聘|招聘|新增)"
)

# 全角转半角用到的偏移量
_FULLWIDTH_OFFSET = 0xFEE0


def _normalize_text(text: str) -> str:
    """全角转半角、转小写、去除空白与常见标点。"""
    if not text:
        return ""
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:                       # 全角空格
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:           # 全角 ASCII
            out.append(chr(code - _FULLWIDTH_OFFSET))
        else:
            out.append(ch)
    text = "".join(out).lower()
    text = re.sub(r"[\s\-_/·、,，.。()（）\[\]【】]", "", text)
    return text


def dedup_key(company: str, title: str, city: str) -> str:
    """生成跨平台稳定的去重键。

    同一家公司的同一个岗位，在不同招聘平台上 URL 往往不同，
    所以不能只按 URL 去重，必须按「公司 + 岗位 + 城市」的组合判断。
    """
    company_norm = _COMPANY_NOISE.sub("", _normalize_text(company))
    title_norm = _TITLE_NOISE.sub("", _normalize_text(title))
    city_norm = _CITY_NOISE.sub("", _normalize_text(city))
    return f"{company_norm}|{title_norm}|{city_norm}"


# ---------------------------------------------------------------- 硬门槛


def check_hard_gates(
    job: Job, max_commute: int = DEFAULT_MAX_COMMUTE
) -> tuple[bool, list[str], list[str], list[str]]:
    """执行硬门槛过滤。

    返回 (是否通过, 淘汰原因, 待核实字段, 提示)。
    """
    rejected: list[str] = []
    unknown: list[str] = []
    warnings: list[str] = []

    # 双休 —— 一票否决
    weekend = (job.weekend or "unknown").strip().lower()
    if weekend in WEEKEND_REJECT:
        label = {
            "single": "单休",
            "big_small": "大小周",
            "rotating": "轮休/月休",
        }.get(weekend, weekend)
        rejected.append(f"双休不达标（{label}）")
    elif weekend in WEEKEND_UNKNOWN:
        # 没提到不等于没有，不能淘汰，但要排到末尾
        unknown.append("双休待核实")
        warnings.append("招聘信息未提及双休，需在初筛时确认")

    # 通勤 —— 一票否决
    if job.commute_minutes is None:
        unknown.append("通勤待核实")
    elif job.commute_minutes > max_commute:
        rejected.append(f"通勤超时（{job.commute_minutes} 分钟 > {max_commute} 分钟）")

    # 社保 —— 加分项，不否决
    insurance = (job.insurance or "unknown").strip().lower()
    if insurance in ("", "unknown"):
        unknown.append("社保待核实")
    elif insurance == "none":
        warnings.append("招聘信息显示无社保，属红线，需重点核实")

    # 必填字段缺失检查，缺字段不否决但必须提示
    for attr, label in (("url", "投递网址"), ("salary", "薪资")):
        if not getattr(job, attr, ""):
            unknown.append(f"{label}待核实")

    if not job.email:
        warnings.append("未查到 HR 邮箱，需走官网投递")

    return (not rejected), rejected, unknown, warnings


# ---------------------------------------------------------------- 评分


def score_dimensions(scores: dict[str, int]) -> tuple[int, dict[str, int]]:
    """把六维分按权重合成综合分。

    缺失的维度按中位数给分，避免因为少填一项就全盘作废。
    """
    normalized: dict[str, int] = {}
    for key in WEIGHTS:
        raw = scores.get(key)
        if raw is None:
            normalized[key] = 50
        else:
            value = int(raw)
            if not 0 <= value <= 100:
                raise ValueError(f"维度分必须在 0-100 之间：{key}={value}")
            normalized[key] = value

    total = sum(normalized[k] * w for k, w in WEIGHTS.items())
    return round(total), normalized


def evaluate(
    job: Job,
    threshold: int = DEFAULT_THRESHOLD,
    max_commute: int = DEFAULT_MAX_COMMUTE,
) -> Verdict:
    """对单个岗位执行完整判定：硬门槛 → 评分 → 入围。"""
    passed, rejected, unknown, warnings = check_hard_gates(job, max_commute)

    scores = dict(job.scores or {})
    # 口碑无法确认归属时，退回中位数而不是瞎猜（见 references/reputation.md）
    if "reputation" not in scores:
        scores["reputation"] = UNKNOWN_REPUTATION_SCORE
        if "未查到该公司专属评价" not in unknown:
            unknown.append("口碑未查到可作为参考的评价")

    total, _ = score_dimensions(scores)

    # 未通过硬门槛的岗位不进入入围判定，避免高分造成误导
    qualified = passed and total >= threshold

    return Verdict(
        job=job,
        passed=passed,
        rejected_by=rejected,
        total=total,
        threshold=threshold,
        qualified=qualified,
        unknown_fields=unknown,
        warnings=warnings,
    )


def rank_key(v: Verdict) -> tuple:
    """排序键。

    规则（见 references/digest.md）：
      1. 通过硬门槛的排在未通过的之前
      2. 越少「待核实」项的越靠前
      3. 综合分降序
      4. 通勤时间短的靠前

    注意：双休待核实的岗位会因为第 2 条被排到后面，
    即使它的分数更高 —— 这是刻意的，确定的岗位优先。
    """
    commute = v.job.commute_minutes if v.job.commute_minutes is not None else 9999
    return (
        not v.passed,           # False(0) 排前面
        len(v.unknown_fields),  # 待核实越少越前
        -v.total,               # 分数降序
        commute,                # 通勤升序
    )


def filter_jobs(
    jobs: list[Job],
    threshold: int = DEFAULT_THRESHOLD,
    max_commute: int = DEFAULT_MAX_COMMUTE,
) -> dict[str, Any]:
    """批量筛选，返回入围清单与筛除统计。"""
    verdicts = [evaluate(j, threshold, max_commute) for j in jobs]
    verdicts.sort(key=rank_key)

    qualified = [v for v in verdicts if v.qualified]
    rejected = [v for v in verdicts if not v.passed]

    # 统计淘汰原因，便于在日报里输出「筛除说明」
    reason_counts: dict[str, int] = {}
    for v in rejected:
        for reason in v.rejected_by:
            head = reason.split("（")[0]
            reason_counts[head] = reason_counts.get(head, 0) + 1

    return {
        "total_candidates": len(jobs),
        "qualified_count": len(qualified),
        "rejected_count": len(rejected),
        "threshold": threshold,
        "max_commute": max_commute,
        "rejection_breakdown": reason_counts,
        "qualified": [v.to_dict() for v in qualified],
        "rejected": [v.to_dict() for v in rejected],
    }


# ---------------------------------------------------------------- 渲染


def format_verdict(v: Verdict) -> str:
    """把一个判定结果渲染成人能读的文本。"""
    job = v.job
    mark = "✅ 入围" if v.qualified else ("❌ 淘汰" if not v.passed else "⚠️ 未达线")
    lines = [
        f"{mark}  {job.company} · {job.title}",
        f"   综合 {v.total}｜" + " · ".join(
            f"{DIMENSION_LABELS[k]} {job.scores.get(k, '-')}" for k in WEIGHTS
        ),
    ]
    if job.salary:
        lines.append(f"   薪资：{job.salary}")
    if job.commute_minutes is not None:
        lines.append(f"   通勤：约 {job.commute_minutes} 分钟 {job.commute_note}".rstrip())
    if v.rejected_by:
        lines.append(f"   淘汰原因：{'；'.join(v.rejected_by)}")
    if v.unknown_fields:
        lines.append(f"   待核实：{'、'.join(v.unknown_fields)}")
    if v.warnings:
        lines.append(f"   提示：{'；'.join(v.warnings)}")
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI


def load_json(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"文件不存在：{path}")
    return json.loads(p.read_text(encoding="utf-8"))


def cmd_score(args: argparse.Namespace) -> int:
    raw = load_json(args.input)
    job = Job.from_dict(raw)
    verdict = evaluate(job, args.threshold, args.max_commute)

    if args.json:
        print(json.dumps(verdict.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_verdict(verdict))
    return 0


def cmd_filter(args: argparse.Namespace) -> int:
    raw = load_json(args.input)
    jobs_raw = raw.get("jobs", raw) if isinstance(raw, dict) else raw
    if not isinstance(jobs_raw, list):
        raise SystemExit("输入必须是岗位数组，或包含 jobs 字段的对象")

    jobs = [Job.from_dict(item) for item in jobs_raw]
    result = filter_jobs(jobs, args.threshold, args.max_commute)

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"已写出：{args.output}")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"候选 {result['total_candidates']} 个 → "
              f"入围 {result['qualified_count']} 个，"
              f"淘汰 {result['rejected_count']} 个（及格线 {result['threshold']}）")
        if result["rejection_breakdown"]:
            print("\n筛除说明：")
            for reason, count in sorted(
                result["rejection_breakdown"].items(), key=lambda x: -x[1]
            ):
                print(f"  - {reason}：{count} 个")
        if result["qualified"]:
            print("\n入围清单（按优先级排序）：")
            for item in result["qualified"]:
                print()
                print(format_verdict(evaluate(Job.from_dict(item["job"]),
                                              args.threshold, args.max_commute)))
    return 0


def cmd_key(args: argparse.Namespace) -> int:
    print(dedup_key(args.company, args.title, args.city))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="score.py",
        description="job-scout 评分与筛选引擎",
    )
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("score", help="对单个岗位评分")
    ps.add_argument("--input", required=True, help="岗位 JSON 文件")
    ps.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    ps.add_argument("--max-commute", type=int, default=DEFAULT_MAX_COMMUTE)
    ps.add_argument("--json", action="store_true", help="输出 JSON")
    ps.set_defaults(func=cmd_score)

    pf = sub.add_parser("filter", help="批量筛选岗位")
    pf.add_argument("--input", required=True, help="岗位数组 JSON 文件")
    pf.add_argument("--output", help="结果写出路径")
    pf.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    pf.add_argument("--max-commute", type=int, default=DEFAULT_MAX_COMMUTE)
    pf.add_argument("--json", action="store_true", help="输出 JSON")
    pf.set_defaults(func=cmd_filter)

    pk = sub.add_parser("key", help="生成去重键")
    pk.add_argument("--company", required=True)
    pk.add_argument("--title", required=True)
    pk.add_argument("--city", default="")
    pk.set_defaults(func=cmd_key)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
