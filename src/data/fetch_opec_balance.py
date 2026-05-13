"""Build an OPEC global oil supply-demand balance dataset.

OPEC's digital Monthly Oil Market Report exposes the balance-of-supply-and-
demand tables as public web content. The project keeps a small official
snapshot here so the paper build remains reproducible even when the website is
temporarily inaccessible from the local environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parent


OPEC_DIR = PROJECT_ROOT / "data" / "external" / "opec"
BALANCE_CSV = OPEC_DIR / "OPEC全球供需平衡表_2025_2026.csv"
SUMMARY_CSV = OPEC_DIR / "OPEC全球供需平衡摘要.csv"
SOURCE_CSV = OPEC_DIR / "OPEC数据来源表.csv"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "OPEC全球供需平衡表接入报告.md"

OPEC_MOMR_BALANCE_URL = "https://publications.opec.org/momr/chapter/147/2738"


@dataclass(frozen=True)
class OpecBalanceRow:
    year: int
    period: str
    metric: str
    value_mbd: float | None
    source_table: str
    note: str


SNAPSHOT_ROWS: list[OpecBalanceRow] = [
    # Table 10-1: DoC production/demand balance for 2025, mb/d.
    OpecBalanceRow(2025, "2024", "世界石油需求", 103.8, "Table 10-1", "2024 estimate"),
    OpecBalanceRow(2025, "1Q25", "世界石油需求", 104.3, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "2Q25", "世界石油需求", 104.2, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "3Q25", "世界石油需求", 105.5, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "4Q25", "世界石油需求", 106.6, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "2025", "世界石油需求", 105.1, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "2025/24", "世界石油需求", 1.3, "Table 10-1", "annual change"),
    OpecBalanceRow(2025, "2024", "非DoC液体产量", 53.2, "Table 10-1", "2024 estimate"),
    OpecBalanceRow(2025, "1Q25", "非DoC液体产量", 53.7, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "2Q25", "非DoC液体产量", 54.2, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "3Q25", "非DoC液体产量", 55.0, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "4Q25", "非DoC液体产量", 53.6, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "2025", "非DoC液体产量", 54.1, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "2025/24", "非DoC液体产量", 0.9, "Table 10-1", "annual change"),
    OpecBalanceRow(2025, "2024", "DoC NGL及非常规液体", 8.5, "Table 10-1", "2024 estimate"),
    OpecBalanceRow(2025, "1Q25", "DoC NGL及非常规液体", 8.7, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "2Q25", "DoC NGL及非常规液体", 8.6, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "3Q25", "DoC NGL及非常规液体", 8.6, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "4Q25", "DoC NGL及非常规液体", 8.7, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "2025", "DoC NGL及非常规液体", 8.6, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "2025/24", "DoC NGL及非常规液体", 0.1, "Table 10-1", "annual change"),
    OpecBalanceRow(2025, "2024", "非DoC总液体及DoC NGL", 61.7, "Table 10-1", "2024 estimate"),
    OpecBalanceRow(2025, "1Q25", "非DoC总液体及DoC NGL", 62.4, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "2Q25", "非DoC总液体及DoC NGL", 62.8, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "3Q25", "非DoC总液体及DoC NGL", 63.6, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "4Q25", "非DoC总液体及DoC NGL", 62.2, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "2025", "非DoC总液体及DoC NGL", 62.7, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "2025/24", "非DoC总液体及DoC NGL", 1.0, "Table 10-1", "annual change"),
    OpecBalanceRow(2025, "2024", "DoC原油需求差额", 42.1, "Table 10-1", "world demand minus non-DoC liquids and DoC NGL"),
    OpecBalanceRow(2025, "1Q25", "DoC原油需求差额", 41.8, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "2Q25", "DoC原油需求差额", 41.4, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "3Q25", "DoC原油需求差额", 41.9, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "4Q25", "DoC原油需求差额", 44.3, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "2025", "DoC原油需求差额", 42.4, "Table 10-1", "2025 forecast"),
    OpecBalanceRow(2025, "2025/24", "DoC原油需求差额", 0.3, "Table 10-1", "annual change"),
    OpecBalanceRow(2025, "2024", "DoC原油产量", 40.9, "Table 10-1", "reported production"),
    OpecBalanceRow(2025, "1Q25", "DoC原油产量", 40.9, "Table 10-1", "reported production"),
    OpecBalanceRow(2025, "2Q25", "DoC原油产量", 41.3, "Table 10-1", "reported production"),
    OpecBalanceRow(2025, "3Q25", "DoC原油产量", 42.5, "Table 10-1", "reported production"),
    OpecBalanceRow(2025, "2024", "供需平衡", -1.2, "Table 10-1", "production minus demand for DoC crude"),
    OpecBalanceRow(2025, "1Q25", "供需平衡", -0.9, "Table 10-1", "production minus demand for DoC crude"),
    OpecBalanceRow(2025, "2Q25", "供需平衡", -0.1, "Table 10-1", "production minus demand for DoC crude"),
    OpecBalanceRow(2025, "3Q25", "供需平衡", 0.5, "Table 10-1", "production minus demand for DoC crude"),
    # Table 10-2: DoC supply/demand balance for 2026, mb/d.
    OpecBalanceRow(2026, "2025", "世界石油需求", 105.1, "Table 10-2", "2025 forecast baseline"),
    OpecBalanceRow(2026, "1Q26", "世界石油需求", 105.6, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "2Q26", "世界石油需求", 105.6, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "3Q26", "世界石油需求", 107.1, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "4Q26", "世界石油需求", 107.9, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "2026", "世界石油需求", 106.5, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "2026/25", "世界石油需求", 1.4, "Table 10-2", "annual change"),
    OpecBalanceRow(2026, "2025", "非DoC液体产量", 54.1, "Table 10-2", "2025 forecast baseline"),
    OpecBalanceRow(2026, "1Q26", "非DoC液体产量", 54.3, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "2Q26", "非DoC液体产量", 54.6, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "3Q26", "非DoC液体产量", 54.8, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "4Q26", "非DoC液体产量", 55.3, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "2026", "非DoC液体产量", 54.7, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "2026/25", "非DoC液体产量", 0.6, "Table 10-2", "annual change"),
    OpecBalanceRow(2026, "2025", "DoC NGL及非常规液体", 8.6, "Table 10-2", "2025 forecast baseline"),
    OpecBalanceRow(2026, "1Q26", "DoC NGL及非常规液体", 8.7, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "2Q26", "DoC NGL及非常规液体", 8.8, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "3Q26", "DoC NGL及非常规液体", 8.7, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "4Q26", "DoC NGL及非常规液体", 8.9, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "2026", "DoC NGL及非常规液体", 8.8, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "2026/25", "DoC NGL及非常规液体", 0.1, "Table 10-2", "annual change"),
    OpecBalanceRow(2026, "2025", "非DoC总液体及DoC NGL", 62.7, "Table 10-2", "2025 forecast baseline"),
    OpecBalanceRow(2026, "1Q26", "非DoC总液体及DoC NGL", 63.0, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "2Q26", "非DoC总液体及DoC NGL", 63.3, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "3Q26", "非DoC总液体及DoC NGL", 63.6, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "4Q26", "非DoC总液体及DoC NGL", 64.2, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "2026", "非DoC总液体及DoC NGL", 63.5, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "2026/25", "非DoC总液体及DoC NGL", 0.8, "Table 10-2", "annual change"),
    OpecBalanceRow(2026, "2025", "DoC原油需求差额", 42.4, "Table 10-2", "2025 forecast baseline"),
    OpecBalanceRow(2026, "1Q26", "DoC原油需求差额", 42.6, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "2Q26", "DoC原油需求差额", 42.2, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "3Q26", "DoC原油需求差额", 43.5, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "4Q26", "DoC原油需求差额", 43.7, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "2026", "DoC原油需求差额", 43.0, "Table 10-2", "2026 forecast"),
    OpecBalanceRow(2026, "2026/25", "DoC原油需求差额", 0.6, "Table 10-2", "annual change"),
]


def build_balance_table() -> pd.DataFrame:
    rows = [
        {
            "年份": row.year,
            "时期": row.period,
            "指标": row.metric,
            "数值_mb每日": row.value_mbd,
            "数值_万桶每日": None if row.value_mbd is None else row.value_mbd * 100,
            "来源表": row.source_table,
            "说明": row.note,
            "来源URL": OPEC_MOMR_BALANCE_URL,
        }
        for row in SNAPSHOT_ROWS
    ]
    return pd.DataFrame(rows)


def value(frame: pd.DataFrame, year: int, period: str, metric: str) -> float:
    row = frame[(frame["年份"] == year) & (frame["时期"] == period) & (frame["指标"] == metric)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["数值_mb每日"])


def build_summary(balance: pd.DataFrame) -> pd.DataFrame:
    demand_growth = value(balance, 2026, "2026/25", "世界石油需求")
    non_doc_growth = value(balance, 2026, "2026/25", "非DoC液体产量")
    total_non_doc_growth = value(balance, 2026, "2026/25", "非DoC总液体及DoC NGL")
    doc_gap_growth = value(balance, 2026, "2026/25", "DoC原油需求差额")
    annual_doc_gap = value(balance, 2026, "2026", "DoC原油需求差额")
    peak_doc_gap = max(
        value(balance, 2026, quarter, "DoC原油需求差额")
        for quarter in ["1Q26", "2Q26", "3Q26", "4Q26"]
    )
    q4_demand = value(balance, 2026, "4Q26", "世界石油需求")
    annual_demand = value(balance, 2026, "2026", "世界石油需求")
    rows = [
        {
            "指标": "OPEC全球需求年增量",
            "数值_mb每日": demand_growth,
            "数值_万桶每日": demand_growth * 100,
            "模型含义": "正常基线下全球需求仍增长，长期需求收缩不能设得过强。",
        },
        {
            "指标": "OPEC非DoC液体供给年增量",
            "数值_mb每日": non_doc_growth,
            "数值_万桶每日": non_doc_growth * 100,
            "模型含义": "非DoC供给存在恢复能力，但增量远小于霍尔木兹封锁冲击量级。",
        },
        {
            "指标": "OPEC非DoC总液体及DoC NGL年增量",
            "数值_mb每日": total_non_doc_growth,
            "数值_万桶每日": total_non_doc_growth * 100,
            "模型含义": "外部供给和NGL缓冲可部分吸收冲击，但不能单独填平缺口。",
        },
        {
            "指标": "OPEC DoC原油需求差额年增量",
            "数值_mb每日": doc_gap_growth,
            "数值_万桶每日": doc_gap_growth * 100,
            "模型含义": "对DoC原油的需求压力仍上升，支持保留长期尾部风险。",
        },
        {
            "指标": "OPEC DoC原油需求差额季节性峰值",
            "数值_mb每日": peak_doc_gap,
            "数值_万桶每日": peak_doc_gap * 100,
            "模型含义": "下半年需求差额更高，长期模型需要保留季节性上行压力。",
        },
        {
            "指标": "OPEC DoC原油需求差额峰值高于年均",
            "数值_mb每日": peak_doc_gap - annual_doc_gap,
            "数值_万桶每日": (peak_doc_gap - annual_doc_gap) * 100,
            "模型含义": "用于温和抬升尾部风险权重，而不是改变短期真实价格拟合。",
        },
        {
            "指标": "OPEC 4Q26世界需求高于年均",
            "数值_mb每日": q4_demand - annual_demand,
            "数值_万桶每日": (q4_demand - annual_demand) * 100,
            "模型含义": "用于约束长期需求恢复背景。",
        },
    ]
    return pd.DataFrame(rows)


def write_source_table() -> None:
    ensure_parent(SOURCE_CSV)
    pd.DataFrame(
        [
            {
                "来源": "OPEC",
                "数据集": "Monthly Oil Market Report - Balance of supply and demand",
                "URL": OPEC_MOMR_BALANCE_URL,
                "单位": "mb/d",
                "当前状态": "已整理官方公开表格快照",
                "模型用途": "约束长期全球需求基线、非DoC供给恢复和DoC需求差额尾部风险。",
            }
        ]
    ).to_csv(SOURCE_CSV, index=False)


def build_report(balance: pd.DataFrame, summary: pd.DataFrame) -> str:
    summary_md = summary.to_markdown(index=False, floatfmt=".2f")
    key_rows = balance[
        (balance["年份"] == 2026)
        & (balance["时期"].isin(["2026", "2026/25", "3Q26", "4Q26"]))
        & (balance["指标"].isin(["世界石油需求", "非DoC液体产量", "DoC原油需求差额"]))
    ][["年份", "时期", "指标", "数值_mb每日", "来源表"]].to_markdown(index=False, floatfmt=".2f")
    return f"""# OPEC 全球供需平衡表接入报告

## 运行结论

本报告把 OPEC Monthly Oil Market Report 中 `Balance of supply and demand` 的 2025--2026 年公开表格整理为中文 CSV。该表不是爬虫新闻数据，也不是本文估计值；它来自 OPEC 官方公开 MOMR 数字版表 10-1 和表 10-2，单位为 mb/d。

需要强调：OPEC 表给的是正常情景下的全球供需基线，不能替代赛题规定的霍尔木兹封锁冲击，也不能直接替代附件真实价格。本文把它用于长期模型参数约束：全球需求增长、非 DoC 供给恢复、DoC 原油需求差额和下半年季节性压力。

## 关键参数摘要

{summary_md}

## 2026 年关键表格摘录

{key_rows}

## 对模型的含义

- OPEC 预计 2026 年世界石油需求较 2025 年增加约 {value(balance, 2026, "2026/25", "世界石油需求"):.1f} mb/d，说明长期需求背景不是自然塌陷，需求收缩参数不能过强。
- OPEC 预计 2026 年非 DoC 液体产量增加约 {value(balance, 2026, "2026/25", "非DoC液体产量"):.1f} mb/d，能解释一部分恢复能力，但远小于霍尔木兹封锁的极端供应冲击。
- DoC 原油需求差额在 2026 年 4Q 达到 {value(balance, 2026, "4Q26", "DoC原油需求差额"):.1f} mb/d，高于全年均值 {value(balance, 2026, "2026", "DoC原油需求差额"):.1f} mb/d，支持长期模型保留下半年尾部风险。

## 输出产物

- `{BALANCE_CSV.relative_to(PROJECT_ROOT)}`
- `{SUMMARY_CSV.relative_to(PROJECT_ROOT)}`
- `{SOURCE_CSV.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    balance = build_balance_table()
    summary = build_summary(balance)
    ensure_parent(BALANCE_CSV)
    ensure_parent(SUMMARY_CSV)
    ensure_parent(REPORT_PATH)
    balance.to_csv(BALANCE_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    write_source_table()
    REPORT_PATH.write_text(build_report(balance, summary), encoding="utf-8")

    print("OPEC global balance table complete")
    print(f"Balance: {BALANCE_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Summary: {SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
