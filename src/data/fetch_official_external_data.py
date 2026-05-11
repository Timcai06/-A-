"""Fetch official external oil-market data used as long-horizon constraints.

The downloaded EIA series are U.S.-specific weekly data. They should be used as
external audit anchors, not as direct replacements for the contest's global SPR
and supply-disruption assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parent


OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "eia"
SOURCE_TABLE = PROJECT_ROOT / "data" / "external" / "官方外生数据来源表.csv"
COMBINED_CSV = OUTPUT_DIR / "美国官方外生约束_合并周度.csv"
SUMMARY_CSV = OUTPUT_DIR / "美国官方外生约束_关键摘要.csv"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "stage8_external_official_data_report.md"


@dataclass(frozen=True)
class EiaSeries:
    series_id: str
    chinese_name: str
    unit: str
    url: str
    output_name: str
    model_role: str


EIA_SERIES = [
    EiaSeries(
        series_id="WCSSTUS1",
        chinese_name="美国SPR原油库存",
        unit="千桶",
        url="https://www.eia.gov/dnav/pet/hist_xls/WCSSTUS1w.xls",
        output_name="美国SPR原油库存_周度.csv",
        model_role="审计战略储备释放强度；美国口径不能直接等同全球联合释放。",
    ),
    EiaSeries(
        series_id="WCESTUS1",
        chinese_name="美国商业原油库存_不含SPR",
        unit="千桶",
        url="https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls",
        output_name="美国商业原油库存_不含SPR_周度.csv",
        model_role="审计商业库存缓冲方向；美国库存不代表全球商业库存。",
    ),
    EiaSeries(
        series_id="WCRFPUS2",
        chinese_name="美国原油产量",
        unit="千桶/日",
        url="https://www.eia.gov/dnav/pet/hist_xls/WCRFPUS2w.xls",
        output_name="美国原油产量_周度.csv",
        model_role="审计非海湾供给弹性；只代表美国供给响应。",
    ),
]

REFERENCE_DATES = {
    "冲突前最近周": pd.Timestamp("2026-02-27"),
    "冲突窗口末附近": pd.Timestamp("2026-05-01"),
}


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "mathmodel-oil-external-data/1.0"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def normalize_eia_frame(raw: bytes, series: EiaSeries) -> pd.DataFrame:
    try:
        frame = pd.read_excel(BytesIO(raw), sheet_name="Data 1")
    except Exception:
        tables = pd.read_html(BytesIO(raw))
        frame = max(tables, key=len)

    frame = frame.rename(columns={frame.columns[0]: "date", frame.columns[1]: "value"})
    frame = frame[["date", "value"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    frame["series_id"] = series.series_id
    frame["指标"] = series.chinese_name
    frame["单位"] = series.unit
    frame["数值_万桶口径"] = frame["value"] / 10.0
    return frame[["date", "series_id", "指标", "value", "单位", "数值_万桶口径"]]


def nearest_value(frame: pd.DataFrame, target: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    prior = frame[frame["date"] <= target]
    if prior.empty:
        row = frame.iloc[0]
    else:
        row = prior.iloc[-1]
    return pd.Timestamp(row["date"]), float(row["value"])


def build_summary(series_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in EIA_SERIES:
        frame = series_frames[spec.series_id]
        latest = frame.iloc[-1]
        before_date, before_value = nearest_value(frame, REFERENCE_DATES["冲突前最近周"])
        end_date, end_value = nearest_value(frame, REFERENCE_DATES["冲突窗口末附近"])
        days = max((end_date - before_date).days, 1)
        change = end_value - before_value
        rows.append(
            {
                "series_id": spec.series_id,
                "指标": spec.chinese_name,
                "单位": spec.unit,
                "冲突前日期": before_date.date().isoformat(),
                "冲突前数值": before_value,
                "窗口末日期": end_date.date().isoformat(),
                "窗口末数值": end_value,
                "窗口变化": change,
                "窗口变化_万桶口径": change / 10.0,
                "窗口日均变化_万桶每日期": change / 10.0 / days,
                "最新日期": pd.Timestamp(latest["date"]).date().isoformat(),
                "最新数值": float(latest["value"]),
                "模型用途": spec.model_role,
            }
        )
    return pd.DataFrame(rows)


def write_source_table() -> None:
    rows = [
        {
            "来源": "EIA",
            "数据集": spec.chinese_name,
            "series_id": spec.series_id,
            "URL": spec.url,
            "单位": spec.unit,
            "当前状态": "已自动抓取",
            "模型用途": spec.model_role,
        }
        for spec in EIA_SERIES
    ]
    rows.extend(
        [
            {
                "来源": "IEA",
                "数据集": "Oil Market Report",
                "series_id": "OMR",
                "URL": "https://www.iea.org/reports/oil-market-report-april-2026",
                "单位": "报告/摘要",
                "当前状态": "官方摘要可引用，完整数据通常需订阅；暂不自动入模",
                "模型用途": "支撑全球供需、库存和炼厂运行的背景，不直接作为数值输入。",
            },
            {
                "来源": "OPEC",
                "数据集": "Monthly Oil Market Report",
                "series_id": "MOMR",
                "URL": "https://www.opec.org/monthly-oil-market-report.html",
                "单位": "报告/附录表",
                "当前状态": "官方入口已记录；按可获取附件表再决定是否自动入模",
                "模型用途": "支撑世界供需平衡和非OPEC供给背景，不直接覆盖赛题参数。",
            },
        ]
    )
    ensure_parent(SOURCE_TABLE)
    pd.DataFrame(rows).to_csv(SOURCE_TABLE, index=False)


def build_report(summary: pd.DataFrame) -> str:
    rows = "\n".join(
        "| {指标} | {冲突前日期} | {冲突前数值:.0f} | {窗口末日期} | {窗口末数值:.0f} | {窗口变化_万桶口径:.1f} | {窗口日均变化_万桶每日期:.2f} |".format(
            **row
        )
        for row in summary.to_dict("records")
    )
    spr = summary[summary["series_id"] == "WCSSTUS1"].iloc[0]
    commercial = summary[summary["series_id"] == "WCESTUS1"].iloc[0]
    production = summary[summary["series_id"] == "WCRFPUS2"].iloc[0]
    return f"""# 阶段8 官方外生数据补充报告

## 运行结论

本阶段新增 EIA 官方周度数据作为长期模型的外生审计约束，包括美国 SPR 原油库存、美国商业原油库存和美国原油产量。所有数据均来自 EIA 官方历史序列，本阶段没有使用新闻爬虫数据，也没有编造任何数值。

需要强调：EIA 周度序列是美国口径，不能直接等同于全球 SPR、全球商业库存或全球供给恢复能力。本文将其用作外生约束和可信度审计，而不是粗暴替换赛题给定的 200--700 万桶/日战略储备释放范围。

## 关键摘要

| 指标 | 冲突前日期 | 冲突前数值 | 窗口末日期 | 窗口末数值 | 窗口变化（万桶口径） | 日均变化（万桶/日） |
|---|---|---:|---|---:|---:|---:|
{rows}

## 对模型的含义

- 美国 SPR 在冲突窗口内变化约 {spr["窗口变化_万桶口径"]:.1f} 万桶，折算日均约 {spr["窗口日均变化_万桶每日期"]:.2f} 万桶/日。这说明真实美国 SPR 释放强度远低于赛题给定的全球/联合释放上限，因此论文中必须说明模型 SPR 参数代表国际协调释放能力，而不是美国单独释放量。
- 美国商业原油库存在窗口内变化约 {commercial["窗口变化_万桶口径"]:.1f} 万桶，说明不能简单假设商业库存必然快速耗尽。长期模型中的商业库存缓冲应写成全球市场缓冲能力，而不是美国库存单点事实。
- 美国原油产量在窗口内变化约 {production["窗口变化_万桶口径"]:.2f} 万桶/日，说明非海湾供给恢复在短期内相对慢，不足以单独抵消霍尔木兹封锁造成的大规模缺口。

## 输出产物

- `{SOURCE_TABLE.relative_to(PROJECT_ROOT)}`
- `{COMBINED_CSV.relative_to(PROJECT_ROOT)}`
- `{SUMMARY_CSV.relative_to(PROJECT_ROOT)}`
- `{(OUTPUT_DIR / "美国SPR原油库存_周度.csv").relative_to(PROJECT_ROOT)}`
- `{(OUTPUT_DIR / "美国商业原油库存_不含SPR_周度.csv").relative_to(PROJECT_ROOT)}`
- `{(OUTPUT_DIR / "美国原油产量_周度.csv").relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    series_frames: dict[str, pd.DataFrame] = {}
    for spec in EIA_SERIES:
        raw = fetch_bytes(spec.url)
        frame = normalize_eia_frame(raw, spec)
        output_path = OUTPUT_DIR / spec.output_name
        ensure_parent(output_path)
        frame.to_csv(output_path, index=False)
        series_frames[spec.series_id] = frame

    combined = pd.concat(series_frames.values(), ignore_index=True)
    summary = build_summary(series_frames)
    ensure_parent(COMBINED_CSV)
    ensure_parent(SUMMARY_CSV)
    ensure_parent(REPORT_PATH)
    combined.to_csv(COMBINED_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    write_source_table()
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")

    print("Official external data fetch complete")
    print(f"Sources: {SOURCE_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Combined: {COMBINED_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Summary: {SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
