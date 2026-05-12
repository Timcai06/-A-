"""Fetch official external oil-market data used as long-horizon constraints.

The EIA series are U.S.-specific weekly data. The JODI series are official
multi-country monthly oil data. Both are external audit anchors; they should
constrain the long-horizon narrative, not replace the contest attachment prices
or the problem-statement shock assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parent


OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "eia"
JODI_DIR = PROJECT_ROOT / "data" / "external" / "jodi"
SOURCE_TABLE = PROJECT_ROOT / "data" / "external" / "官方外生数据来源表.csv"
COMBINED_CSV = OUTPUT_DIR / "美国官方外生约束_合并周度.csv"
SUMMARY_CSV = OUTPUT_DIR / "美国官方外生约束_关键摘要.csv"
JODI_MONTHLY_CSV = JODI_DIR / "JODI多国月度汇总_关键指标.csv"
JODI_SUMMARY_CSV = JODI_DIR / "JODI多国外生约束_关键摘要.csv"
JODI_SOURCE_CSV = JODI_DIR / "JODI数据来源表.csv"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "官方外生数据审计报告.md"


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

JODI_YEARS = [2025, 2026]
JODI_BASE = "https://www.jodidata.org/_resources/files/downloads/oil-data/annual-csv"

PRODUCT_NAMES = {
    "CRUDEOIL": "原油",
    "NGL": "天然气凝析液",
    "OTHERCRUDE": "其他原油",
    "TOTCRUDE": "原油及凝析油合计",
    "TOTPRODS": "油品合计",
}

FLOW_NAMES = {
    "INDPROD": "本国产量",
    "CLOSTLV": "期末库存",
    "TOTIMPSB": "总进口",
    "TOTEXPSB": "总出口",
    "REFGROUT": "炼厂总产出",
    "TOTDEMO": "总需求",
}

KEY_JODI_METRICS = [
    {
        "dataset": "primary",
        "energy_product": "TOTCRUDE",
        "flow": "INDPROD",
        "unit": "KBD",
        "指标": "多国原油及凝析油产量",
        "模型用途": "约束长期非海湾供给恢复和全球供给基线。",
    },
    {
        "dataset": "primary",
        "energy_product": "TOTCRUDE",
        "flow": "CLOSTLV",
        "unit": "KBBL",
        "指标": "多国原油及凝析油期末库存",
        "模型用途": "审计全球/多国库存缓冲数量级。",
    },
    {
        "dataset": "primary",
        "energy_product": "TOTCRUDE",
        "flow": "TOTIMPSB",
        "unit": "KBD",
        "指标": "多国原油及凝析油进口",
        "模型用途": "审计贸易流对封锁冲击的暴露方向。",
    },
    {
        "dataset": "primary",
        "energy_product": "TOTCRUDE",
        "flow": "TOTEXPSB",
        "unit": "KBD",
        "指标": "多国原油及凝析油出口",
        "模型用途": "审计贸易流和替代供应能力。",
    },
    {
        "dataset": "secondary",
        "energy_product": "TOTPRODS",
        "flow": "TOTDEMO",
        "unit": "KBD",
        "指标": "多国油品总需求",
        "模型用途": "约束长期需求调整和需求弹性叙述。",
    },
    {
        "dataset": "secondary",
        "energy_product": "TOTPRODS",
        "flow": "REFGROUT",
        "unit": "KBD",
        "指标": "多国油品炼厂总产出",
        "模型用途": "审计炼厂系统对原油供应冲击的吸收能力。",
    },
    {
        "dataset": "secondary",
        "energy_product": "TOTPRODS",
        "flow": "CLOSTLV",
        "unit": "KBBL",
        "指标": "多国油品期末库存",
        "模型用途": "审计成品油库存缓冲和库存耗尽风险。",
    },
]

REFERENCE_DATES = {
    "冲突前最近周": pd.Timestamp("2026-02-27"),
    "冲突窗口末附近": pd.Timestamp("2026-05-01"),
}

REFERENCE_MONTHS = {
    "冲突前最近月": pd.Period("2026-02", freq="M"),
    "上年同期月": pd.Period("2025-02", freq="M"),
    "上年末月": pd.Period("2025-12", freq="M"),
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


def jodi_url(dataset: str, year: int) -> str:
    if year == 2026:
        name = f"{dataset}year{year}.csv"
    else:
        name = f"{year}.csv"
    return f"{JODI_BASE}/{dataset}/{name}"


def raw_jodi_path(dataset: str, year: int) -> Path:
    chinese_dataset = "一次石油" if dataset == "primary" else "二次石油"
    return JODI_DIR / f"JODI原始{chinese_dataset}_{year}.csv"


def normalize_jodi_frame(raw: bytes, dataset: str, year: int) -> pd.DataFrame:
    frame = pd.read_csv(BytesIO(raw), dtype=str)
    frame["OBS_VALUE"] = pd.to_numeric(frame["OBS_VALUE"], errors="coerce")
    frame = frame.dropna(subset=["OBS_VALUE"]).copy()
    frame["月份"] = pd.PeriodIndex(frame["TIME_PERIOD"], freq="M")
    frame["dataset"] = dataset
    frame["year"] = year
    frame["国家地区代码"] = frame["REF_AREA"]
    frame["能源产品"] = frame["ENERGY_PRODUCT"]
    frame["流量类型"] = frame["FLOW_BREAKDOWN"]
    frame["单位"] = frame["UNIT_MEASURE"]
    frame["数值"] = frame["OBS_VALUE"]
    return frame[
        [
            "dataset",
            "year",
            "月份",
            "国家地区代码",
            "能源产品",
            "流量类型",
            "单位",
            "数值",
            "ASSESSMENT_CODE",
        ]
    ]


def build_jodi_monthly_summary(jodi_frames: list[pd.DataFrame]) -> pd.DataFrame:
    all_data = pd.concat(jodi_frames, ignore_index=True)
    rows: list[pd.DataFrame] = []
    for spec in KEY_JODI_METRICS:
        subset = all_data[
            (all_data["dataset"] == spec["dataset"])
            & (all_data["能源产品"] == spec["energy_product"])
            & (all_data["流量类型"] == spec["flow"])
            & (all_data["单位"] == spec["unit"])
        ].copy()
        if subset.empty:
            continue
        grouped = (
            subset.groupby("月份", as_index=False)
            .agg(多国上报数值=("数值", "sum"), 上报国家地区数=("国家地区代码", "nunique"))
            .sort_values("月份")
        )
        grouped["指标"] = spec["指标"]
        grouped["单位"] = spec["unit"]
        grouped["能源产品"] = spec["energy_product"]
        grouped["流量类型"] = spec["flow"]
        grouped["能源产品中文"] = PRODUCT_NAMES.get(spec["energy_product"], spec["energy_product"])
        grouped["流量类型中文"] = FLOW_NAMES.get(spec["flow"], spec["flow"])
        grouped["模型用途"] = spec["模型用途"]
        rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    summary = pd.concat(rows, ignore_index=True)
    summary["月份"] = summary["月份"].astype(str)
    return summary[
        [
            "月份",
            "指标",
            "单位",
            "多国上报数值",
            "上报国家地区数",
            "能源产品",
            "能源产品中文",
            "流量类型",
            "流量类型中文",
            "模型用途",
        ]
    ]


def nearest_jodi_month(frame: pd.DataFrame, metric: str, target: pd.Period) -> tuple[str, float, int]:
    subset = frame[frame["指标"] == metric].copy()
    if subset.empty:
        return "", float("nan"), 0
    subset["月份_period"] = pd.PeriodIndex(subset["月份"], freq="M")
    prior = subset[subset["月份_period"] <= target]
    row = prior.iloc[-1] if not prior.empty else subset.iloc[0]
    return str(row["月份"]), float(row["多国上报数值"]), int(row["上报国家地区数"])


def comparable_jodi_value(
    all_data: pd.DataFrame,
    spec: dict[str, str],
    target: pd.Period,
    countries: set[str] | None = None,
) -> tuple[str, float, int, set[str]]:
    subset = all_data[
        (all_data["dataset"] == spec["dataset"])
        & (all_data["能源产品"] == spec["energy_product"])
        & (all_data["流量类型"] == spec["flow"])
        & (all_data["单位"] == spec["unit"])
    ].copy()
    if countries is not None:
        subset = subset[subset["国家地区代码"].isin(countries)]
    if subset.empty:
        return "", float("nan"), 0, set()
    prior = subset[subset["月份"] <= target]
    if prior.empty:
        month = subset["月份"].min()
    else:
        month = prior["月份"].max()
    month_data = subset[subset["月份"] == month]
    country_set = set(month_data["国家地区代码"].dropna().astype(str))
    return str(month), float(month_data["数值"].sum()), len(country_set), country_set


def build_jodi_key_summary(monthly: pd.DataFrame, jodi_frames: list[pd.DataFrame]) -> pd.DataFrame:
    all_data = pd.concat(jodi_frames, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for spec in KEY_JODI_METRICS:
        current_month, current_value, current_areas = nearest_jodi_month(
            monthly, spec["指标"], REFERENCE_MONTHS["冲突前最近月"]
        )
        _, _, _, current_countries = comparable_jodi_value(
            all_data, spec, pd.Period(current_month, freq="M")
        )
        yoy_month, yoy_value, yoy_areas, _ = comparable_jodi_value(
            all_data, spec, REFERENCE_MONTHS["上年同期月"], current_countries
        )
        year_end_month, year_end_value, year_end_areas, _ = comparable_jodi_value(
            all_data, spec, REFERENCE_MONTHS["上年末月"], current_countries
        )
        rows.append(
            {
                "指标": spec["指标"],
                "单位": spec["unit"],
                "冲突前可得月份": current_month,
                "冲突前多国上报数值": current_value,
                "冲突前上报国家地区数": current_areas,
                "同口径上年同期月份": yoy_month,
                "同口径上年同期数值": yoy_value,
                "同口径同比变化": current_value - yoy_value if pd.notna(yoy_value) else float("nan"),
                "同口径上年末月份": year_end_month,
                "同口径上年末数值": year_end_value,
                "同口径较上年末变化": current_value - year_end_value if pd.notna(year_end_value) else float("nan"),
                "同口径上年同期国家地区数": yoy_areas,
                "同口径上年末国家地区数": year_end_areas,
                "模型用途": spec["模型用途"],
            }
        )
    return pd.DataFrame(rows)


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
                "来源": "JODI",
                "数据集": "JODI Oil World Database Primary",
                "series_id": "JODI_PRIMARY",
                "URL": jodi_url("primary", 2026),
                "单位": "KBD/KBBL",
                "当前状态": "已自动抓取 2025 和 2026 年官方 CSV",
                "模型用途": "审计多国原油及凝析油产量、库存、进出口数量级。",
            },
            {
                "来源": "JODI",
                "数据集": "JODI Oil World Database Secondary",
                "series_id": "JODI_SECONDARY",
                "URL": jodi_url("secondary", 2026),
                "单位": "KBD/KBBL",
                "当前状态": "已自动抓取 2025 和 2026 年官方 CSV",
                "模型用途": "审计多国油品需求、炼厂产出和产品库存数量级。",
            },
        ]
    )
    ensure_parent(SOURCE_TABLE)
    source_frame = pd.DataFrame(rows)
    source_frame.to_csv(SOURCE_TABLE, index=False)
    ensure_parent(JODI_SOURCE_CSV)
    source_frame[source_frame["来源"] == "JODI"].to_csv(JODI_SOURCE_CSV, index=False)


def build_report(summary: pd.DataFrame, jodi_summary: pd.DataFrame | None = None) -> str:
    rows = "\n".join(
        "| {指标} | {冲突前日期} | {冲突前数值:.0f} | {窗口末日期} | {窗口末数值:.0f} | {窗口变化_万桶口径:.1f} | {窗口日均变化_万桶每日期:.2f} |".format(
            **row
        )
        for row in summary.to_dict("records")
    )
    spr = summary[summary["series_id"] == "WCSSTUS1"].iloc[0]
    commercial = summary[summary["series_id"] == "WCESTUS1"].iloc[0]
    production = summary[summary["series_id"] == "WCRFPUS2"].iloc[0]
    jodi_section = ""
    if jodi_summary is not None and not jodi_summary.empty:
        jodi_rows = "\n".join(
            "| {指标} | {冲突前可得月份} | {冲突前多国上报数值:.2f} | {单位} | {冲突前上报国家地区数} | {同口径同比变化:.2f} | {同口径较上年末变化:.2f} |".format(
                **row
            )
            for row in jodi_summary.to_dict("records")
        )
        jodi_section = f"""
## JODI 多国口径补充

JODI Oil World Database 由国际能源论坛协调，数据来自多个国际组织和成员经济体上报。本文下载 2025 年完整年度 CSV 和 2026 年当前年度 CSV，抽取一次石油和二次石油中的产量、库存、进出口、油品需求与炼厂产出。由于 JODI 是多国上报口径，存在国家覆盖和发布时间滞后，本文把它用于全球/多国数量级审计，不反向替代附件真实价格和题面冲击参数。

| 指标 | 冲突前可得月份 | 多国上报数值 | 单位 | 上报国家地区数 | 同口径同比变化 | 同口径较上年末变化 |
|---|---|---:|---|---:|---:|---:|
{jodi_rows}

上表的变化量采用“2026 年 2 月已上报国家/地区”作为同口径样本，再回看这些国家/地区在 2025 年 2 月和 2025 年 12 月的数值，避免把 JODI 上报覆盖差异误读成真实供需变化。JODI 补充带来两点约束：第一，长期模型中“非海湾供给恢复”和“需求调整”应被解释为全球多国系统的缓慢响应，而不是某个国家单独变量；第二，库存缓冲应同时区分原油库存和成品油库存，不能只用美国商业库存代表全球库存。
"""

    return f"""# 官方外生数据补充报告

## 运行结论

本阶段新增 EIA 官方周度数据和 JODI 官方多国月度数据作为长期模型的外生审计约束。EIA 包括美国 SPR 原油库存、美国商业原油库存和美国原油产量；JODI 包括多国上报的原油及凝析油产量、库存、进出口，以及油品需求、炼厂产出和产品库存。所有数据均来自官方公开序列，本阶段没有使用新闻爬虫数据，也没有编造任何数值。

需要强调：EIA 周度序列是美国口径，不能直接等同于全球 SPR、全球商业库存或全球供给恢复能力；JODI 是多国上报口径，存在覆盖差异和发布时间滞后，也不能直接写成完整全球总量。本文将它们用作外生约束和可信度审计，而不是粗暴替换赛题给定的 200--700 万桶/日战略储备释放范围或附件真实价格。

## 关键摘要

| 指标 | 冲突前日期 | 冲突前数值 | 窗口末日期 | 窗口末数值 | 窗口变化（万桶口径） | 日均变化（万桶/日） |
|---|---|---:|---|---:|---:|---:|
{rows}

## 对模型的含义

- 美国 SPR 在冲突窗口内变化约 {spr["窗口变化_万桶口径"]:.1f} 万桶，折算日均约 {spr["窗口日均变化_万桶每日期"]:.2f} 万桶/日。这说明真实美国 SPR 释放强度远低于赛题给定的全球/联合释放上限，因此论文中必须说明模型 SPR 参数代表国际协调释放能力，而不是美国单独释放量。
- 美国商业原油库存在窗口内变化约 {commercial["窗口变化_万桶口径"]:.1f} 万桶，说明不能简单假设商业库存必然快速耗尽。长期模型中的商业库存缓冲应写成全球市场缓冲能力，而不是美国库存单点事实。
- 美国原油产量在窗口内变化约 {production["窗口变化_万桶口径"]:.2f} 万桶/日，说明非海湾供给恢复在短期内相对慢，不足以单独抵消霍尔木兹封锁造成的大规模缺口。

{jodi_section}

## 输出产物

- `{SOURCE_TABLE.relative_to(PROJECT_ROOT)}`
- `{COMBINED_CSV.relative_to(PROJECT_ROOT)}`
- `{SUMMARY_CSV.relative_to(PROJECT_ROOT)}`
- `{JODI_MONTHLY_CSV.relative_to(PROJECT_ROOT)}`
- `{JODI_SUMMARY_CSV.relative_to(PROJECT_ROOT)}`
- `{JODI_SOURCE_CSV.relative_to(PROJECT_ROOT)}`
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

    jodi_frames: list[pd.DataFrame] = []
    for year in JODI_YEARS:
        for dataset in ("primary", "secondary"):
            raw = fetch_bytes(jodi_url(dataset, year))
            raw_path = raw_jodi_path(dataset, year)
            ensure_parent(raw_path)
            raw_path.write_bytes(raw)
            jodi_frames.append(normalize_jodi_frame(raw, dataset, year))
    jodi_monthly = build_jodi_monthly_summary(jodi_frames)
    jodi_key_summary = build_jodi_key_summary(jodi_monthly, jodi_frames)
    ensure_parent(JODI_MONTHLY_CSV)
    ensure_parent(JODI_SUMMARY_CSV)
    jodi_monthly.to_csv(JODI_MONTHLY_CSV, index=False)
    jodi_key_summary.to_csv(JODI_SUMMARY_CSV, index=False)

    write_source_table()
    REPORT_PATH.write_text(build_report(summary, jodi_key_summary), encoding="utf-8")

    print("Official external data fetch complete")
    print(f"Sources: {SOURCE_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"Combined: {COMBINED_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Summary: {SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"JODI monthly: {JODI_MONTHLY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"JODI summary: {JODI_SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
