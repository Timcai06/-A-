"""Prepare Brent futures term-structure inputs.

This module is intentionally conservative.  The contest attachment contains a
front/active Brent futures price series, but not a full futures curve.  Without
official multi-maturity settlement prices, we should not manufacture M1-M3 or
M1-M6 spreads.  The script therefore creates a reproducible ingestion contract:

1. record authoritative candidate sources and public qualitative evidence;
2. create a manual official-data template;
3. if an official multi-tenor CSV is later supplied, normalize it and compute
   daily term-structure metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parents


FUTURES_DIR = PROJECT_ROOT / "data" / "external" / "futures"
OUTPUT_DIR = PROJECT_ROOT / "output" / "risk"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "期货期限结构数据接入报告.md"

MANUAL_INPUT_CSV = FUTURES_DIR / "布伦特期货多期限结算价_手工官方.csv"
TEMPLATE_CSV = FUTURES_DIR / "布伦特期货多期限结算价_模板.csv"
NORMALIZED_CSV = FUTURES_DIR / "布伦特期货多期限结算价_标准化.csv"
NASDAQ_CONTINUOUS_CSV = FUTURES_DIR / "布伦特连续期限结算价_Nasdaq_CHRIS.csv"
DAILY_METRICS_CSV = OUTPUT_DIR / "布伦特期货期限结构_日度指标.csv"
SOURCE_CSV = FUTURES_DIR / "期货期限结构来源表.csv"
EVIDENCE_CSV = FUTURES_DIR / "期限结构公开证据表.csv"
NASDAQ_REPORT_CSV = FUTURES_DIR / "Nasdaq_CHRIS拉取状态.csv"

REQUIRED_COLUMNS = [
    "trade_date",
    "contract_code",
    "contract_month",
    "settlement_price",
    "source_name",
    "source_url",
]
NASDAQ_TENOR_RANKS = [1, 3, 6]
NASDAQ_START_DATE = "2026-03-01"
NASDAQ_END_DATE = "2026-05-05"


@dataclass(frozen=True)
class TermStructureResult:
    normalized: pd.DataFrame
    daily_metrics: pd.DataFrame
    sources: pd.DataFrame
    evidence: pd.DataFrame
    nasdaq_status: pd.DataFrame
    status: str


def build_sources() -> pd.DataFrame:
    rows = [
        {
            "数据对象": "Brent多期限官方结算价",
            "推荐来源": "ICE End of Day / ICE Report Center",
            "URL": "https://www.ice.com/report/83",
            "当前状态": "作为首选来源；完整历史多期限结算价通常需要网页交互或授权下载",
            "是否入模": "尚未入模，等待可复现CSV",
        },
        {
            "数据对象": "Brent连续期限结算价",
            "推荐来源": "Nasdaq Data Link CHRIS/ICE_B1, ICE_B3, ICE_B6",
            "URL": "https://data.nasdaq.com/data/CHRIS",
            "当前状态": "已建立可选API入口；需要 NASDAQ_DATA_LINK_API_KEY 环境变量",
            "是否入模": "若成功拉取，可作为可复现连续期限结构代理；需在论文中说明它是连续合约代理而非逐个实际交割月合约",
        },
        {
            "数据对象": "EIA期限结构公开证据",
            "推荐来源": "EIA Today in Energy",
            "URL": "https://www.eia.gov/todayinenergy/detail.php?id=67544",
            "当前状态": "记录Dated Brent相对front-month futures的公开描述",
            "是否入模": "仅作论文外部证据，不作为连续数值输入",
        },
        {
            "数据对象": "Brent/WTI期货价差公开证据",
            "推荐来源": "EIA Today in Energy",
            "URL": "https://www.eia.gov/todayinenergy/detail.php?id=67424",
            "当前状态": "记录Q1冲突后Brent front-month与WTI价差异常",
            "是否入模": "仅作市场定价背景，不替代Brent曲线",
        },
    ]
    return pd.DataFrame(rows)


def build_public_evidence() -> pd.DataFrame:
    rows = [
        {
            "日期": "2026-04-24",
            "证据类型": "现货-近月期货价差",
            "公开事实": "EIA报道Dated Brent在4月初相对front-month Brent futures出现超过25美元/桶的溢价。",
            "数据来源": "EIA Today in Energy; data source listed as Intercontinental Exchange and Thomson Reuters",
            "URL": "https://www.eia.gov/todayinenergy/detail.php?id=67544",
            "建模含义": "说明封锁后近端实货市场极度紧张，期限结构/backwardation可作为长期风险约束的重要候选变量。",
            "使用边界": "这是公开描述性证据，不是逐日连续M1-M3/M1-M6历史数据，不能直接计算模型输入。",
        },
        {
            "日期": "2026-04-07",
            "证据类型": "近月期货冲击",
            "公开事实": "EIA报道Brent front-month futures在2026年一季度末达到约118美元/桶。",
            "数据来源": "EIA Today in Energy; data source listed as CME Group and Bloomberg L.P.",
            "URL": "https://www.eia.gov/todayinenergy/detail.php?id=67424",
            "建模含义": "说明冲突窗口近月期货本身已显著重定价，与附件主力合约高位平台一致。",
            "使用边界": "用于外部一致性说明，不补全多期限曲线。",
        },
        {
            "日期": "2026-04-07",
            "证据类型": "Brent-WTI跨市场价差",
            "公开事实": "EIA报道Brent-WTI futures spread在3月31日达到约25美元/桶，3月平均约11美元/桶。",
            "数据来源": "EIA Today in Energy; data source listed as CME Group and Bloomberg L.P.",
            "URL": "https://www.eia.gov/todayinenergy/detail.php?id=67424",
            "建模含义": "说明冲突冲击具有区域运输和替代供应含义，支持后续继续寻找期限结构、运费和保险费变量。",
            "使用边界": "Brent-WTI不是Brent内部期限结构，不能替代M1-M3或M1-M6。",
        },
    ]
    return pd.DataFrame(rows)


def write_template() -> None:
    if TEMPLATE_CSV.exists():
        return
    template = pd.DataFrame(
        [
            {
                "trade_date": "2026-04-01",
                "contract_code": "BRN Jun26",
                "contract_month": "2026-06",
                "settlement_price": "",
                "source_name": "ICE End of Day",
                "source_url": "https://www.ice.com/report/83",
            }
        ],
        columns=REQUIRED_COLUMNS,
    )
    template.to_csv(TEMPLATE_CSV, index=False)


def normalize_manual_input(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required futures term-structure columns: {missing}")

    df = raw[REQUIRED_COLUMNS].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    contract_month_date = pd.to_datetime(df["contract_month"].astype(str) + "-01", errors="coerce")
    df["contract_month_period"] = contract_month_date.dt.to_period("M")
    df["contract_month"] = df["contract_month_period"].astype(str)
    df["settlement_price"] = pd.to_numeric(df["settlement_price"], errors="coerce")
    df = df.dropna(subset=["trade_date", "contract_month_period", "settlement_price"]).sort_values(
        ["trade_date", "contract_month_period"]
    )
    df["tenor_rank"] = df.groupby("trade_date").cumcount() + 1
    df["tenor_label"] = "M" + df["tenor_rank"].astype(str)
    df["days_to_delivery_month"] = (
        df["contract_month_period"].dt.to_timestamp(how="start") - df["trade_date"]
    ).dt.days
    return df[
        [
            "trade_date",
            "contract_code",
            "contract_month",
            "tenor_rank",
            "tenor_label",
            "days_to_delivery_month",
            "settlement_price",
            "source_name",
            "source_url",
        ]
    ].reset_index(drop=True)


def fetch_nasdaq_chris_dataset(tenor_rank: int, api_key: str) -> pd.DataFrame:
    params = urlencode(
        {
            "start_date": NASDAQ_START_DATE,
            "end_date": NASDAQ_END_DATE,
            "api_key": api_key,
        }
    )
    url = f"https://data.nasdaq.com/api/v3/datasets/CHRIS/ICE_B{tenor_rank}.csv?{params}"
    request = Request(url, headers={"User-Agent": "mathmodel-oil-term-structure/1.0"})
    with urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf-8")

    df = pd.read_csv(StringIO(raw))
    lower = {column.lower(): column for column in df.columns}
    date_col = lower.get("date")
    settle_col = lower.get("settle") or lower.get("settlement") or lower.get("last")
    if date_col is None or settle_col is None:
        raise ValueError(f"Nasdaq CHRIS ICE_B{tenor_rank} missing Date/Settle columns: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(df[date_col], errors="coerce"),
            "tenor_rank": tenor_rank,
            "tenor_label": f"M{tenor_rank}",
            "settlement_price": pd.to_numeric(df[settle_col], errors="coerce"),
            "source_name": "Nasdaq Data Link CHRIS/ICE_B",
            "source_url": f"https://data.nasdaq.com/api/v3/datasets/CHRIS/ICE_B{tenor_rank}.csv",
        }
    )
    return out.dropna(subset=["trade_date", "settlement_price"]).sort_values("trade_date")


def empty_nasdaq_continuous() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "trade_date",
            "tenor_rank",
            "tenor_label",
            "settlement_price",
            "source_name",
            "source_url",
        ]
    )


def fetch_optional_nasdaq_chris() -> tuple[pd.DataFrame, pd.DataFrame]:
    api_key = os.environ.get("NASDAQ_DATA_LINK_API_KEY", "").strip()
    status_rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    if not api_key:
        status_rows.append(
            {
                "来源": "Nasdaq Data Link CHRIS",
                "状态": "skipped_missing_api_key",
                "说明": "未设置 NASDAQ_DATA_LINK_API_KEY，跳过自动拉取。",
            }
        )
        return empty_nasdaq_continuous(), pd.DataFrame(status_rows)

    for tenor_rank in NASDAQ_TENOR_RANKS:
        try:
            frame = fetch_nasdaq_chris_dataset(tenor_rank, api_key)
            frames.append(frame)
            status_rows.append(
                {
                    "来源": f"CHRIS/ICE_B{tenor_rank}",
                    "状态": "ok",
                    "说明": f"拉取 {len(frame)} 行。",
                }
            )
        except Exception as exc:  # noqa: BLE001
            status_rows.append(
                {
                    "来源": f"CHRIS/ICE_B{tenor_rank}",
                    "状态": "failed",
                    "说明": str(exc),
                }
            )

    combined = pd.concat(frames, ignore_index=True) if frames else empty_nasdaq_continuous()
    return combined, pd.DataFrame(status_rows)


def normalized_from_nasdaq(continuous: pd.DataFrame) -> pd.DataFrame:
    if continuous.empty:
        return empty_normalized()
    df = continuous.copy()
    df["contract_code"] = "CHRIS/ICE_B" + df["tenor_rank"].astype(str)
    df["contract_month"] = "continuous_" + df["tenor_label"]
    df["days_to_delivery_month"] = np.nan
    return df[
        [
            "trade_date",
            "contract_code",
            "contract_month",
            "tenor_rank",
            "tenor_label",
            "days_to_delivery_month",
            "settlement_price",
            "source_name",
            "source_url",
        ]
    ].sort_values(["trade_date", "tenor_rank"])


def empty_normalized() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "trade_date",
            "contract_code",
            "contract_month",
            "tenor_rank",
            "tenor_label",
            "days_to_delivery_month",
            "settlement_price",
            "source_name",
            "source_url",
        ]
    )


def build_daily_metrics(normalized: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trade_date",
        "front_month_price",
        "m3_price",
        "m6_price",
        "m1_m3_spread",
        "m1_m6_spread",
        "m1_m3_annualized_slope",
        "term_structure_state",
        "curve_contract_count",
    ]
    if normalized.empty:
        return pd.DataFrame(columns=columns)

    pivot = normalized.pivot_table(
        index="trade_date",
        columns="tenor_label",
        values="settlement_price",
        aggfunc="mean",
    ).sort_index()
    out = pd.DataFrame(index=pivot.index)
    out["front_month_price"] = pivot.get("M1")
    out["m3_price"] = pivot.get("M3")
    out["m6_price"] = pivot.get("M6")
    out["m1_m3_spread"] = out["front_month_price"] - out["m3_price"]
    out["m1_m6_spread"] = out["front_month_price"] - out["m6_price"]
    out["m1_m3_annualized_slope"] = out["m1_m3_spread"] / out["front_month_price"] * 6.0
    out["curve_contract_count"] = normalized.groupby("trade_date")["contract_month"].nunique()
    out["term_structure_state"] = np.select(
        [out["m1_m3_spread"] > 0.5, out["m1_m3_spread"] < -0.5],
        ["backwardation", "contango"],
        default="flat_or_missing",
    )
    out = out.reset_index()
    return out[columns]


def build_report(result: TermStructureResult) -> str:
    if result.daily_metrics.empty:
        data_section = """当前未发现 `data/external/futures/布伦特期货多期限结算价_手工官方.csv`，也未通过 Nasdaq Data Link API 拉取到 CHRIS 连续期限合约。因此本轮没有生成可用于入模的连续 M1-M3、M1-M6 数值输入，也没有把期限结构写入长期模型参数。"""
    else:
        latest = result.daily_metrics.dropna(subset=["front_month_price"]).tail(1).iloc[0]
        data_section = (
            f"已生成 {len(result.daily_metrics)} 个交易日的期限结构指标。"
            f"最近日期为 {pd.to_datetime(latest['trade_date']).date()}，"
            f"M1-M3价差为 {latest['m1_m3_spread']:.2f} 美元/桶。"
        )

    source_rows = "\n".join(
        f"| {row['数据对象']} | {row['推荐来源']} | {row['URL']} | {row['当前状态']} | {row['是否入模']} |"
        for row in result.sources.to_dict("records")
    )
    evidence_rows = "\n".join(
        f"| {row['日期']} | {row['证据类型']} | {row['公开事实']} | {row['建模含义']} | {row['使用边界']} |"
        for row in result.evidence.to_dict("records")
    )
    nasdaq_rows = "\n".join(
        f"| {row['来源']} | {row['状态']} | {row['说明']} |"
        for row in result.nasdaq_status.to_dict("records")
    )

    return f"""# 期货期限结构数据接入报告

## 核心结论

期限结构是长期模型下一步最值得补的市场定价变量。它比新闻情绪更接近交易层面的库存压力、近端短缺和远期风险预期；但它必须来自可复现的多期限官方结算价，不能用附件主力合约或网页截图反推出一条曲线。

{data_section}

## 数据源判断

| 数据对象 | 推荐来源 | URL | 当前状态 | 是否入模 |
|---|---|---|---|---|
{source_rows}

## 已记录的公开证据

| 日期 | 证据类型 | 公开事实 | 建模含义 | 使用边界 |
|---|---|---|---|---|
{evidence_rows}

## 自动拉取状态

| 来源 | 状态 | 说明 |
|---|---|---|
{nasdaq_rows}

## 后续接入规则

若获得 ICE 或其他可复现来源的 Brent 多期限结算价，应整理为：

`data/external/futures/布伦特期货多期限结算价_手工官方.csv`

字段必须包括：

| 字段 | 含义 |
|---|---|
| `trade_date` | 交易日期 |
| `contract_code` | 合约代码或合约名 |
| `contract_month` | 交割月份，格式如 `2026-06` |
| `settlement_price` | 官方结算价，美元/桶 |
| `source_name` | 来源名称 |
| `source_url` | 来源链接 |

脚本会自动生成：

- `{NORMALIZED_CSV.relative_to(PROJECT_ROOT)}`
- `{NASDAQ_CONTINUOUS_CSV.relative_to(PROJECT_ROOT)}`
- `{DAILY_METRICS_CSV.relative_to(PROJECT_ROOT)}`
- `{SOURCE_CSV.relative_to(PROJECT_ROOT)}`
- `{EVIDENCE_CSV.relative_to(PROJECT_ROOT)}`
- `{NASDAQ_REPORT_CSV.relative_to(PROJECT_ROOT)}`

## 建模使用边界

- 有完整多期限数据前，期限结构只作为外部证据和下一步计划，不进入模型参数。
- 有 ICE 实际交割月数据后，优先使用 M1-M3、M1-M6 价差和曲线斜率约束长期状态转移概率，不直接替代附件真实价格。
- 若使用 Nasdaq CHRIS 连续期限合约，应写成“连续期限结构代理”，不能声称它等同于逐个实际交割月官方曲线。
- 若曲线处于强 backwardation，可提高“近端短缺/升级”概率；若曲线转为 contango，可提高“缓和/库存修复”概率。
"""


def main() -> None:
    ensure_parents([TEMPLATE_CSV, NORMALIZED_CSV, DAILY_METRICS_CSV, SOURCE_CSV, EVIDENCE_CSV, REPORT_PATH])
    write_template()
    sources = build_sources()
    evidence = build_public_evidence()

    nasdaq_continuous, nasdaq_status = fetch_optional_nasdaq_chris()
    nasdaq_continuous.to_csv(NASDAQ_CONTINUOUS_CSV, index=False)
    nasdaq_status.to_csv(NASDAQ_REPORT_CSV, index=False)

    if MANUAL_INPUT_CSV.exists():
        normalized = normalize_manual_input(MANUAL_INPUT_CSV)
        status = "official_manual_csv_loaded"
    elif not nasdaq_continuous.empty:
        normalized = normalized_from_nasdaq(nasdaq_continuous)
        status = "nasdaq_chris_continuous_proxy_loaded"
    else:
        normalized = empty_normalized()
        status = "waiting_for_official_or_api_multi_tenor_data"

    daily_metrics = build_daily_metrics(normalized)
    sources.to_csv(SOURCE_CSV, index=False)
    evidence.to_csv(EVIDENCE_CSV, index=False)
    normalized.to_csv(NORMALIZED_CSV, index=False)
    daily_metrics.to_csv(DAILY_METRICS_CSV, index=False)

    result = TermStructureResult(
        normalized=normalized,
        daily_metrics=daily_metrics,
        sources=sources,
        evidence=evidence,
        nasdaq_status=nasdaq_status,
        status=status,
    )
    REPORT_PATH.write_text(build_report(result), encoding="utf-8")

    print("Futures term-structure data access prepared")
    print(f"Status: {status}")
    print(f"Template: {TEMPLATE_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Daily metrics: {DAILY_METRICS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
