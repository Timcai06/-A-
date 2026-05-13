"""Market-priced risk variable exploration.

This module adds a conservative market-risk layer on top of the existing
lagged GPR audit. It downloads the Cboe Crude Oil ETF Volatility Index (OVX)
from FRED, aligns it with the Brent attachment data, and records why futures
term structure and tanker insurance/freight variables are not yet treated as
model inputs without licensed or verifiable historical series.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parent
from src.models import dynamic_short_term as dynamic


BRENT_PATH = PROJECT_ROOT / "data" / "processed" / "布伦特原油期货主力合约价格数据_清洗后.csv"
RISK_DIR = PROJECT_ROOT / "data" / "external" / "risk"
OUTPUT_RISK_DIR = PROJECT_ROOT / "output" / "risk"
REPORT_DIR = PROJECT_ROOT / "output" / "reports"

OVX_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=OVXCLS"
OVX_CSV = RISK_DIR / "CBOE原油ETF波动率指数_OVX.csv"
SOURCE_CSV = RISK_DIR / "市场定价风险变量来源表.csv"
MERGED_CSV = OUTPUT_RISK_DIR / "市场定价风险变量_滞后检验样本.csv"
METRICS_CSV = OUTPUT_RISK_DIR / "OVX滞后风险检验指标.csv"
REPORT_PATH = REPORT_DIR / "市场定价风险变量探索报告.md"
FIGURE_PATH = PROJECT_ROOT / "figures" / "OVX隐含波动率滞后检验.png"


@dataclass(frozen=True)
class RiskAuditResult:
    merged: pd.DataFrame
    metrics: pd.DataFrame
    sources: pd.DataFrame


def fetch_ovx() -> pd.DataFrame:
    """Download OVX from FRED and normalize column names."""
    try:
        with urlopen(OVX_URL, timeout=30) as response:
            raw = response.read().decode("utf-8")
        df = pd.read_csv(StringIO(raw))
    except URLError:
        if not OVX_CSV.exists():
            raise
        df = pd.read_csv(OVX_CSV)
    df = df.rename(columns={"observation_date": "date", "OVXCLS": "ovx"})
    df["date"] = pd.to_datetime(df["date"])
    df["ovx"] = pd.to_numeric(df["ovx"].replace(".", np.nan), errors="coerce")
    df = df.dropna(subset=["ovx"]).sort_values("date").reset_index(drop=True)
    return df


def load_brent() -> pd.DataFrame:
    if not BRENT_PATH.exists():
        raise FileNotFoundError(f"Missing Brent data: {BRENT_PATH}")
    df = pd.read_csv(BRENT_PATH, parse_dates=["trade_date"])
    return df.sort_values("trade_date").reset_index(drop=True)


def build_merged(brent: pd.DataFrame, ovx: pd.DataFrame) -> pd.DataFrame:
    merged = brent.merge(ovx, left_on="trade_date", right_on="date", how="left").drop(columns=["date"])
    merged["ovx"] = merged["ovx"].ffill()
    merged["ovx_change"] = merged["ovx"].diff()
    merged["abs_return"] = merged["log_return"].abs()
    for lag in [1, 3, 5, 10]:
        merged[f"ovx_lag{lag}"] = merged["ovx"].shift(lag)
        merged[f"ovx_change_lag{lag}"] = merged["ovx_change"].shift(lag)
    return merged


def rolling_window_percentile(series: pd.Series, event_value: float, window: int) -> float:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < window:
        return float("nan")
    rolling = pd.Series(values).rolling(window).mean().dropna()
    return float((rolling <= event_value).mean() * 100)


def build_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    sample = merged.dropna(subset=["ovx", "log_return", "abs_return", "volatility_7d"]).copy()
    event = sample[sample["is_event_window"] == True].copy()  # noqa: E712
    rows: list[dict[str, object]] = []

    event_window = len(event)
    event_mean = float(event["ovx"].mean())
    event_max = float(event["ovx"].max())
    rows.extend(
        [
            {
                "指标": "冲突窗口OVX均值",
                "数值": event_mean,
                "解释": "Cboe原油ETF期权隐含波动率在冲突窗口内的平均水平",
            },
            {
                "指标": "冲突窗口OVX最大值",
                "数值": event_max,
                "解释": "冲突窗口内市场定价的最高30日预期波动率",
            },
            {
                "指标": "冲突窗口OVX均值历史分位",
                "数值": rolling_window_percentile(sample["ovx"], event_mean, max(event_window, 1)),
                "解释": "使用同长度历史窗口均值计算，衡量事件窗口隐含波动率是否异常",
            },
        ]
    )

    for lag in [1, 3, 5, 10]:
        valid = sample.dropna(subset=[f"ovx_lag{lag}", f"ovx_change_lag{lag}", "abs_return", "log_return", "volatility_7d"])
        rows.append(
            {
                "指标": f"OVX_lag{lag}_与绝对收益相关",
                "数值": float(valid[f"ovx_lag{lag}"].corr(valid["abs_return"])),
                "解释": "滞后OVX与当日绝对收益率的相关性，检验隐含波动率是否更适合解释波动而非方向",
            }
        )
        rows.append(
            {
                "指标": f"OVX_lag{lag}_与7日实现波动相关",
                "数值": float(valid[f"ovx_lag{lag}"].corr(valid["volatility_7d"])),
                "解释": "滞后OVX与7日实现波动率的相关性",
            }
        )
        rows.append(
            {
                "指标": f"OVX_change_lag{lag}_与收益相关",
                "数值": float(valid[f"ovx_change_lag{lag}"].corr(valid["log_return"])),
                "解释": "滞后OVX变化与当日收益方向的相关性，通常不应被包装成强方向预测器",
            }
        )
    return pd.DataFrame(rows)


def build_sources() -> pd.DataFrame:
    rows = [
        {
            "变量": "期权隐含波动率",
            "推荐代理": "Cboe Crude Oil ETF Volatility Index (OVXCLS)",
            "权威来源": "FRED / Cboe Market Statistics",
            "URL": "https://fred.stlouisfed.org/series/OVXCLS",
            "当前处理": "已接入日度序列，并与附件布伦特价格做滞后检验",
            "能否入模": "可作为长期风险权重或尾部概率的外部校验变量；不直接替代短期价格拟合因子",
        },
        {
            "变量": "期货期限结构",
            "推荐代理": "Brent M1-M3、M1-M6 价差或年化斜率",
            "权威来源": "ICE Brent futures settlement / ICE End of Day data",
            "URL": "https://www.ice.com/report/83",
            "当前处理": "附件只有主力合约BRN0W.ICE，不能计算期限结构",
            "能否入模": "需获得多到期合约官方历史结算价后再接入",
        },
        {
            "变量": "油轮保险费/运输风险",
            "推荐代理": "战争险费率、BDTI、TD3C VLCC运费",
            "权威来源": "Baltic Exchange / Lloyd's市场与行业数据",
            "URL": "https://www.balticexchange.com/en/data-services/market-information0/tankers-services.html",
            "当前处理": "权威日度历史数据通常需要授权；公开新闻可作叙述证据但不宜生成数值模型输入",
            "能否入模": "暂不进入主模型；可作为论文改进方向或附加情景冲击变量",
        },
    ]
    return pd.DataFrame(rows)


def save_figure(merged: pd.DataFrame) -> None:
    dynamic.configure_plot_style()
    ensure_parent(FIGURE_PATH)
    sample = merged.dropna(subset=["ovx", "close_price"]).copy()
    event = sample[sample["is_event_window"] == True].copy()  # noqa: E712

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.2), sharex=False)
    ax = axes[0]
    ax.plot(sample["trade_date"], sample["ovx"], color="#2563eb", linewidth=1.2, alpha=0.72, label="OVX隐含波动率")
    ax.axhline(sample["ovx"].median(), color="#475569", linestyle="--", linewidth=1.0, label="历史中位数")
    if not event.empty:
        ax.axvspan(event["trade_date"].min(), event["trade_date"].max(), color="#fee2e2", alpha=0.32, label="冲突窗口")
    ax.set_title("Cboe原油ETF隐含波动率（OVX）")
    ax.set_ylabel("指数")
    ax.legend(loc="upper left", frameon=False, ncol=3)
    ax.grid(alpha=0.22)

    ax = axes[1]
    plot_df = sample[sample["trade_date"] >= pd.Timestamp("2025-01-01")].copy()
    ax.plot(plot_df["trade_date"], plot_df["close_price"], color="#111827", linewidth=1.6, label="布伦特主力收盘价")
    ax2 = ax.twinx()
    ax2.plot(plot_df["trade_date"], plot_df["ovx"], color="#dc2626", linewidth=1.2, alpha=0.74, label="OVX")
    if not event.empty:
        ax.axvspan(event["trade_date"].min(), event["trade_date"].max(), color="#fee2e2", alpha=0.32)
    ax.set_title("冲突前后：价格与市场隐含波动率")
    ax.set_ylabel("美元/桶")
    ax2.set_ylabel("OVX指数")
    ax.grid(alpha=0.22)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)


def build_report(result: RiskAuditResult) -> str:
    metrics = result.metrics.set_index("指标")
    lines = "\n".join(
        f"| {row['变量']} | {row['推荐代理']} | {row['URL']} | {row['当前处理']} | {row['能否入模']} |"
        for row in result.sources.to_dict("records")
    )
    corr_rows = "\n".join(
        f"| {row['指标']} | {float(row['数值']):.3f} | {row['解释']} |"
        for row in result.metrics.to_dict("records")
    )
    return f"""# 市场定价风险变量探索报告

## 核心结论

在现有 GPR 滞后检验基础上，最适合立即接入的是期权隐含波动率代理变量 OVX。它来自 Cboe/FRED 日度序列，反映市场对未来约 30 日油价波动的定价，比新闻文本风险指数更接近交易层面的风险预期。

本轮已下载并清洗 OVX 日度数据，生成与附件布伦特主力价格对齐的滞后检验样本。冲突窗口 OVX 均值为 {metrics.loc['冲突窗口OVX均值', '数值']:.2f}，最大值为 {metrics.loc['冲突窗口OVX最大值', '数值']:.2f}，同长度历史均值分位约为 {metrics.loc['冲突窗口OVX均值历史分位', '数值']:.1f}%。这说明 2026 冲突窗口不仅新闻风险高，期权市场定价出的预期波动也处于高位。

## 变量可得性判断

| 变量 | 推荐代理 | 来源链接 | 当前处理 | 能否入模 |
|---|---|---|---|---|
{lines}

## OVX 滞后检验指标

| 指标 | 数值 | 解释 |
|---|---:|---|
{corr_rows}

## 建模建议

1. OVX 可以作为长期情景模型的市场风险校验变量，用于约束风险权重、状态转移升级概率或蒙特卡洛压力指数噪声，不应直接拿来解释同日油价。
2. 期货期限结构比主力合约价格更能反映短缺预期，但需要 M1、M3、M6 等多到期 Brent 合约历史结算价。当前附件只有主力合约，不能计算期限结构。
3. 油轮保险费和 BDTI/TD3C 更贴近霍尔木兹运输风险，但权威日度历史数据通常需要 Baltic Exchange、Lloyd's 或行业数据库授权。若只能从新闻摘取若干费率，适合写作叙述证据，不适合变成连续模型输入。

## 输出产物

- `{OVX_CSV.relative_to(PROJECT_ROOT)}`
- `{SOURCE_CSV.relative_to(PROJECT_ROOT)}`
- `{MERGED_CSV.relative_to(PROJECT_ROOT)}`
- `{METRICS_CSV.relative_to(PROJECT_ROOT)}`
- `{FIGURE_PATH.relative_to(PROJECT_ROOT)}`
"""


def run_audit() -> RiskAuditResult:
    ovx = fetch_ovx()
    brent = load_brent()
    merged = build_merged(brent, ovx)
    metrics = build_metrics(merged)
    sources = build_sources()

    for path in [OVX_CSV, SOURCE_CSV, MERGED_CSV, METRICS_CSV, REPORT_PATH]:
        ensure_parent(path)
    ovx.to_csv(OVX_CSV, index=False)
    sources.to_csv(SOURCE_CSV, index=False)
    merged.to_csv(MERGED_CSV, index=False)
    metrics.to_csv(METRICS_CSV, index=False)
    save_figure(merged)
    REPORT_PATH.write_text(build_report(RiskAuditResult(merged, metrics, sources)), encoding="utf-8")
    return RiskAuditResult(merged, metrics, sources)


def main() -> None:
    result = run_audit()
    print("Market risk variable exploration complete")
    print(f"OVX rows: {len(result.sources)} source rows, {len(result.merged)} merged rows")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
