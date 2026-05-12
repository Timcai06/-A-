"""Stage 9 lagged geopolitical-risk index audit.

The GPR index is news-based, so this module treats it as an external audit
signal rather than as a same-day fitting input.  The main identification guard
is simple: only lagged monthly GPR is used when relating risk to Brent returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO
from typing import Any
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parent
from src.models import dynamic_short_term as dynamic


GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
GPR_PAGE = "https://www.matteoiacoviello.com/gpr.htm"

RISK_DIR = PROJECT_ROOT / "data" / "external" / "risk"
GPR_MONTHLY_CSV = RISK_DIR / "地缘政治风险指数_月度.csv"
AUDIT_SAMPLE_CSV = RISK_DIR / "地缘政治风险与油价_滞后审计样本.csv"
SOURCE_TABLE_CSV = RISK_DIR / "地缘风险数据来源表.csv"

OUTPUT_DIR = PROJECT_ROOT / "output" / "risk"
AUDIT_METRICS_CSV = OUTPUT_DIR / "滞后风险指数审计指标.csv"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "滞后地缘风险指数审计报告.md"

EVENT_FIGURE = PROJECT_ROOT / "figures" / "gpr_event_window_audit.png"
SCATTER_FIGURE = PROJECT_ROOT / "figures" / "lagged_gpr_brent_return.png"


@dataclass(frozen=True)
class RegressionResult:
    alpha: float
    beta: float
    r2: float
    n_obs: int


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "mathmodel-oil-risk-audit/1.0"})
    with urlopen(request, timeout=40) as response:
        return response.read()


def normalize_gpr(raw: bytes) -> pd.DataFrame:
    frame = pd.read_excel(BytesIO(raw))
    keep_columns = {
        "month": "月份",
        "GPR": "GPR_全球综合",
        "GPRT": "GPRT_威胁",
        "GPRA": "GPRA_行动",
        "GPRC_SAU": "GPRC_SAU_沙特",
        "GPRC_RUS": "GPRC_RUS_俄罗斯",
        "GPRC_USA": "GPRC_USA_美国",
        "GPRC_CHN": "GPRC_CHN_中国",
    }
    missing = [column for column in keep_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"GPR file is missing expected columns: {missing}")

    normalized = frame[list(keep_columns)].rename(columns=keep_columns).copy()
    normalized["月份"] = pd.to_datetime(normalized["月份"], errors="coerce")
    numeric_columns = [column for column in normalized.columns if column != "月份"]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(subset=["月份", "GPR_全球综合"]).sort_values("月份").reset_index(drop=True)
    normalized["数据下载日期"] = date.today().isoformat()
    normalized["来源URL"] = GPR_URL

    base = normalized[(normalized["月份"] >= "1985-01-01") & (normalized["月份"] <= "2025-12-01")]
    gpr_mean = float(base["GPR_全球综合"].mean())
    gpr_std = float(base["GPR_全球综合"].std(ddof=0))
    normalized["GPR_历史标准分"] = (normalized["GPR_全球综合"] - gpr_mean) / gpr_std
    normalized["GPR_历史分位数"] = normalized["GPR_全球综合"].rank(pct=True)
    return normalized


def load_gpr_monthly(refresh: bool = False) -> pd.DataFrame:
    if GPR_MONTHLY_CSV.exists() and not refresh:
        return pd.read_csv(GPR_MONTHLY_CSV, parse_dates=["月份"])

    raw = fetch_bytes(GPR_URL)
    normalized = normalize_gpr(raw)
    ensure_parent(GPR_MONTHLY_CSV)
    normalized.to_csv(GPR_MONTHLY_CSV, index=False)
    write_source_table()
    return normalized


def write_source_table() -> None:
    rows = [
        {
            "来源": "Caldara-Iacoviello GPR Index",
            "数据集": "Monthly Geopolitical Risk Index",
            "URL": GPR_URL,
            "官网说明页": GPR_PAGE,
            "频率": "月度",
            "当前状态": "已自动下载并标准化",
            "模型用途": "作为地缘风险溢价的外部审计；只做滞后检验，不作为同月拟合输入。",
        },
        {
            "来源": "Caldara and Iacoviello (2022)",
            "数据集": "Measuring Geopolitical Risk",
            "URL": "https://www.aeaweb.org/articles?id=10.1257/aer.20191823",
            "官网说明页": GPR_PAGE,
            "频率": "论文/方法",
            "当前状态": "已记录为论文引用依据",
            "模型用途": "说明 GPR 是新闻文章占比构造的风险指数，使用时必须处理内生性问题。",
        },
    ]
    ensure_parent(SOURCE_TABLE_CSV)
    pd.DataFrame(rows).to_csv(SOURCE_TABLE_CSV, index=False)


def load_brent_monthly() -> pd.DataFrame:
    config = dynamic.load_yaml(dynamic.BASE_CONFIG_PATH)
    daily_path = PROJECT_ROOT / config["paths"]["processed_daily_csv"]
    daily = pd.read_csv(daily_path, parse_dates=["trade_date"]).sort_values("trade_date")
    daily["月份"] = daily["trade_date"].values.astype("datetime64[M]")

    rows: list[dict[str, Any]] = []
    for month, group in daily.groupby("月份", sort=True):
        rows.append(
            {
                "月份": pd.Timestamp(month),
                "月末收盘价": float(group["close_price"].iloc[-1]),
                "月均收盘价": float(group["close_price"].mean()),
                "月内最高价": float(group["high_price"].max()),
                "月内最低价": float(group["low_price"].min()),
                "月内交易日数": int(len(group)),
                "月度实现波动率": float(group["log_return"].std(ddof=0) * np.sqrt(len(group))),
                "月内最大单日绝对收益": float(group["return_pct"].abs().max()),
            }
        )

    monthly = pd.DataFrame(rows).sort_values("月份").reset_index(drop=True)
    monthly["油价月度对数收益"] = np.log(monthly["月末收盘价"] / monthly["月末收盘价"].shift(1))
    monthly["油价月度收益率"] = monthly["月末收盘价"].pct_change()
    return monthly


def build_audit_sample(gpr: pd.DataFrame, oil: pd.DataFrame) -> pd.DataFrame:
    sample = oil.merge(gpr, on="月份", how="left")
    for column in ["GPR_全球综合", "GPRT_威胁", "GPRA_行动", "GPRC_SAU_沙特", "GPR_历史标准分"]:
        sample[f"{column}_滞后1月"] = sample[column].shift(1)
        sample[f"{column}_滞后2月"] = sample[column].shift(2)
    sample["是否冲突窗口月"] = sample["月份"].between(pd.Timestamp("2026-03-01"), pd.Timestamp("2026-05-01"))
    ensure_parent(AUDIT_SAMPLE_CSV)
    sample.to_csv(AUDIT_SAMPLE_CSV, index=False)
    return sample


def corr(frame: pd.DataFrame, x: str, y: str) -> float:
    subset = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 3:
        return float("nan")
    return float(subset[x].corr(subset[y]))


def ols(y: pd.Series, x: pd.Series) -> RegressionResult:
    data = pd.DataFrame({"y": y, "x": x}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3:
        return RegressionResult(float("nan"), float("nan"), float("nan"), len(data))
    x_matrix = np.column_stack([np.ones(len(data)), data["x"].to_numpy(dtype=float)])
    beta = np.linalg.lstsq(x_matrix, data["y"].to_numpy(dtype=float), rcond=None)[0]
    pred = x_matrix @ beta
    ss_res = float(((data["y"] - pred) ** 2).sum())
    ss_tot = float(((data["y"] - data["y"].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return RegressionResult(alpha=float(beta[0]), beta=float(beta[1]), r2=r2, n_obs=len(data))


def build_metrics(sample: pd.DataFrame, gpr: pd.DataFrame) -> pd.DataFrame:
    audit = sample[(sample["月份"] >= "2017-10-01") & (sample["月份"] <= "2026-04-01")].copy()
    reg_return = ols(audit["油价月度对数收益"], audit["GPR_历史标准分_滞后1月"])
    reg_vol = ols(audit["月度实现波动率"], audit["GPR_历史标准分_滞后1月"])
    historical = gpr[(gpr["月份"] >= "1985-01-01") & (gpr["月份"] <= "2025-12-01")]

    def percentile(value: float) -> float:
        return float((historical["GPR_全球综合"] <= value).mean())

    march_2026 = gpr[gpr["月份"] == "2026-03-01"].iloc[0]
    april_2026 = gpr[gpr["月份"] == "2026-04-01"].iloc[0]
    sample_march = sample[sample["月份"] == "2026-03-01"].iloc[0]
    sample_april = sample[sample["月份"] == "2026-04-01"].iloc[0]

    rows = [
        ("同期GPR与本月油价月收益相关系数", corr(audit, "GPR_历史标准分", "油价月度对数收益"), "描述性同步关系，不用于预测。"),
        ("滞后1月GPR与本月油价月收益相关系数", corr(audit, "GPR_历史标准分_滞后1月", "油价月度对数收益"), "用于避免同月新闻-价格反馈循环的核心检验。"),
        ("滞后2月GPR与本月油价月收益相关系数", corr(audit, "GPR_历史标准分_滞后2月", "油价月度对数收益"), "滞后稳健性参考。"),
        ("滞后1月GPR与本月实现波动率相关系数", corr(audit, "GPR_历史标准分_滞后1月", "月度实现波动率"), "风险指数更可能解释波动和尾部风险。"),
        ("滞后1月GPR收益回归beta", reg_return.beta, f"OLS: return_t = a + b GPR_z(t-1), n={reg_return.n_obs}, R2={reg_return.r2:.3f}。"),
        ("滞后1月GPR波动回归beta", reg_vol.beta, f"OLS: vol_t = a + b GPR_z(t-1), n={reg_vol.n_obs}, R2={reg_vol.r2:.3f}。"),
        ("2026年3月GPR历史分位数", percentile(float(march_2026["GPR_全球综合"])), "冲突爆发月风险读数在1985-2025历史样本中的位置。"),
        ("2026年4月GPR历史分位数", percentile(float(april_2026["GPR_全球综合"])), "冲突后续月风险读数在1985-2025历史样本中的位置。"),
        ("2026年3月可用滞后GPR", float(sample_march["GPR_全球综合_滞后1月"]), "3月预测只能使用2月GPR，不能使用3月同步新闻指数。"),
        ("2026年4月可用滞后GPR", float(sample_april["GPR_全球综合_滞后1月"]), "4月预测可使用3月已实现GPR，适合解释风险延续。"),
    ]
    metrics = pd.DataFrame(rows, columns=["指标", "数值", "解释"])
    ensure_parent(AUDIT_METRICS_CSV)
    metrics.to_csv(AUDIT_METRICS_CSV, index=False)
    return metrics


def save_figures(sample: pd.DataFrame) -> None:
    dynamic.configure_plot_style()
    recent = sample[(sample["月份"] >= "2025-01-01") & (sample["月份"] <= "2026-05-01")].copy()

    ensure_parent(EVENT_FIGURE)
    fig, ax1 = plt.subplots(figsize=(11.2, 6.0))
    ax1.plot(recent["月份"], recent["月末收盘价"], color="#2563eb", linewidth=2.2, marker="o", label="布伦特月末收盘价")
    ax1.set_ylabel("美元/桶", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.axvspan(pd.Timestamp("2026-03-01"), pd.Timestamp("2026-05-01"), color="#fee2e2", alpha=0.42, label="冲突窗口月")

    ax2 = ax1.twinx()
    ax2.plot(recent["月份"], recent["GPR_全球综合"], color="#dc2626", linewidth=2.0, marker="s", label="GPR同步读数")
    ax2.plot(
        recent["月份"],
        recent["GPR_全球综合_滞后1月"],
        color="#f97316",
        linewidth=1.8,
        linestyle="--",
        marker="^",
        label="GPR滞后1月可用读数",
    )
    ax2.set_ylabel("GPR指数", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", ncol=2)
    ax1.set_title("地缘风险指数审计：同步风险与滞后可用风险")
    ax1.set_xlabel("月份")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(EVENT_FIGURE, dpi=220)
    plt.close(fig)

    scatter = sample[(sample["月份"] >= "2017-10-01") & (sample["月份"] <= "2026-04-01")].dropna(
        subset=["GPR_历史标准分_滞后1月", "油价月度对数收益"]
    )
    ensure_parent(SCATTER_FIGURE)
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    colors = np.where(scatter["是否冲突窗口月"], "#dc2626", "#2563eb")
    ax.scatter(scatter["GPR_历史标准分_滞后1月"], scatter["油价月度对数收益"] * 100, c=colors, alpha=0.78)
    reg = ols(scatter["油价月度对数收益"] * 100, scatter["GPR_历史标准分_滞后1月"])
    x_vals = np.linspace(scatter["GPR_历史标准分_滞后1月"].min(), scatter["GPR_历史标准分_滞后1月"].max(), 100)
    ax.plot(x_vals, reg.alpha + reg.beta * x_vals, color="#111827", linewidth=1.8, label=f"线性趋势 R²={reg.r2:.2f}")
    ax.axhline(0, color="#6b7280", linewidth=1.0, linestyle="--")
    ax.set_title("滞后1月GPR与布伦特月度收益")
    ax.set_xlabel("GPR历史标准分（滞后1月）")
    ax.set_ylabel("布伦特月度对数收益（%）")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(SCATTER_FIGURE, dpi=220)
    plt.close(fig)


def fmt_metric(metrics: pd.DataFrame, name: str, percent: bool = False) -> str:
    value = float(metrics.loc[metrics["指标"] == name, "数值"].iloc[0])
    return f"{value:.1%}" if percent else f"{value:.3f}"


def build_report(sample: pd.DataFrame, metrics: pd.DataFrame) -> str:
    march = sample[sample["月份"] == "2026-03-01"].iloc[0]
    april = sample[sample["月份"] == "2026-04-01"].iloc[0]
    metric_rows = "\n".join(
        f"| {row['指标']} | {float(row['数值']):.4f} | {row['解释']} |" for row in metrics.to_dict("records")
    )

    return f"""# 滞后地缘风险指数审计报告

## 运行结论

本审计接入 Caldara-Iacoviello 官方 GPR 月度指数，用来审计模型中的地缘风险溢价是否有外部事实支撑。这里没有抓取新闻原文，也没有把同月新闻指数直接塞进短期模型拟合；核心处理是使用 \(t-1\) 月 GPR 解释 \(t\) 月油价收益和波动，从机制上降低“油价上涨导致新闻恐慌、新闻恐慌再解释油价”的内生性风险。

从结果看，同步 GPR 与冲突爆发月油价冲击高度同向，但这只能作为事后事实审计。真正可用于预测或解释下一期的变量是滞后 GPR：2026 年 3 月的模型不能使用 3 月 GPR，只能使用 2 月 GPR；2026 年 4 月才可以使用 3 月已经实现的 GPR 去解释风险延续。

## 关键指标

| 指标 | 数值 | 解释 |
|---|---:|---|
{metric_rows}

## 冲突窗口读数

| 月份 | 布伦特月末收盘价 | 油价月度收益率 | 同月GPR | 滞后1月GPR |
|---|---:|---:|---:|---:|
| 2026-03 | {march['月末收盘价']:.2f} | {march['油价月度收益率']:.2%} | {march['GPR_全球综合']:.2f} | {march['GPR_全球综合_滞后1月']:.2f} |
| 2026-04 | {april['月末收盘价']:.2f} | {april['油价月度收益率']:.2%} | {april['GPR_全球综合']:.2f} | {april['GPR_全球综合_滞后1月']:.2f} |

## 对论文写法的约束

1. 可以写：GPR 官方指数显示 2026 年 3 月和 4 月地缘风险读数显著升高，因此模型中设置风险溢价和恐慌衰减具有外部事实基础。
2. 不能写：本文使用 2026 年 3 月 GPR 准确预测了 2026 年 3 月油价上涨。这个写法会触发内生性循环质疑。
3. 更稳妥的写法：GPR 作为外部审计变量，不参与短期窗口同月拟合；若作为预测变量，只采用滞后一期读数。

## 输出产物

- `{GPR_MONTHLY_CSV.relative_to(PROJECT_ROOT)}`
- `{AUDIT_SAMPLE_CSV.relative_to(PROJECT_ROOT)}`
- `{AUDIT_METRICS_CSV.relative_to(PROJECT_ROOT)}`
- `{SOURCE_TABLE_CSV.relative_to(PROJECT_ROOT)}`
- `{EVENT_FIGURE.relative_to(PROJECT_ROOT)}`
- `{SCATTER_FIGURE.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    gpr = load_gpr_monthly(refresh=not GPR_MONTHLY_CSV.exists())
    oil = load_brent_monthly()
    sample = build_audit_sample(gpr, oil)
    metrics = build_metrics(sample, gpr)
    save_figures(sample)
    ensure_parent(REPORT_PATH)
    REPORT_PATH.write_text(build_report(sample, metrics), encoding="utf-8")
    write_source_table()

    print("Lagged GPR risk audit complete")
    print(f"GPR monthly: {GPR_MONTHLY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Audit sample: {AUDIT_SAMPLE_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Metrics: {AUDIT_METRICS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
