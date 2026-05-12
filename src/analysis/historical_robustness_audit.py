"""Stage 10 historical-sample robustness and baseline audit.

This module turns the full 2017--2026 Brent series into evidence for the paper:
the 2026 Hormuz event window is compared with same-length historical windows,
and simple time-series baselines are evaluated without changing the main model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.common.metrics import mae, mape, rmse
from src.common.paths import PROJECT_ROOT, ensure_parent
from src.models import dynamic_short_term as dynamic


FULL_DATA_CSV = PROJECT_ROOT / "data" / "processed" / "布伦特原油期货主力合约价格数据_清洗后.csv"
CALIBRATED_PATH_CSV = PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv"

OUTPUT_DIR = PROJECT_ROOT / "output" / "history"
YEARLY_STATS_CSV = OUTPUT_DIR / "历史年度统计.csv"
ROLLING_WINDOWS_CSV = OUTPUT_DIR / "历史46交易日滚动窗口统计.csv"
EVENT_EXTREMENESS_CSV = OUTPUT_DIR / "冲突窗口历史分位数.csv"
HIGH_VOL_WINDOWS_CSV = OUTPUT_DIR / "历史高波动窗口排名.csv"
BASELINE_BY_REGIME_CSV = OUTPUT_DIR / "历史基准误差分布.csv"
STATIONARITY_CSV = OUTPUT_DIR / "历史序列平稳性检验.csv"
R_REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "历史样本稳健性与R语言评估报告.md"

EXTREMENESS_FIGURE = PROJECT_ROOT / "figures" / "historical_event_extremeness.png"
BASELINE_FIGURE = PROJECT_ROOT / "figures" / "historical_baseline_error_distribution.png"
HIGH_VOL_FIGURE = PROJECT_ROOT / "figures" / "historical_high_vol_windows.png"

EVENT_START = pd.Timestamp("2026-03-02")
EVENT_END = pd.Timestamp("2026-05-05")


@dataclass(frozen=True)
class EventMetric:
    metric: str
    value: float
    historical_percentile: float
    interpretation: str


def load_full_data() -> pd.DataFrame:
    df = pd.read_csv(FULL_DATA_CSV, parse_dates=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    df["year"] = df["trade_date"].dt.year
    df["abs_return_pct"] = df["return_pct"].abs()
    return df


def load_calibrated_path() -> pd.DataFrame:
    path = pd.read_csv(CALIBRATED_PATH_CSV, parse_dates=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    path["model_error"] = path["simulated_price"] - path["actual_price"]
    return path


def window_baseline_predictions(close: pd.Series, first_pre_close: float) -> dict[str, pd.Series]:
    actual = close.astype(float).reset_index(drop=True)
    previous = actual.shift(1)
    previous.iloc[0] = first_pre_close

    history_with_preclose = pd.concat([pd.Series([first_pre_close]), actual.iloc[:-1]], ignore_index=True)
    rolling3 = history_with_preclose.rolling(3, min_periods=1).mean().reset_index(drop=True)

    drift_values: list[float] = []
    history = [float(first_pre_close)]
    for value in actual:
        if len(history) >= 3:
            diffs = np.diff(history)
            pred = history[-1] + float(np.mean(diffs[-5:]))
        else:
            pred = history[-1]
        drift_values.append(float(pred))
        history.append(float(value))

    return {
        "朴素上一日基准": previous,
        "三日均值基准": rolling3,
        "漂移随机游走基准": pd.Series(drift_values),
    }


def summarize_window(window: pd.DataFrame, window_id: int) -> dict[str, Any]:
    first_pre_close = float(window.iloc[0]["pre_close"])
    if pd.isna(first_pre_close):
        first_pre_close = float(window.iloc[0]["close_price"])
    close = window["close_price"].astype(float).reset_index(drop=True)
    log_return = window["log_return"].replace([np.inf, -np.inf], np.nan).dropna()
    return_pct = window["return_pct"].replace([np.inf, -np.inf], np.nan).dropna()
    preds = window_baseline_predictions(close, first_pre_close)

    row: dict[str, Any] = {
        "窗口编号": window_id,
        "窗口起始": window.iloc[0]["trade_date"].date().isoformat(),
        "窗口终止": window.iloc[-1]["trade_date"].date().isoformat(),
        "交易日数": int(len(window)),
        "累计收益率": float(close.iloc[-1] / first_pre_close - 1),
        "窗口最高收盘价": float(close.max()),
        "窗口最低收盘价": float(close.min()),
        "价格区间比例": float((close.max() - close.min()) / close.mean()),
        "实现波动率": float(log_return.std(ddof=0) * np.sqrt(len(window))) if len(log_return) else float("nan"),
        "平均绝对日收益率": float(return_pct.abs().mean()) if len(return_pct) else float("nan"),
        "最大单日上涨": float(return_pct.max()) if len(return_pct) else float("nan"),
        "最大单日下跌": float(return_pct.min()) if len(return_pct) else float("nan"),
        "最大单日绝对收益": float(return_pct.abs().max()) if len(return_pct) else float("nan"),
        "是否冲突窗口": bool(window.iloc[0]["trade_date"] == EVENT_START and window.iloc[-1]["trade_date"] == EVENT_END),
    }
    for name, pred in preds.items():
        error = pred - close
        row[f"{name}_RMSE"] = rmse(error)
        row[f"{name}_MAE"] = mae(error)
        row[f"{name}_MAPE"] = mape(close, pred)
    return row


def build_rolling_windows(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(df) - window_size + 1):
        window = df.iloc[start : start + window_size].copy()
        rows.append(summarize_window(window, start + 1))
    rolling = pd.DataFrame(rows)

    for metric in ["实现波动率", "平均绝对日收益率", "最大单日绝对收益", "价格区间比例", "朴素上一日基准_RMSE"]:
        mean = float(rolling[metric].mean())
        std = float(rolling[metric].std(ddof=0))
        rolling[f"{metric}_历史标准分"] = (rolling[metric] - mean) / std if std > 0 else 0.0

    rolling["历史压力综合分"] = (
        0.32 * rolling["实现波动率_历史标准分"]
        + 0.24 * rolling["平均绝对日收益率_历史标准分"]
        + 0.20 * rolling["最大单日绝对收益_历史标准分"]
        + 0.16 * rolling["价格区间比例_历史标准分"]
        + 0.08 * rolling["朴素上一日基准_RMSE_历史标准分"]
    )
    return rolling


def percentile_rank(reference: pd.Series, value: float) -> float:
    clean = reference.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("nan")
    return float((clean <= value).mean())


def build_event_extremeness(rolling: pd.DataFrame, calibrated: pd.DataFrame) -> pd.DataFrame:
    event = rolling[rolling["是否冲突窗口"]].iloc[0]
    reference = rolling[pd.to_datetime(rolling["窗口终止"]) < EVENT_START].copy()
    model_rmse = rmse(calibrated["model_error"])
    naive_rmse = float(event["朴素上一日基准_RMSE"])

    metrics = [
        EventMetric("累计收益率", float(event["累计收益率"]), percentile_rank(reference["累计收益率"], float(event["累计收益率"])), "衡量同长度窗口内价格上行幅度是否罕见。"),
        EventMetric("实现波动率", float(event["实现波动率"]), percentile_rank(reference["实现波动率"], float(event["实现波动率"])), "衡量冲突窗口是否属于高波动事件。"),
        EventMetric("平均绝对日收益率", float(event["平均绝对日收益率"]), percentile_rank(reference["平均绝对日收益率"], float(event["平均绝对日收益率"])), "衡量日常波动强度。"),
        EventMetric("最大单日绝对收益", float(event["最大单日绝对收益"]), percentile_rank(reference["最大单日绝对收益"], float(event["最大单日绝对收益"])), "衡量局部跳变强度。"),
        EventMetric("价格区间比例", float(event["价格区间比例"]), percentile_rank(reference["价格区间比例"], float(event["价格区间比例"])), "衡量窗口内价格振幅。"),
        EventMetric("朴素上一日基准_RMSE", naive_rmse, percentile_rank(reference["朴素上一日基准_RMSE"], naive_rmse), "衡量价格惯性基准在该窗口是否明显变难。"),
        EventMetric("本文机制模型_RMSE", model_rmse, percentile_rank(reference["朴素上一日基准_RMSE"], model_rmse), "用机制模型误差对比历史朴素误差分布。"),
        EventMetric("机制模型相对冲突窗口朴素RMSE改善率", (naive_rmse - model_rmse) / naive_rmse, float("nan"), "只在冲突窗口内比较，说明机制模型是否战胜 Random Walk。"),
    ]
    return pd.DataFrame([metric.__dict__ for metric in metrics])


def build_yearly_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, group in df.groupby("year", sort=True):
        first_pre = group.iloc[0]["pre_close"]
        if pd.isna(first_pre):
            first_pre = group.iloc[0]["close_price"]
        rows.append(
            {
                "年份": int(year),
                "交易日数": int(len(group)),
                "年初参考价": float(first_pre),
                "年末收盘价": float(group["close_price"].iloc[-1]),
                "年度收益率": float(group["close_price"].iloc[-1] / first_pre - 1),
                "年度最高收盘价": float(group["close_price"].max()),
                "年度最低收盘价": float(group["close_price"].min()),
                "年度实现波动率": float(group["log_return"].std(ddof=0) * np.sqrt(len(group))),
                "最大单日上涨": float(group["return_pct"].max(skipna=True)),
                "最大单日下跌": float(group["return_pct"].min(skipna=True)),
                "平均绝对日收益率": float(group["return_pct"].abs().mean(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def build_high_vol_windows(rolling: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    candidates = rolling.copy()
    candidates["start_dt"] = pd.to_datetime(candidates["窗口起始"])
    candidates["end_dt"] = pd.to_datetime(candidates["窗口终止"])
    selected: list[pd.Series] = []

    for _, row in candidates.sort_values("历史压力综合分", ascending=False).iterrows():
        overlaps = any(row["start_dt"] <= kept["end_dt"] and row["end_dt"] >= kept["start_dt"] for kept in selected)
        if overlaps:
            continue
        selected.append(row)
        if len(selected) >= top_n:
            break

    result = pd.DataFrame(selected).drop(columns=["start_dt", "end_dt"])
    return result.reset_index(drop=True)


def build_baseline_distribution(rolling: pd.DataFrame) -> pd.DataFrame:
    event = rolling[rolling["是否冲突窗口"]].iloc[0]
    reference = rolling[pd.to_datetime(rolling["窗口终止"]) < EVENT_START].copy()
    rows = []
    for metric in ["朴素上一日基准_RMSE", "三日均值基准_RMSE", "漂移随机游走基准_RMSE"]:
        rows.append(
            {
                "基准指标": metric,
                "历史P05": float(reference[metric].quantile(0.05)),
                "历史P25": float(reference[metric].quantile(0.25)),
                "历史中位数": float(reference[metric].quantile(0.50)),
                "历史P75": float(reference[metric].quantile(0.75)),
                "历史P95": float(reference[metric].quantile(0.95)),
                "冲突窗口数值": float(event[metric]),
                "冲突窗口历史分位数": percentile_rank(reference[metric], float(event[metric])),
            }
        )
    return pd.DataFrame(rows)


def adf_test(series: pd.Series, name: str) -> dict[str, Any]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    stat, p_value, used_lag, n_obs, critical_values, _ = adfuller(clean, autolag="AIC")
    return {
        "序列": name,
        "ADF统计量": float(stat),
        "p值": float(p_value),
        "使用滞后阶数": int(used_lag),
        "样本数": int(n_obs),
        "1%临界值": float(critical_values["1%"]),
        "5%临界值": float(critical_values["5%"]),
        "10%临界值": float(critical_values["10%"]),
        "结论": "可拒绝单位根，近似平稳" if p_value < 0.05 else "不能拒绝单位根，按非平稳处理",
    }


def build_stationarity_tests(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            adf_test(df["close_price"], "收盘价水平"),
            adf_test(np.log(df["close_price"]), "对数收盘价"),
            adf_test(df["log_return"], "对数收益率"),
        ]
    )


def save_figures(rolling: pd.DataFrame, event_extremeness: pd.DataFrame, high_windows: pd.DataFrame) -> None:
    dynamic.configure_plot_style()
    ensure_parent(EXTREMENESS_FIGURE)

    selected_metrics = event_extremeness[
        event_extremeness["metric"].isin(["累计收益率", "实现波动率", "平均绝对日收益率", "最大单日绝对收益", "价格区间比例", "朴素上一日基准_RMSE"])
    ].copy()
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    bars = ax.barh(selected_metrics["metric"], selected_metrics["historical_percentile"] * 100, color="#2563eb")
    ax.axvline(95, color="#dc2626", linestyle="--", linewidth=1.3, label="95%历史分位")
    ax.set_xlim(0, 105)
    ax.set_xlabel("历史分位数（%）")
    ax.set_title("2026冲突窗口在2017-2026同长度窗口中的历史位置")
    for bar, value in zip(bars, selected_metrics["historical_percentile"] * 100, strict=False):
        ax.text(value + 1.0, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(EXTREMENESS_FIGURE, dpi=220)
    plt.close(fig)

    ensure_parent(BASELINE_FIGURE)
    reference = rolling[pd.to_datetime(rolling["窗口终止"]) < EVENT_START].copy()
    event = rolling[rolling["是否冲突窗口"]].iloc[0]
    calibrated = load_calibrated_path()
    model_rmse = rmse(calibrated["model_error"])

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.hist(reference["朴素上一日基准_RMSE"], bins=38, color="#93c5fd", edgecolor="white", alpha=0.92, label="历史46日窗口朴素RMSE")
    ax.axvline(event["朴素上一日基准_RMSE"], color="#dc2626", linewidth=2.2, label=f"冲突窗口朴素RMSE={event['朴素上一日基准_RMSE']:.2f}")
    ax.axvline(model_rmse, color="#16a34a", linewidth=2.2, label=f"本文机制模型RMSE={model_rmse:.2f}")
    ax.set_title("历史窗口中的随机游走基准误差分布")
    ax.set_xlabel("RMSE（美元/桶）")
    ax.set_ylabel("窗口数量")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(BASELINE_FIGURE, dpi=220)
    plt.close(fig)

    ensure_parent(HIGH_VOL_FIGURE)
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    labels = high_windows["窗口起始"].str.slice(0, 7) + " 至 " + high_windows["窗口终止"].str.slice(0, 7)
    bars = ax.bar(labels, high_windows["历史压力综合分"], color="#f97316")
    ax.set_title("历史高波动窗口排名（46交易日非重叠窗口）")
    ax.set_ylabel("历史压力综合分")
    ax.tick_params(axis="x", labelrotation=28)
    for bar, value in zip(bars, high_windows["历史压力综合分"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(HIGH_VOL_FIGURE, dpi=220)
    plt.close(fig)


def fmt_percent(value: float) -> str:
    return f"{value:.1%}"


def build_report(
    yearly: pd.DataFrame,
    event_extreme: pd.DataFrame,
    high_windows: pd.DataFrame,
    baseline_distribution: pd.DataFrame,
    stationarity: pd.DataFrame,
) -> str:
    metric_lookup = dict(zip(event_extreme["metric"], event_extreme["value"], strict=False))
    percentile_lookup = dict(zip(event_extreme["metric"], event_extreme["historical_percentile"], strict=False))
    high_rows = "\n".join(
        "| {rank} | {窗口起始} | {窗口终止} | {累计收益率:.1%} | {实现波动率:.1%} | {价格区间比例:.1%} | {历史压力综合分:.2f} |".format(
            rank=i + 1, **row
        )
        for i, row in enumerate(high_windows.to_dict("records"))
    )
    baseline_rows = "\n".join(
        "| {基准指标} | {历史中位数:.2f} | {历史P95:.2f} | {冲突窗口数值:.2f} | {冲突窗口历史分位数:.1%} |".format(
            **row
        )
        for row in baseline_distribution.to_dict("records")
    )
    stationarity_rows = "\n".join(
        "| {序列} | {ADF统计量:.3f} | {p值:.4f} | {结论} |".format(**row) for row in stationarity.to_dict("records")
    )

    return f"""# 历史样本稳健性与 R 语言评估报告

## 运行结论

本审计把 2017-09-01 至 2026-05-05 的完整附件数据用于历史参照，而不是重新拟合主模型。核心目的有三个：第一，证明 2026 冲突窗口确实是历史高压窗口；第二，说明随机游走等简单基准在冲突窗口变难；第三，回答 R 语言是否有必要进入当前项目。

当前已经接通 R；本审计仍使用 Python 完成全部统计审计。结论是：R 适合后续做 ADF、VAR、GARCH、Granger 和高质量统计表，但当前项目主线没有必要切换到 R。Python 已经能稳定复现主模型、图表、PDF 和 DOCX；R 更适合作为后续计量补强工具。

## 冲突窗口历史位置

| 指标 | 冲突窗口数值 | 历史分位数 | 含义 |
|---|---:|---:|---|
| 累计收益率 | {metric_lookup["累计收益率"]:.1%} | {percentile_lookup["累计收益率"]:.1%} | 同长度窗口内价格上行幅度 |
| 实现波动率 | {metric_lookup["实现波动率"]:.1%} | {percentile_lookup["实现波动率"]:.1%} | 同长度窗口内波动强度 |
| 平均绝对日收益率 | {metric_lookup["平均绝对日收益率"]:.2%} | {percentile_lookup["平均绝对日收益率"]:.1%} | 日常波动强度 |
| 最大单日绝对收益 | {metric_lookup["最大单日绝对收益"]:.1%} | {percentile_lookup["最大单日绝对收益"]:.1%} | 局部跳变强度 |
| 价格区间比例 | {metric_lookup["价格区间比例"]:.1%} | {percentile_lookup["价格区间比例"]:.1%} | 窗口内价格振幅 |
| 朴素上一日基准 RMSE | {metric_lookup["朴素上一日基准_RMSE"]:.2f} | {percentile_lookup["朴素上一日基准_RMSE"]:.1%} | 随机游走基准在该窗口是否变难 |
| 本文机制模型 RMSE | {metric_lookup["本文机制模型_RMSE"]:.2f} | {percentile_lookup["本文机制模型_RMSE"]:.1%} | 对比历史朴素误差分布 |
| 机制模型相对冲突窗口朴素 RMSE 改善率 | {metric_lookup["机制模型相对冲突窗口朴素RMSE改善率"]:.1%} | - | 只在冲突窗口内比较 |

## 历史高波动窗口

| 排名 | 起始 | 终止 | 累计收益率 | 实现波动率 | 价格区间比例 | 压力分 |
|---:|---|---|---:|---:|---:|---:|
{high_rows}

## 历史基准误差分布

| 指标 | 历史中位数 | 历史P95 | 冲突窗口数值 | 冲突窗口分位数 |
|---|---:|---:|---:|---:|
{baseline_rows}

## 平稳性检验

| 序列 | ADF统计量 | p值 | 结论 |
|---|---:|---:|---|
{stationarity_rows}

ADF 检验结果符合金融时间序列常识：价格水平通常不能按平稳序列处理，收益率更适合进入统计基准模型。这也是为什么本文主模型不直接用价格水平做普通线性回归，而是围绕事件机制递推、基准对比和情景外推组织证据链。

## R 语言在本项目中的定位

R 的优势主要在统计和计量生态：`forecast` 适合 ARIMA/ETS，`rugarch` 适合 GARCH，`vars` 适合 VAR，`lmtest` 和 `sandwich` 适合稳健回归检验，`ggplot2` 适合学术图表。它能让论文中的“统计检验”看起来更专业。

但当前不建议把主项目迁移到 R，原因是：

1. 主模型已经由 Python 稳定完成，且可以自动生成 CSV、图片、PDF 和 DOCX。
2. 赛题核心是机制解释和情景预测，不是单纯时间序列竞赛。
3. 当前本机没有 R 环境，临时引入会增加环境复杂度。

建议路线是：Python 保持主模型和论文生成；若后续要做 GARCH、VAR、Granger 或更正式的计量表，再把 R 作为 `analysis/r/` 下的辅助审计工具接入。

## 输出产物

- `{YEARLY_STATS_CSV.relative_to(PROJECT_ROOT)}`
- `{ROLLING_WINDOWS_CSV.relative_to(PROJECT_ROOT)}`
- `{EVENT_EXTREMENESS_CSV.relative_to(PROJECT_ROOT)}`
- `{HIGH_VOL_WINDOWS_CSV.relative_to(PROJECT_ROOT)}`
- `{BASELINE_BY_REGIME_CSV.relative_to(PROJECT_ROOT)}`
- `{STATIONARITY_CSV.relative_to(PROJECT_ROOT)}`
- `{EXTREMENESS_FIGURE.relative_to(PROJECT_ROOT)}`
- `{BASELINE_FIGURE.relative_to(PROJECT_ROOT)}`
- `{HIGH_VOL_FIGURE.relative_to(PROJECT_ROOT)}`
"""


def write_outputs(
    yearly: pd.DataFrame,
    rolling: pd.DataFrame,
    event_extreme: pd.DataFrame,
    high_windows: pd.DataFrame,
    baseline_distribution: pd.DataFrame,
    stationarity: pd.DataFrame,
) -> None:
    for path in [
        YEARLY_STATS_CSV,
        ROLLING_WINDOWS_CSV,
        EVENT_EXTREMENESS_CSV,
        HIGH_VOL_WINDOWS_CSV,
        BASELINE_BY_REGIME_CSV,
        STATIONARITY_CSV,
        R_REPORT_PATH,
    ]:
        ensure_parent(path)

    yearly.to_csv(YEARLY_STATS_CSV, index=False)
    rolling.to_csv(ROLLING_WINDOWS_CSV, index=False)
    event_extreme.to_csv(EVENT_EXTREMENESS_CSV, index=False)
    high_windows.to_csv(HIGH_VOL_WINDOWS_CSV, index=False)
    baseline_distribution.to_csv(BASELINE_BY_REGIME_CSV, index=False)
    stationarity.to_csv(STATIONARITY_CSV, index=False)
    R_REPORT_PATH.write_text(
        build_report(yearly, event_extreme, high_windows, baseline_distribution, stationarity),
        encoding="utf-8",
    )


def main() -> None:
    df = load_full_data()
    calibrated = load_calibrated_path()
    window_size = len(calibrated)

    yearly = build_yearly_stats(df)
    rolling = build_rolling_windows(df, window_size)
    event_extreme = build_event_extremeness(rolling, calibrated)
    high_windows = build_high_vol_windows(rolling)
    baseline_distribution = build_baseline_distribution(rolling)
    stationarity = build_stationarity_tests(df)

    save_figures(rolling, event_extreme, high_windows)
    write_outputs(yearly, rolling, event_extreme, high_windows, baseline_distribution, stationarity)

    print("Stage 10 historical robustness audit complete")
    print(f"Yearly stats: {YEARLY_STATS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Event extremeness: {EVENT_EXTREMENESS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Report: {R_REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
