"""Stage 1 data cleaning and exploratory figure generation.

This module reads the official Brent futures CSV, creates model-ready features,
exports cleaned datasets, renders the baseline figures, and writes a short
stage report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "base.yml"


@dataclass(frozen=True)
class Stage1Paths:
    raw_csv: Path
    processed_daily_csv: Path
    processed_event_window_csv: Path
    figures_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class EventWindow:
    start: pd.Timestamp
    end: pd.Timestamp


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(config: dict[str, Any]) -> Stage1Paths:
    paths = config["paths"]
    return Stage1Paths(
        raw_csv=PROJECT_ROOT / paths["raw_brent_csv"],
        processed_daily_csv=PROJECT_ROOT / paths["processed_daily_csv"],
        processed_event_window_csv=PROJECT_ROOT / paths["processed_event_window_csv"],
        figures_dir=PROJECT_ROOT / paths["figures_dir"],
        reports_dir=PROJECT_ROOT / paths["reports_dir"],
    )


def resolve_event_window(config: dict[str, Any]) -> EventWindow:
    window = config["event_window"]
    return EventWindow(
        start=pd.Timestamp(window["conflict_start"]),
        end=pd.Timestamp(window["data_end"]),
    )


def clean_brent_data(raw_csv: Path, event_window: EventWindow) -> pd.DataFrame:
    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv}")

    df = pd.read_csv(raw_csv, na_values=["NA", ""])
    df = df.rename(
        columns={
            "time": "trade_date",
            "preClose": "pre_close",
            "open": "open_price",
            "high": "high_price",
            "low": "low_price",
            "close": "close_price",
        }
    )

    required_columns = [
        "trade_date",
        "thscode",
        "pre_close",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="raise")
    numeric_columns = ["pre_close", "open_price", "high_price", "low_price", "close_price"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("trade_date").reset_index(drop=True)
    df["log_return"] = np.log(df["close_price"] / df["close_price"].shift(1))
    df["return_pct"] = df["close_price"].pct_change()
    df["volatility_7d"] = df["log_return"].rolling(window=7, min_periods=3).std()
    df["volatility_14d"] = df["log_return"].rolling(window=14, min_periods=5).std()
    df["volatility_30d"] = df["log_return"].rolling(window=30, min_periods=10).std()
    df["is_event_window"] = df["trade_date"].between(event_window.start, event_window.end)

    ordered_columns = [
        "trade_date",
        "thscode",
        "pre_close",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "log_return",
        "return_pct",
        "volatility_7d",
        "volatility_14d",
        "volatility_30d",
        "is_event_window",
    ]
    return df[ordered_columns]


def export_datasets(df: pd.DataFrame, paths: Stage1Paths) -> pd.DataFrame:
    paths.processed_daily_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.processed_event_window_csv.parent.mkdir(parents=True, exist_ok=True)

    event_df = df[df["is_event_window"]].copy()
    df.to_csv(paths.processed_daily_csv, index=False)
    event_df.to_csv(paths.processed_event_window_csv, index=False)
    return event_df


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def save_price_trend(df: pd.DataFrame, figures_dir: Path) -> Path:
    figure_path = figures_dir / "price_trend.png"
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(df["trade_date"], df["close_price"], color="#2563eb", linewidth=1.4, label="Close price")
    ax.set_title("Brent Futures Close Price Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/barrel")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    return figure_path


def save_event_window_price(event_df: pd.DataFrame, event_window: EventWindow, figures_dir: Path) -> Path:
    figure_path = figures_dir / "event_window_price.png"
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        event_df["trade_date"],
        event_df["close_price"],
        color="#dc2626",
        linewidth=1.8,
        marker="o",
        markersize=2.8,
        label="Event-window close price",
    )
    ax.axvline(event_window.start, color="#111827", linestyle="--", linewidth=1.0, label="Conflict start")
    ax.set_title("Brent Close Price During Event Window")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/barrel")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    return figure_path


def save_return_volatility(df: pd.DataFrame, figures_dir: Path) -> Path:
    figure_path = figures_dir / "return_volatility.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)

    axes[0].plot(df["trade_date"], df["return_pct"] * 100, color="#059669", linewidth=1.0)
    axes[0].axhline(0, color="#111827", linewidth=0.8)
    axes[0].set_title("Daily Return")
    axes[0].set_ylabel("Return (%)")

    axes[1].plot(df["trade_date"], df["volatility_7d"], color="#f97316", linewidth=1.2, label="7-day")
    axes[1].plot(df["trade_date"], df["volatility_14d"], color="#7c3aed", linewidth=1.2, label="14-day")
    axes[1].plot(df["trade_date"], df["volatility_30d"], color="#0f766e", linewidth=1.2, label="30-day")
    axes[1].set_title("Rolling Log-return Volatility")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Std. dev.")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    return figure_path


def generate_figures(df: pd.DataFrame, event_df: pd.DataFrame, event_window: EventWindow, paths: Stage1Paths) -> list[Path]:
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    return [
        save_price_trend(df, paths.figures_dir),
        save_event_window_price(event_df, event_window, paths.figures_dir),
        save_return_volatility(df, paths.figures_dir),
    ]


def fmt_float(value: float | np.floating[Any] | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def build_report(df: pd.DataFrame, event_df: pd.DataFrame, figures: list[Path], paths: Stage1Paths) -> str:
    missing_counts = df[
        ["pre_close", "open_price", "high_price", "low_price", "close_price", "log_return", "return_pct"]
    ].isna().sum()

    peak_row = event_df.loc[event_df["close_price"].idxmax()]
    high_peak_row = event_df.loc[event_df["high_price"].idxmax()]
    trough_row = event_df.loc[event_df["close_price"].idxmin()]
    ohlc_anomalies = find_ohlc_anomalies(df)

    figure_lines = "\n".join(f"- `{figure.relative_to(PROJECT_ROOT)}`" for figure in figures)
    missing_lines = "\n".join(f"- `{column}`: {count}" for column, count in missing_counts.items())

    report = f"""# 阶段 1 数据清洗与探索报告

## 运行结论

阶段 1 已完成基础数据清洗、冲突窗口截取和探索性图表生成。当前产物可作为阶段 2 传统供需基准模型和阶段 3 动态递推模型的真实价格输入。

## 数据范围

- 原始记录数：{len(df)}
- 日期范围：{df["trade_date"].min().date()} 至 {df["trade_date"].max().date()}
- 冲突窗口记录数：{len(event_df)}
- 冲突窗口范围：{event_df["trade_date"].min().date()} 至 {event_df["trade_date"].max().date()}

## 关键价格统计

| 指标 | 数值 |
|---|---:|
| 全样本最低收盘价 | {fmt_float(df["close_price"].min(), 2)} |
| 全样本最高收盘价 | {fmt_float(df["close_price"].max(), 2)} |
| 冲突窗口最低收盘价 | {fmt_float(event_df["close_price"].min(), 2)} |
| 冲突窗口最高收盘价 | {fmt_float(event_df["close_price"].max(), 2)} |
| 冲突窗口最高盘中价 | {fmt_float(event_df["high_price"].max(), 2)} |
| 冲突窗口均值 | {fmt_float(event_df["close_price"].mean(), 2)} |
| 冲突窗口末日收盘价 | {fmt_float(event_df["close_price"].iloc[-1], 2)} |

冲突窗口内最高收盘价出现在 {peak_row["trade_date"].date()}，为 {fmt_float(peak_row["close_price"], 2)} USD/barrel。
冲突窗口内最高盘中价出现在 {high_peak_row["trade_date"].date()}，为 {fmt_float(high_peak_row["high_price"], 2)} USD/barrel。
冲突窗口内最低收盘价出现在 {trough_row["trade_date"].date()}，为 {fmt_float(trough_row["close_price"], 2)} USD/barrel。

注意：赛题文字中提到的峰值约 126 USD/barrel 与附件 CSV 的收盘价口径不完全一致。后续建模建议以附件 CSV 的 `close_price` 作为拟合目标，同时在论文中说明题面峰值可视为外部叙述口径或极端冲击参考。

## 数据质量提示

- 日期重复：{df["trade_date"].duplicated().sum()}
- 收盘价缺失：{df["close_price"].isna().sum()}
- OHLC 异常行数：{len(ohlc_anomalies)}

OHLC 异常指 `high_price < low_price`、`high_price < open/close` 或 `low_price > open/close`。当前异常来自原始附件中 `close_price` 略低于 `low_price` 的两条历史记录，均不在冲突窗口内。阶段 1 不修正原始价格，只在质量报告中保留提示。

## 缺失值统计

{missing_lines}

说明：第一条记录的 `pre_close` 缺失来自原始 CSV 的 `NA`，收益率第一行缺失属于正常时间序列计算结果。

## 输出数据

- `{paths.processed_daily_csv.relative_to(PROJECT_ROOT)}`
- `{paths.processed_event_window_csv.relative_to(PROJECT_ROOT)}`

## 输出图表

{figure_lines}

## 对后续阶段的作用

- 阶段 2：提供真实价格区间，用于对比传统供需基准模型的高估结果。
- 阶段 3：提供冲突窗口真实价格路径，用于动态递推模型拟合。
- 阶段 4：提供 `actual_price`，用于 RMSE、MAE、峰值误差等校准指标。
- 阶段 7：三张基础图可进入论文的数据说明部分。
"""
    return report


def find_ohlc_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    anomaly_mask = (
        (df["high_price"] < df["low_price"])
        | (df["high_price"] < df["open_price"])
        | (df["high_price"] < df["close_price"])
        | (df["low_price"] > df["open_price"])
        | (df["low_price"] > df["close_price"])
    )
    columns = [
        "trade_date",
        "thscode",
        "pre_close",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "is_event_window",
    ]
    return df.loc[anomaly_mask, columns].copy()


def write_report(df: pd.DataFrame, event_df: pd.DataFrame, figures: list[Path], paths: Stage1Paths) -> Path:
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    anomaly_path = paths.reports_dir / "阶段1_OHLC异常记录.csv"
    ohlc_anomalies = find_ohlc_anomalies(df)
    ohlc_anomalies.to_csv(anomaly_path, index=False)

    report_path = paths.reports_dir / "stage1_data_cleaning_report.md"
    report_path.write_text(build_report(df, event_df, figures, paths), encoding="utf-8")
    return report_path


def main() -> None:
    config = load_config()
    paths = resolve_paths(config)
    event_window = resolve_event_window(config)

    df = clean_brent_data(paths.raw_csv, event_window)
    event_df = export_datasets(df, paths)
    figures = generate_figures(df, event_df, event_window, paths)
    report_path = write_report(df, event_df, figures, paths)

    print("Stage 1 complete")
    print(f"Cleaned data: {paths.processed_daily_csv.relative_to(PROJECT_ROOT)}")
    print(f"Event window data: {paths.processed_event_window_csv.relative_to(PROJECT_ROOT)}")
    for figure in figures:
        print(f"Figure: {figure.relative_to(PROJECT_ROOT)}")
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
