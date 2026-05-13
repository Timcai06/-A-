"""Rolling validation for historical ML short-term benchmarks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.short_term_ml_features import OUTPUT_CSV as FEATURE_CSV
from src.analysis.short_term_ml_features import main as build_features
from src.analysis.short_term_ml_residual_model import fit_ridge, feature_columns, predict_ridge
from src.common.metrics import direction_hit_rate, mae, mape, rmse
from src.common.paths import PROJECT_ROOT, ensure_parents
from src.models import dynamic_short_term as dynamic


ROLLING_RESULT_CSV = PROJECT_ROOT / "output" / "calibration" / "短期历史滚动验证结果.csv"
ROLLING_DAILY_CSV = PROJECT_ROOT / "output" / "calibration" / "短期历史滚动日度预测.csv"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "短期历史滚动验证报告.md"
FIGURE_PATH = PROJECT_ROOT / "paper" / "figures" / "短期历史滚动验证误差分布.png"

ALPHA = 300.0
MIN_TRAIN_ROWS = 500
WINDOW_SIZE = 46
EVENT_START = pd.Timestamp("2026-03-02")
EVENT_END = pd.Timestamp("2026-05-05")


def load_feature_table() -> pd.DataFrame:
    if not FEATURE_CSV.exists():
        build_features()
    features = pd.read_csv(FEATURE_CSV, parse_dates=["trade_date"])
    columns = feature_columns(features)
    needed = columns + ["trade_date", "pre_close_filled", "target_close_price", "target_log_return", "是否冲突窗口"]
    return features[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def expanding_ridge_predictions(features: pd.DataFrame) -> pd.DataFrame:
    columns = feature_columns(features)
    rows: list[dict[str, float | pd.Timestamp | bool]] = []

    X_all = features[columns].to_numpy(float)
    y_all = features["target_log_return"].to_numpy(float)
    pre_close = features["pre_close_filled"].to_numpy(float)
    actual = features["target_close_price"].to_numpy(float)

    for i in range(MIN_TRAIN_ROWS, len(features)):
        train_X = X_all[:i]
        train_y = y_all[:i]
        beta, mu, sigma = fit_ridge(train_X, train_y, ALPHA)
        pred_return = float(predict_ridge(X_all[i : i + 1], beta, mu, sigma)[0])
        ridge_price = float(pre_close[i] * np.exp(pred_return))
        naive_price = float(pre_close[i])
        actual_price = float(actual[i])
        rows.append(
            {
                "trade_date": features.loc[i, "trade_date"],
                "actual_price": actual_price,
                "naive_price": naive_price,
                "ridge_price": ridge_price,
                "ridge_error": ridge_price - actual_price,
                "naive_error": naive_price - actual_price,
                "是否冲突窗口": bool(features.loc[i, "是否冲突窗口"]),
            }
        )

    return pd.DataFrame(rows)


def window_metrics(window: pd.DataFrame) -> dict[str, float]:
    ridge_error = window["ridge_price"] - window["actual_price"]
    naive_error = window["naive_price"] - window["actual_price"]
    ridge_rmse = rmse(ridge_error)
    naive_rmse = rmse(naive_error)
    return {
        "开始日期": window["trade_date"].iloc[0],
        "结束日期": window["trade_date"].iloc[-1],
        "样本数": len(window),
        "Ridge_RMSE": ridge_rmse,
        "Ridge_MAE": mae(ridge_error),
        "Ridge_MAPE": mape(window["actual_price"], window["ridge_price"]),
        "Ridge_方向命中率": direction_hit_rate(window["actual_price"], window["ridge_price"]),
        "Naive_RMSE": naive_rmse,
        "Naive_MAE": mae(naive_error),
        "Naive_MAPE": mape(window["actual_price"], window["naive_price"]),
        "Naive_方向命中率": direction_hit_rate(window["actual_price"], window["naive_price"]),
        "Ridge相对Naive_RMSE改善率": (naive_rmse - ridge_rmse) / naive_rmse * 100 if naive_rmse else float("nan"),
        "是否冲突窗口": bool(window["是否冲突窗口"].all()),
    }


def rolling_window_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for start in range(0, len(daily) - WINDOW_SIZE + 1):
        window = daily.iloc[start : start + WINDOW_SIZE].copy()
        rows.append(window_metrics(window))

    metrics = pd.DataFrame(rows)
    event_mask = (daily["trade_date"] >= EVENT_START) & (daily["trade_date"] <= EVENT_END)
    event_window = daily[event_mask].copy()
    if len(event_window) == WINDOW_SIZE:
        event_metrics = window_metrics(event_window)
        event_metrics["是否冲突窗口"] = True
        metrics = pd.concat([metrics, pd.DataFrame([event_metrics])], ignore_index=True)
    return metrics.drop_duplicates(subset=["开始日期", "结束日期"], keep="last").reset_index(drop=True)


def draw_figure(metrics: pd.DataFrame) -> None:
    dynamic.configure_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    normal = metrics[~metrics["是否冲突窗口"]].copy()
    event = metrics[metrics["是否冲突窗口"]].tail(1)
    ax.hist(normal["Ridge相对Naive_RMSE改善率"], bins=30, color="#2563eb", alpha=0.72, label="历史46日窗口")
    if not event.empty:
        value = float(event["Ridge相对Naive_RMSE改善率"].iloc[0])
        ax.axvline(value, color="#dc2626", linewidth=2.2, label=f"冲突窗口：{value:.1f}%")
    ax.axvline(0, color="#111827", linewidth=1.0)
    ax.set_title("历史滚动窗口中 Ridge 相对朴素基准的 RMSE 改善分布")
    ax.set_xlabel("Ridge 相对朴素基准 RMSE 改善率（%）")
    ax.set_ylabel("窗口数量")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH)
    plt.close(fig)


def build_report(metrics: pd.DataFrame, daily: pd.DataFrame) -> str:
    normal = metrics[~metrics["是否冲突窗口"]].copy()
    event = metrics[metrics["是否冲突窗口"]].tail(1)
    event_row = event.iloc[0] if not event.empty else None
    improvement = normal["Ridge相对Naive_RMSE改善率"]
    better_ratio = float((improvement > 0).mean() * 100)
    q10, q50, q90 = improvement.quantile([0.10, 0.50, 0.90]).to_list()

    event_text = "未找到完整冲突窗口。"
    if event_row is not None:
        event_improvement = float(event_row["Ridge相对Naive_RMSE改善率"])
        event_percentile = float((improvement <= event_improvement).mean() * 100)
        event_text = (
            f"冲突窗口 Ridge 相对朴素基准 RMSE 改善率为 {event_improvement:.2f}%，"
            f"在历史 46 日窗口中约处于 {event_percentile:.1f}% 分位。"
        )

    return f"""# 短期历史滚动验证报告

## 核心结论

本报告用扩展窗口 Ridge 模型对 2017--2026 历史价格样本做一日提前滚动预测，并按 46 个交易日窗口统计误差。它用于判断历史价格机器学习模型是否稳定优于朴素上一日基准。

历史窗口中，Ridge 相对朴素基准 RMSE 改善率的 10%/50%/90% 分位分别为 {q10:.2f}%、{q50:.2f}%、{q90:.2f}%。Ridge 优于朴素基准的窗口占比为 {better_ratio:.1f}%。

{event_text}

这说明纯历史价格机器学习模型并不是稳定压倒朴素基准的强模型；它可以作为辅助基准，但不能单独解释霍尔木兹冲突窗口。短期主线仍应是机制模型，机器学习只能用于有限的收益率修正或残差修正。

## 设置

| 项目 | 数值 |
|---|---:|
| Ridge alpha | {ALPHA:.1f} |
| 最小训练行数 | {MIN_TRAIN_ROWS} |
| 滚动窗口长度 | {WINDOW_SIZE} |
| 日度预测数 | {len(daily)} |
| 46日窗口数 | {len(metrics)} |

## 输出

- `{ROLLING_DAILY_CSV.relative_to(PROJECT_ROOT)}`
- `{ROLLING_RESULT_CSV.relative_to(PROJECT_ROOT)}`
- `{FIGURE_PATH.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    ensure_parents([ROLLING_RESULT_CSV, ROLLING_DAILY_CSV, REPORT_PATH, FIGURE_PATH])
    features = load_feature_table()
    daily = expanding_ridge_predictions(features)
    metrics = rolling_window_metrics(daily)
    daily.to_csv(ROLLING_DAILY_CSV, index=False)
    metrics.to_csv(ROLLING_RESULT_CSV, index=False)
    draw_figure(metrics)
    REPORT_PATH.write_text(build_report(metrics, daily), encoding="utf-8")

    print("Short-term historical rolling validation complete")
    print(f"Daily predictions: {ROLLING_DAILY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Window metrics: {ROLLING_RESULT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
