"""Machine-learning baselines for the short-term model.

This module intentionally starts with a low-degree Ridge model implemented with
NumPy.  It avoids adding a heavy dependency and provides a transparent first
answer to whether data-driven historical features can improve the event-window
short-term model.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.short_term_ml_features import OUTPUT_CSV as FEATURE_CSV
from src.analysis.short_term_ml_features import main as build_features
from src.common.metrics import direction_hit_rate, mae, mape, rmse
from src.common.paths import PROJECT_ROOT, ensure_parents
from src.models import dynamic_short_term as dynamic


@dataclass(frozen=True)
class MLResidualPaths:
    mechanism_path_csv = PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv"
    online_correction_csv = PROJECT_ROOT / "output" / "calibration" / "短期模型在线残差校正路径.csv"
    comparison_csv = PROJECT_ROOT / "output" / "calibration" / "短期机器学习模型对比.csv"
    prediction_csv = PROJECT_ROOT / "output" / "calibration" / "短期机器学习残差修正路径.csv"
    report_path = PROJECT_ROOT / "output" / "reports" / "短期机器学习残差修正报告.md"
    figure_path = PROJECT_ROOT / "paper" / "figures" / "短期机器学习残差修正对比.png"


ALPHA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
EVENT_START = pd.Timestamp("2026-03-02")
VALIDATION_START = pd.Timestamp("2024-01-01")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if not FEATURE_CSV.exists():
        build_features()
    if not MLResidualPaths.mechanism_path_csv.exists():
        raise FileNotFoundError(f"Missing mechanism path: {MLResidualPaths.mechanism_path_csv}")

    features = pd.read_csv(FEATURE_CSV, parse_dates=["trade_date"])
    mechanism = pd.read_csv(MLResidualPaths.mechanism_path_csv, parse_dates=["trade_date"])
    online = None
    if MLResidualPaths.online_correction_csv.exists():
        online = pd.read_csv(MLResidualPaths.online_correction_csv, parse_dates=["trade_date"])
    return features, mechanism, online


def feature_columns(features: pd.DataFrame) -> list[str]:
    excluded = {
        "trade_date",
        "pre_close_filled",
        "target_close_price",
        "target_log_return",
        "target_price_change",
        "是否冲突窗口",
    }
    return [column for column in features.columns if column not in excluded]


def clean_training_frame(features: pd.DataFrame) -> pd.DataFrame:
    columns = feature_columns(features)
    needed = columns + ["target_log_return", "target_close_price", "pre_close_filled", "trade_date"]
    return features[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    penalty = np.diag([0.0] + [alpha] * Xs.shape[1])
    beta = np.linalg.solve(Xd.T @ Xd + penalty, Xd.T @ y)
    return beta, mu, sigma


def predict_ridge(X: np.ndarray, beta: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    Xs = (X - mu) / sigma
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    return Xd @ beta


def choose_alpha(train: pd.DataFrame, columns: list[str]) -> tuple[float, pd.DataFrame]:
    train_part = train[train["trade_date"] < VALIDATION_START].copy()
    valid_part = train[(train["trade_date"] >= VALIDATION_START) & (train["trade_date"] < EVENT_START)].copy()
    if len(valid_part) < 60:
        split = int(len(train) * 0.8)
        train_part = train.iloc[:split].copy()
        valid_part = train.iloc[split:].copy()

    rows: list[dict[str, float]] = []
    for alpha in ALPHA_GRID:
        beta, mu, sigma = fit_ridge(train_part[columns].to_numpy(float), train_part["target_log_return"].to_numpy(float), alpha)
        pred_return = predict_ridge(valid_part[columns].to_numpy(float), beta, mu, sigma)
        pred_price = valid_part["pre_close_filled"].to_numpy(float) * np.exp(pred_return)
        error = pred_price - valid_part["target_close_price"].to_numpy(float)
        rows.append({"alpha": alpha, "验证RMSE": rmse(error), "验证MAE": mae(error)})

    score = pd.DataFrame(rows).sort_values(["验证RMSE", "验证MAE"]).reset_index(drop=True)
    return float(score.iloc[0]["alpha"]), score


def model_metrics(df: pd.DataFrame, name: str, price_column: str, description: str) -> dict[str, float | str]:
    actual = df["actual_price"]
    predicted = df[price_column]
    error = predicted - actual
    return {
        "模型": name,
        "RMSE": rmse(error),
        "MAE": mae(error),
        "MAPE": mape(actual, predicted),
        "方向命中率": direction_hit_rate(actual, predicted),
        "最大绝对误差": float(np.max(np.abs(error))),
        "说明": description,
    }


def build_predictions(
    features: pd.DataFrame,
    mechanism: pd.DataFrame,
    online: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable = clean_training_frame(features)
    columns = feature_columns(features)
    pre_event = usable[usable["trade_date"] < EVENT_START].copy()
    event_features = usable[usable["trade_date"].isin(mechanism["trade_date"])].copy()

    best_alpha, alpha_scores = choose_alpha(pre_event, columns)
    beta, mu, sigma = fit_ridge(pre_event[columns].to_numpy(float), pre_event["target_log_return"].to_numpy(float), best_alpha)
    pred_return = predict_ridge(event_features[columns].to_numpy(float), beta, mu, sigma)

    event_predictions = event_features[["trade_date", "pre_close_filled", "target_close_price"]].copy()
    event_predictions["ridge_predicted_return"] = pred_return
    event_predictions["ridge_price"] = event_predictions["pre_close_filled"] * np.exp(event_predictions["ridge_predicted_return"])

    result = mechanism.merge(event_predictions, on="trade_date", how="inner")
    result["naive_price"] = result["pre_close_filled"]
    result["ridge_return_correction"] = result["ridge_price"] - result["pre_close_filled"]
    result["mechanism_plus_ridge_return"] = result["simulated_price"] + result["ridge_return_correction"]
    if online is not None and "online_corrected_price" in online.columns:
        result = result.merge(online[["trade_date", "online_corrected_price"]], on="trade_date", how="left")

    metrics = [
        model_metrics(result, "机制递推主模型", "simulated_price", "当前短期主模型"),
        model_metrics(result, "历史Ridge收益率模型", "ridge_price", "仅用2017-2026冲突前价格特征训练的一日收益率模型"),
        model_metrics(result, "机制+Ridge收益率修正", "mechanism_plus_ridge_return", "将历史Ridge预测的正常收益变化叠加到机制路径"),
        model_metrics(result, "朴素上一日基准", "naive_price", "预测今日价格等于上一交易日前收盘价"),
    ]
    if "online_corrected_price" in result.columns:
        metrics.insert(
            1,
            model_metrics(result, "在线残差校正模型", "online_corrected_price", "只使用滞后误差的一阶在线校正"),
        )
    comparison = pd.DataFrame(metrics)
    naive_rmse = float(comparison.loc[comparison["模型"] == "朴素上一日基准", "RMSE"].iloc[0])
    mechanism_rmse = float(comparison.loc[comparison["模型"] == "机制递推主模型", "RMSE"].iloc[0])
    comparison["相对朴素基准RMSE改善率"] = (naive_rmse - comparison["RMSE"]) / naive_rmse * 100
    comparison["相对机制主模型RMSE改善率"] = (mechanism_rmse - comparison["RMSE"]) / mechanism_rmse * 100

    return result, comparison, alpha_scores


def draw_figure(predictions: pd.DataFrame) -> None:
    dynamic.configure_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})

    ax = axes[0]
    ax.plot(predictions["trade_date"], predictions["actual_price"], color="#111827", linewidth=2.1, label="实际收盘价")
    ax.plot(predictions["trade_date"], predictions["simulated_price"], color="#dc2626", linewidth=1.8, label="机制递推主模型")
    if "online_corrected_price" in predictions.columns:
        ax.plot(
            predictions["trade_date"],
            predictions["online_corrected_price"],
            color="#2563eb",
            linewidth=1.7,
            linestyle="--",
            label="在线残差校正",
        )
    ax.plot(predictions["trade_date"], predictions["ridge_price"], color="#7c3aed", linewidth=1.5, label="历史Ridge收益率")
    ax.set_title("短期机器学习基准与机制模型对比")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper left", ncols=2)

    base_error = predictions["simulated_price"] - predictions["actual_price"]
    ridge_error = predictions["ridge_price"] - predictions["actual_price"]
    axes[1].plot(predictions["trade_date"], base_error, color="#dc2626", linewidth=1.5, label="机制主模型误差")
    axes[1].plot(predictions["trade_date"], ridge_error, color="#7c3aed", linewidth=1.5, label="历史Ridge误差")
    axes[1].axhline(0, color="#111827", linewidth=1.0)
    axes[1].set_title("误差对比")
    axes[1].set_xlabel("日期")
    axes[1].set_ylabel("美元/桶")
    axes[1].legend(loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(MLResidualPaths.figure_path)
    plt.close(fig)


def build_report(comparison: pd.DataFrame, alpha_scores: pd.DataFrame) -> str:
    best_alpha = float(alpha_scores.iloc[0]["alpha"])
    rows = "\n".join(
        "| {模型} | {RMSE:.3f} | {MAE:.3f} | {MAPE:.3f}% | {方向命中率:.1f}% | {最大绝对误差:.3f} | {相对机制主模型RMSE改善率:.2f}% | {相对朴素基准RMSE改善率:.2f}% |".format(
            **row
        )
        for row in comparison.to_dict("records")
    )
    alpha_rows = "\n".join(
        f"| {row['alpha']:.1f} | {row['验证RMSE']:.3f} | {row['验证MAE']:.3f} |"
        for row in alpha_scores.to_dict("records")
    )

    best_model = comparison.sort_values(["RMSE", "MAE"]).iloc[0]
    mechanism = comparison[comparison["模型"] == "机制递推主模型"].iloc[0]
    ridge = comparison[comparison["模型"] == "历史Ridge收益率模型"].iloc[0]

    return f"""# 短期机器学习残差修正报告

## 核心结论

本轮使用 2017--2026 冲突前历史价格特征训练 Ridge 一日收益率模型，并在 2026 冲突窗口上做外部测试。结果显示，历史价格机器学习模型可以作为重要基准，但不能替代综合机制递推主模型。

当前 RMSE 最低的模型为 **{best_model["模型"]}**，RMSE 为 {float(best_model["RMSE"]):.3f}。机制递推主模型 RMSE 为 {float(mechanism["RMSE"]):.3f}，历史 Ridge 收益率模型 RMSE 为 {float(ridge["RMSE"]):.3f}。

这说明普通历史收益率模型很难单独解释霍尔木兹封锁冲突窗口。机器学习的价值主要是作为基准和辅助修正，而不是取代赛题机制模型。

## 训练方式

- 特征来源：`output/calibration/短期机器学习特征样本.csv`
- 训练数据：2026-03-02 之前的历史价格特征。
- 验证数据：优先使用 2024-01-01 至冲突前样本选择 Ridge 正则强度。
- 测试数据：2026-03-02 至 2026-05-05 冲突窗口。
- 最优 Ridge alpha：{best_alpha:.1f}

## 指标对比

| 模型 | RMSE | MAE | MAPE | 方向命中率 | 最大绝对误差 | 相对机制主模型RMSE改善 | 相对朴素基准RMSE改善 |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

## Ridge 正则强度验证

| alpha | 验证RMSE | 验证MAE |
|---:|---:|---:|
{alpha_rows}

## 建模判断

如果历史 Ridge 模型或机制加 Ridge 修正没有稳定优于机制主模型，就不能为了“用了机器学习”而把它放到最终主模型位置。下一步应继续尝试更严格的滚动验证、状态空间滤波或低自由度非线性模型，而不是直接上 LSTM/Transformer。

## 输出

- `{MLResidualPaths.comparison_csv.relative_to(PROJECT_ROOT)}`
- `{MLResidualPaths.prediction_csv.relative_to(PROJECT_ROOT)}`
- `{MLResidualPaths.figure_path.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    ensure_parents(
        [
            MLResidualPaths.comparison_csv,
            MLResidualPaths.prediction_csv,
            MLResidualPaths.report_path,
            MLResidualPaths.figure_path,
        ]
    )
    features, mechanism, online = load_inputs()
    predictions, comparison, alpha_scores = build_predictions(features, mechanism, online)
    predictions.to_csv(MLResidualPaths.prediction_csv, index=False)
    comparison.to_csv(MLResidualPaths.comparison_csv, index=False)
    draw_figure(predictions)
    MLResidualPaths.report_path.write_text(build_report(comparison, alpha_scores), encoding="utf-8")

    print("Short-term ML residual audit complete")
    print(f"Comparison: {MLResidualPaths.comparison_csv.relative_to(PROJECT_ROOT)}")
    print(f"Predictions: {MLResidualPaths.prediction_csv.relative_to(PROJECT_ROOT)}")
    print(f"Report: {MLResidualPaths.report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
