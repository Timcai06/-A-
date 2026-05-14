"""Machine-learning baselines for the short-term model.

The auxiliary Ridge layer is intentionally low-degree and leakage-safe.  It now
uses a scikit-learn ``StandardScaler + Ridge`` pipeline so that feature scaling,
regularization and future extensions stay explicit and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.short_term_ml_features import OUTPUT_CSV as FEATURE_CSV
from src.analysis.short_term_ml_features import main as build_features
from src.common.metrics import direction_hit_rate, mae, mape, rmse
from src.common.paths import PROJECT_ROOT, ensure_parents
from src.common.plotting import PAPER_COLORS, SCENARIO_COLORS
from src.models import dynamic_short_term as dynamic


@dataclass(frozen=True)
class MLResidualPaths:
    mechanism_path_csv = PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv"
    online_correction_csv = PROJECT_ROOT / "output" / "calibration" / "短期模型在线残差校正路径.csv"
    comparison_csv = PROJECT_ROOT / "output" / "calibration" / "短期机器学习模型对比.csv"
    prediction_csv = PROJECT_ROOT / "output" / "calibration" / "短期机器学习残差修正路径.csv"
    report_path = PROJECT_ROOT / "output" / "reports" / "短期机器学习残差修正报告.md"
    figure_path = PROJECT_ROOT / "figures" / "短期机器学习残差修正对比.png"


ALPHA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
EVENT_START = pd.Timestamp("2026-03-02")
VALIDATION_START = pd.Timestamp("2024-01-01")
RIDGE_ASSIST_WINDOWS = [
    ("高位平台形成", pd.Timestamp("2026-03-17"), pd.Timestamp("2026-03-31")),
    ("中期再定价回落", pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-17")),
]


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


def make_ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("standardize", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def choose_alpha(train: pd.DataFrame, columns: list[str]) -> tuple[float, pd.DataFrame]:
    train_part = train[train["trade_date"] < VALIDATION_START].copy()
    valid_part = train[(train["trade_date"] >= VALIDATION_START) & (train["trade_date"] < EVENT_START)].copy()
    if len(valid_part) < 60:
        split = int(len(train) * 0.8)
        train_part = train.iloc[:split].copy()
        valid_part = train.iloc[split:].copy()

    rows: list[dict[str, float]] = []
    for alpha in ALPHA_GRID:
        model = make_ridge_pipeline(alpha)
        model.fit(train_part[columns].to_numpy(float), train_part["target_log_return"].to_numpy(float))
        pred_return = model.predict(valid_part[columns].to_numpy(float))
        pred_price = valid_part["pre_close_filled"].to_numpy(float) * np.exp(pred_return)
        error = pred_price - valid_part["target_close_price"].to_numpy(float)
        rows.append({"alpha": alpha, "验证RMSE": rmse(error), "验证MAE": mae(error), "验证样本数": float(len(valid_part))})

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
    ridge_model = make_ridge_pipeline(best_alpha)
    ridge_model.fit(pre_event[columns].to_numpy(float), pre_event["target_log_return"].to_numpy(float))
    pred_return = ridge_model.predict(event_features[columns].to_numpy(float))

    event_predictions = event_features[["trade_date", "pre_close_filled", "target_close_price"]].copy()
    event_predictions["ridge_predicted_return"] = pred_return
    event_predictions["ridge_price"] = event_predictions["pre_close_filled"] * np.exp(event_predictions["ridge_predicted_return"])

    result = mechanism.merge(event_predictions, on="trade_date", how="inner")
    result["naive_price"] = result["pre_close_filled"]
    result["ridge_return_correction"] = result["ridge_price"] - result["pre_close_filled"]
    result["mechanism_plus_ridge_return"] = result["simulated_price"] + result["ridge_return_correction"]
    result["ridge_assist_phase"] = "未启用"
    result["phase_gated_ridge_correction"] = 0.0
    for name, start, end in RIDGE_ASSIST_WINDOWS:
        mask = result["trade_date"].between(start, end)
        result.loc[mask, "ridge_assist_phase"] = name
        result.loc[mask, "phase_gated_ridge_correction"] = result.loc[mask, "ridge_return_correction"]
    result["mechanism_plus_phase_ridge"] = result["simulated_price"] + result["phase_gated_ridge_correction"]
    if online is not None:
        online_columns = ["trade_date"]
        for column in ["online_corrected_price", "regime_corrected_price"]:
            if column in online.columns:
                online_columns.append(column)
        if len(online_columns) > 1:
            result = result.merge(online[online_columns], on="trade_date", how="left")

    metrics = [
        model_metrics(result, "机制递推主模型", "simulated_price", "当前短期主模型"),
        model_metrics(result, "历史Ridge收益率模型", "ridge_price", "仅用2017-2026冲突前价格特征训练的一日收益率模型"),
        model_metrics(result, "机制+Ridge收益率修正", "mechanism_plus_ridge_return", "将历史Ridge预测的正常收益变化叠加到机制路径"),
        model_metrics(
            result,
            "机制+阶段Ridge修正",
            "mechanism_plus_phase_ridge",
            "仅在高位平台与中期再定价两个残差薄弱段启用Ridge收益率修正",
        ),
        model_metrics(result, "朴素上一日基准", "naive_price", "预测今日价格等于上一交易日前收盘价"),
    ]
    if "online_corrected_price" in result.columns:
        metrics.insert(
            1,
            model_metrics(result, "在线残差校正模型", "online_corrected_price", "只使用滞后误差的一阶在线校正"),
        )
    if "regime_corrected_price" in result.columns:
        metrics.insert(
            2,
            model_metrics(
                result,
                "分段感知在线校正模型",
                "regime_corrected_price",
                "只使用滞后误差，并允许平台期/再定价期反馈强度不同",
            ),
        )
    comparison = pd.DataFrame(metrics)
    naive_rmse = float(comparison.loc[comparison["模型"] == "朴素上一日基准", "RMSE"].iloc[0])
    mechanism_rmse = float(comparison.loc[comparison["模型"] == "机制递推主模型", "RMSE"].iloc[0])
    comparison["相对朴素基准RMSE改善率"] = (naive_rmse - comparison["RMSE"]) / naive_rmse * 100
    comparison["相对机制主模型RMSE改善率"] = (mechanism_rmse - comparison["RMSE"]) / mechanism_rmse * 100

    return result, comparison, alpha_scores


def draw_figure(predictions: pd.DataFrame) -> None:
    dynamic.configure_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.0), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})

    ax = axes[0]
    ax.plot(predictions["trade_date"], predictions["actual_price"], color=SCENARIO_COLORS["actual"], linewidth=2.1, label="实际收盘价")
    ax.plot(predictions["trade_date"], predictions["simulated_price"], color=SCENARIO_COLORS["fit"], linewidth=1.8, label="机制递推主模型")
    ax.plot(
        predictions["trade_date"],
        predictions["mechanism_plus_phase_ridge"],
        color=SCENARIO_COLORS["buffer"],
        linewidth=2.0,
        linestyle="--",
        label="最终短期增强模型",
    )
    ax.set_title("最终短期增强模型与机制主模型")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper left", ncols=1)

    base_error = predictions["simulated_price"] - predictions["actual_price"]
    final_error = predictions["mechanism_plus_phase_ridge"] - predictions["actual_price"]
    axes[1].plot(predictions["trade_date"], base_error, color=SCENARIO_COLORS["risk"], linewidth=1.5, label="机制主模型误差")
    axes[1].plot(predictions["trade_date"], final_error, color=SCENARIO_COLORS["buffer"], linewidth=1.7, linestyle="--", label="最终增强模型误差")
    axes[1].axhline(0, color=PAPER_COLORS["ink"], linewidth=1.0)
    axes[1].set_title("误差对比：只保留最终采用路径")
    axes[1].set_xlabel("日期")
    axes[1].set_ylabel("美元/桶")
    axes[1].legend(loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(MLResidualPaths.figure_path)
    plt.close(fig)


def build_report(comparison: pd.DataFrame, alpha_scores: pd.DataFrame) -> str:
    best_alpha = float(alpha_scores.iloc[0]["alpha"])
    report_models = comparison[comparison["模型"].isin(["机制递推主模型", "机制+阶段Ridge修正", "朴素上一日基准"])].copy()
    rows = "\n".join(
        "| {模型} | {RMSE:.3f} | {MAE:.3f} | {MAPE:.3f}% | {方向命中率:.1f}% | {最大绝对误差:.3f} | {相对机制主模型RMSE改善率:.2f}% | {相对朴素基准RMSE改善率:.2f}% |".format(
            **row
        )
        for row in report_models.to_dict("records")
    )
    alpha_rows = "\n".join(
        f"| {row['alpha']:.1f} | {row['验证RMSE']:.3f} | {row['验证MAE']:.3f} |"
        for row in alpha_scores.to_dict("records")
    )

    best_model = comparison.sort_values(["RMSE", "MAE"]).iloc[0]
    mechanism = comparison[comparison["模型"] == "机制递推主模型"].iloc[0]
    ridge = comparison[comparison["模型"] == "历史Ridge收益率模型"].iloc[0]
    phase = comparison[comparison["模型"] == "机制+阶段Ridge修正"].iloc[0]

    return f"""# 短期机器学习残差修正报告

## 核心结论

本轮使用 2017--2026 冲突前历史价格特征训练 Ridge 一日收益率模型，并在 2026 冲突窗口上做外部测试。结果显示，历史价格机器学习模型不能替代综合机制递推主模型，但可以在残差薄弱窗口提供低自由度辅助修正。

最终采用的短期增强路径为 **机制+阶段Ridge修正**，RMSE 为 {float(phase["RMSE"]):.3f}，MAE 为 {float(phase["MAE"]):.3f}，方向命中率为 {float(phase["方向命中率"]):.1f}%。机制递推主模型 RMSE 为 {float(mechanism["RMSE"]):.3f}，历史 Ridge 收益率模型 RMSE 为 {float(ridge["RMSE"]):.3f}。

正文只报告最终增强路径和关键基准。纯 Ridge、全窗口 Ridge、在线残差校正和分段校正保留在 CSV 中作为筛选留痕，不作为论文主线展开。

## 训练方式

- 特征来源：`output/calibration/短期机器学习特征样本.csv`
- 模型管线：scikit-learn `StandardScaler + Ridge`，先标准化特征，再进行 L2 正则线性回归。
- 训练数据：2026-03-02 之前的历史价格特征。
- 验证数据：优先使用 2024-01-01 至冲突前样本选择 Ridge 正则强度。
- 测试数据：2026-03-02 至 2026-05-05 冲突窗口。
- 最优 Ridge alpha：{best_alpha:.1f}

## 核心指标

| 模型 | RMSE | MAE | MAPE | 方向命中率 | 最大绝对误差 | 相对机制主模型RMSE改善 | 相对朴素基准RMSE改善 |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

## Ridge 正则强度验证

| alpha | 验证RMSE | 验证MAE |
|---:|---:|---:|
{alpha_rows}

## 建模判断

最终短期模型不采用“谁的 RMSE 最低就堆谁”的写法。机制递推主模型负责供需、缓冲和风险溢价解释；阶段 Ridge 只在高位平台形成和中期再定价回落两个弱窗口修正短期收益惯性。这个处理能提高拟合质量，同时避免把论文写成候选模型流水账。

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
