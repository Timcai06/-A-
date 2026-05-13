"""Causal residual correction for the short-term mechanism model.

The main short-term model remains the mechanism recursion.  This module adds a
small online correction layer that uses only lagged model errors.  It is closer
to adaptive filtering than to a new black-box model: at day t, the correction
may use errors observed before day t, but never the current day's actual price.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.metrics import mae, rmse
from src.common.paths import PROJECT_ROOT, ensure_parents
from src.models import dynamic_short_term as dynamic


@dataclass(frozen=True)
class ResidualCorrectionPaths:
    base_path_csv = PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv"
    corrected_path_csv = PROJECT_ROOT / "output" / "calibration" / "短期模型在线残差校正路径.csv"
    metrics_csv = PROJECT_ROOT / "output" / "calibration" / "短期模型在线残差校正指标.csv"
    report_path = PROJECT_ROOT / "output" / "reports" / "短期模型在线残差校正报告.md"
    figure_path = PROJECT_ROOT / "paper" / "figures" / "短期模型在线残差校正效果.png"


MIN_TRAIN_DAYS = 10
INITIAL_ERROR_FEEDBACK = 0.20
RIDGE_ALPHA = 1.0
MAX_FEEDBACK_COEFFICIENT = 0.50


def load_base_path() -> pd.DataFrame:
    if not ResidualCorrectionPaths.base_path_csv.exists():
        raise FileNotFoundError(f"Missing calibrated path: {ResidualCorrectionPaths.base_path_csv}")
    df = pd.read_csv(ResidualCorrectionPaths.base_path_csv, parse_dates=["trade_date"])
    required = {"trade_date", "actual_price", "simulated_price"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Calibrated path is missing columns: {sorted(missing)}")
    return df


def expanding_feedback_coefficients(model_error: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Estimate one-step residual feedback using only past errors."""
    errors = model_error.to_numpy(dtype=float)
    coefficients = np.zeros(len(errors), dtype=float)
    predicted_error = np.zeros(len(errors), dtype=float)

    for i in range(1, len(errors)):
        lagged_error = errors[i - 1]
        if i < MIN_TRAIN_DAYS:
            coefficient = INITIAL_ERROR_FEEDBACK
        else:
            x = errors[: i - 1]
            y = errors[1:i]
            denominator = float(np.dot(x, x) + RIDGE_ALPHA)
            coefficient = float(np.dot(x, y) / denominator) if denominator else 0.0
            coefficient = float(np.clip(coefficient, 0.0, MAX_FEEDBACK_COEFFICIENT))

        coefficients[i] = coefficient
        predicted_error[i] = coefficient * lagged_error

    return coefficients, predicted_error


def add_online_correction(base_path: pd.DataFrame) -> pd.DataFrame:
    df = base_path.copy()
    df["机制模型误差"] = df["simulated_price"] - df["actual_price"]
    coefficients, predicted_error = expanding_feedback_coefficients(df["机制模型误差"])
    df["在线校正系数"] = coefficients
    df["预测误差校正量"] = predicted_error
    df["online_corrected_price"] = df["simulated_price"] - df["预测误差校正量"]
    df["在线校正后误差"] = df["online_corrected_price"] - df["actual_price"]
    return df


def directional_accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    actual_direction = np.sign(actual.diff().iloc[1:])
    predicted_direction = np.sign(predicted.diff().iloc[1:])
    return float((actual_direction == predicted_direction).mean() * 100)


def model_metrics(df: pd.DataFrame, model_name: str, price_column: str, description: str) -> dict[str, float | str]:
    error = df[price_column] - df["actual_price"]
    return {
        "模型": model_name,
        "RMSE": rmse(error),
        "MAE": mae(error),
        "MAPE": float(np.mean(np.abs(error) / df["actual_price"]) * 100),
        "方向命中率": directional_accuracy(df["actual_price"], df[price_column]),
        "最大绝对误差": float(np.max(np.abs(error))),
        "说明": description,
    }


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    naive = df["actual_price"].shift(1)
    naive.iloc[0] = df["actual_price"].iloc[0]
    metric_rows = [
        model_metrics(df, "机制递推主模型", "simulated_price", "不使用当日真实价格的综合机制递推结果"),
        model_metrics(
            df,
            "在线残差校正模型",
            "online_corrected_price",
            "只使用上一交易日以前已观测误差进行一阶自适应校正",
        ),
        model_metrics(df.assign(naive_price=naive), "朴素上一日基准", "naive_price", "预测今日价格等于上一交易日真实收盘价"),
    ]
    metrics = pd.DataFrame(metric_rows)
    base_rmse = float(metrics.loc[metrics["模型"] == "机制递推主模型", "RMSE"].iloc[0])
    corrected_rmse = float(metrics.loc[metrics["模型"] == "在线残差校正模型", "RMSE"].iloc[0])
    naive_rmse = float(metrics.loc[metrics["模型"] == "朴素上一日基准", "RMSE"].iloc[0])
    metrics["相对机制主模型RMSE改善率"] = (base_rmse - metrics["RMSE"]) / base_rmse * 100
    metrics["相对朴素基准RMSE改善率"] = (naive_rmse - metrics["RMSE"]) / naive_rmse * 100
    metrics.loc[metrics["模型"] == "机制递推主模型", "相对机制主模型RMSE改善率"] = 0.0
    metrics.loc[metrics["模型"] == "朴素上一日基准", "相对朴素基准RMSE改善率"] = 0.0
    return metrics


def draw_figure(df: pd.DataFrame) -> None:
    dynamic.configure_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})
    ax = axes[0]
    ax.plot(df["trade_date"], df["actual_price"], color="#111827", linewidth=2.1, label="实际收盘价")
    ax.plot(df["trade_date"], df["simulated_price"], color="#dc2626", linewidth=1.8, label="机制递推主模型")
    ax.plot(
        df["trade_date"],
        df["online_corrected_price"],
        color="#2563eb",
        linewidth=1.8,
        linestyle="--",
        label="在线残差校正模型",
    )
    ax.axhspan(110, 120, color="#10b981", alpha=0.10, label="题面110-120区间")
    ax.set_title("短期模型在线残差校正效果")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper left", ncols=2)

    axes[1].bar(df["trade_date"], df["机制模型误差"], color="#ef4444", alpha=0.58, label="机制主模型误差")
    axes[1].plot(df["trade_date"], df["在线校正后误差"], color="#2563eb", linewidth=1.7, label="校正后误差")
    axes[1].axhline(0, color="#111827", linewidth=1.0)
    axes[1].set_title("逐日误差对比")
    axes[1].set_xlabel("日期")
    axes[1].set_ylabel("美元/桶")
    axes[1].legend(loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ResidualCorrectionPaths.figure_path)
    plt.close(fig)


def build_report(metrics: pd.DataFrame, corrected_path: pd.DataFrame) -> str:
    def value(model: str, column: str) -> float:
        return float(metrics.loc[metrics["模型"] == model, column].iloc[0])

    rows = "\n".join(
        "| {模型} | {RMSE:.3f} | {MAE:.3f} | {MAPE:.3f}% | {方向命中率:.1f}% | {最大绝对误差:.3f} | {相对机制主模型RMSE改善率:.2f}% | {相对朴素基准RMSE改善率:.2f}% |".format(
            **row
        )
        for row in metrics.to_dict("records")
    )
    mean_coefficient = float(corrected_path["在线校正系数"].mean())
    final_coefficient = float(corrected_path["在线校正系数"].iloc[-1])
    rmse_gain = value("在线残差校正模型", "相对机制主模型RMSE改善率")
    mae_gain = (
        (value("机制递推主模型", "MAE") - value("在线残差校正模型", "MAE"))
        / value("机制递推主模型", "MAE")
        * 100
    )

    return f"""# 短期模型在线残差校正报告

## 核心结论

本轮没有替换综合机制递推主模型，而是在其后增加一个可解释的在线校正层。第 $t$ 日校正只使用 $t-1$ 日以前已经观测到的模型误差，不能读取第 $t$ 日真实价格，因此它不是事后马后炮曲线。

在线校正后，RMSE 从 {value("机制递推主模型", "RMSE"):.3f} 降至 {value("在线残差校正模型", "RMSE"):.3f}，相对机制主模型改善 {rmse_gain:.2f}%；MAE 从 {value("机制递推主模型", "MAE"):.3f} 降至 {value("在线残差校正模型", "MAE"):.3f}，改善 {mae_gain:.2f}%。最大绝对误差从 {value("机制递推主模型", "最大绝对误差"):.3f} 降至 {value("在线残差校正模型", "最大绝对误差"):.3f}。这说明短期模型仍存在可利用的短期误差惯性，但可利用幅度有限。

需要诚实说明的是，在线校正模型的方向命中率为 {value("在线残差校正模型", "方向命中率"):.1f}%，低于机制主模型的 {value("机制递推主模型", "方向命中率"):.1f}%。因此论文中不应把它替代主模型，而应作为“误差自适应增强”证据：它能降低水平误差和极端误差，但机制主模型仍负责方向解释和经济含义。

## 方法

设机制主模型第 $t$ 日误差为

```text
e_t = P_t^{{model}} - P_t^{{actual}}
```

在线校正层估计一阶误差反馈：

```text
e_t = rho_t * e_{{t-1}} + noise_t
P_t^{{corrected}} = P_t^{{model}} - rho_t * e_{{t-1}}
```

其中 $rho_t$ 使用扩展窗口岭回归估计，只允许使用 $1$ 至 $t-1$ 日误差，并限制在 $[0, 0.5]$ 内，防止校正层过度追逐噪声。平均在线校正系数为 {mean_coefficient:.3f}，窗口末日系数为 {final_coefficient:.3f}。

## 指标对比

| 模型 | RMSE | MAE | MAPE | 方向命中率 | 最大绝对误差 | 相对机制主模型RMSE改善 | 相对朴素基准RMSE改善 |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

## 论文使用建议

在线残差校正适合放在短期模型检验部分，作为“机器学习/自适应过滤思想的轻量引入”。它的价值是证明：我们不是没有考虑数据驱动方法，而是在样本很短、解释性很重要的赛题里，只把数据驱动层用于修正残差，不让黑箱模型吞掉主机制。

## 输出产物

- `{ResidualCorrectionPaths.corrected_path_csv.relative_to(PROJECT_ROOT)}`
- `{ResidualCorrectionPaths.metrics_csv.relative_to(PROJECT_ROOT)}`
- `{ResidualCorrectionPaths.figure_path.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    ensure_parents(
        [
            ResidualCorrectionPaths.corrected_path_csv,
            ResidualCorrectionPaths.metrics_csv,
            ResidualCorrectionPaths.report_path,
            ResidualCorrectionPaths.figure_path,
        ]
    )
    base_path = load_base_path()
    corrected_path = add_online_correction(base_path)
    metrics = compute_metrics(corrected_path)

    corrected_path.to_csv(ResidualCorrectionPaths.corrected_path_csv, index=False)
    metrics.to_csv(ResidualCorrectionPaths.metrics_csv, index=False)
    draw_figure(corrected_path)
    ResidualCorrectionPaths.report_path.write_text(build_report(metrics, corrected_path), encoding="utf-8")

    print("Short-term online residual correction complete")
    print(f"Corrected path: {ResidualCorrectionPaths.corrected_path_csv.relative_to(PROJECT_ROOT)}")
    print(f"Metrics: {ResidualCorrectionPaths.metrics_csv.relative_to(PROJECT_ROOT)}")
    print(f"Report: {ResidualCorrectionPaths.report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
