"""Evaluation metrics for Stage 4 calibration candidates."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.common.metrics import mae, rmse


def segment_error_rows(simulation: pd.DataFrame) -> list[dict[str, Any]]:
    df = simulation.copy()
    df["error"] = df["simulated_price"] - df["actual_price"]
    df["abs_error"] = df["error"].abs()
    segments = [
        ("全窗口", df.index == df.index),
        ("前期冲击_0到14天", df["day_index"] <= 14),
        ("中期平台_15到35天", (df["day_index"] > 14) & (df["day_index"] <= 35)),
        ("后期再定价_36天以后", df["day_index"] > 35),
        ("高价平台_actual大于等于110", df["actual_price"] >= 110),
        ("低价回落_actual小于100", df["actual_price"] < 100),
    ]
    rows: list[dict[str, Any]] = []
    for name, mask in segments:
        sub = df[mask].copy()
        rows.append(
            {
                "分段": name,
                "样本数": len(sub),
                "RMSE": rmse(sub["error"]),
                "MAE": mae(sub["error"]),
                "平均偏差": float(sub["error"].mean()) if len(sub) else float("nan"),
                "最大绝对误差": float(sub["abs_error"].max()) if len(sub) else float("nan"),
                "实际最低价": float(sub["actual_price"].min()) if len(sub) else float("nan"),
                "实际最高价": float(sub["actual_price"].max()) if len(sub) else float("nan"),
            }
        )
    return rows


def evaluate_simulation(simulation: pd.DataFrame) -> dict[str, float]:
    df = simulation.copy()
    df["error"] = df["simulated_price"] - df["actual_price"]
    peak_error = float(df["simulated_price"].max() - df["actual_price"].max())
    final_error = float(df["simulated_price"].iloc[-1] - df["actual_price"].iloc[-1])
    full_rmse = rmse(df["error"])
    full_mae = mae(df["error"])
    high_rmse = rmse(df.loc[df["actual_price"] >= 110, "error"])
    low_rmse = rmse(df.loc[df["actual_price"] < 100, "error"])
    early_rmse = rmse(df.loc[df["day_index"] <= 14, "error"])
    late_rmse = rmse(df.loc[df["day_index"] > 35, "error"])
    mid_rmse = rmse(df.loc[(df["day_index"] > 14) & (df["day_index"] <= 35), "error"])

    composite_score = (
        full_rmse
        + 0.20 * abs(peak_error)
        + 0.25 * abs(final_error)
        + 0.15 * high_rmse
        + 0.12 * early_rmse
        + 0.12 * mid_rmse
        + 0.18 * late_rmse
        + 0.18 * low_rmse
    )
    platform_score = high_rmse + 0.40 * abs(peak_error) + 0.40 * abs(final_error)
    return {
        "RMSE": full_rmse,
        "MAE": full_mae,
        "峰值误差": peak_error,
        "末日误差": final_error,
        "高价平台RMSE": high_rmse,
        "低价回落RMSE": low_rmse,
        "前期RMSE": early_rmse,
        "中期RMSE": mid_rmse,
        "后期RMSE": late_rmse,
        "模拟峰值": float(df["simulated_price"].max()),
        "实际峰值": float(df["actual_price"].max()),
        "模拟末日价格": float(df["simulated_price"].iloc[-1]),
        "实际末日价格": float(df["actual_price"].iloc[-1]),
        "综合得分": composite_score,
        "平台解释得分": platform_score,
    }


def excellence_penalty(metrics: dict[str, float]) -> float:
    return (
        0.15 * max(metrics["前期RMSE"] - 5.0, 0) ** 2
        + 0.15 * max(metrics["中期RMSE"] - 5.0, 0) ** 2
        + 0.15 * max(metrics["后期RMSE"] - 5.0, 0) ** 2
        + 0.08 * max(abs(metrics["峰值误差"]) - 5.0, 0) ** 2
        + 0.10 * max(abs(metrics["末日误差"]) - 3.0, 0) ** 2
    )


def fit_quality_score(metrics: dict[str, float]) -> float:
    """Balanced score used for the final local fit refinement pass."""
    return (
        metrics["RMSE"]
        + 0.16 * metrics["中期RMSE"]
        + 0.14 * metrics["高价平台RMSE"]
        + 0.14 * metrics["低价回落RMSE"]
        + 0.10 * metrics["前期RMSE"]
        + 0.10 * metrics["后期RMSE"]
        + 0.10 * abs(metrics["峰值误差"])
        + 0.18 * abs(metrics["末日误差"])
    )
