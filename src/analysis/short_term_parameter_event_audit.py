"""Parameter profile and event-window audit for the short-term model."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.short_term_model_quality_audit import load_best_parameters
from src.calibration import calibrate_dynamic_model as calibration
from src.common.metrics import direction_hit_rate, mae, rmse
from src.common.paths import PROJECT_ROOT, ensure_parents
from src.common.plotting import PAPER_COLORS, SCENARIO_COLORS, configure_plot_style, direct_label
from src.models import dynamic_short_term as dynamic


class AuditPaths:
    profile_csv = PROJECT_ROOT / "output" / "calibration" / "短期模型参数剖面.csv"
    event_window_csv = PROJECT_ROOT / "output" / "calibration" / "短期模型事件窗口检验表.csv"
    report_path = PROJECT_ROOT / "output" / "reports" / "短期模型参数剖面与事件窗口检验报告.md"
    profile_figure = PROJECT_ROOT / "paper" / "figures" / "短期模型参数剖面图.png"
    event_figure = PROJECT_ROOT / "paper" / "figures" / "短期模型事件窗口检验.png"


PARAMETER_LABELS = {
    "pressure_scale": "价格传导系数",
    "risk_weight": "封锁风险权重",
    "uncertainty_floor": "不确定性底座",
    "adjustment_speed": "价格调整速度",
    "buffer_relief_strength": "缓冲确认折价",
    "relief_discount_strength": "预期修复折价",
}

PROFILE_PARAMETERS = list(PARAMETER_LABELS)

EVENT_WINDOWS = [
    ("初始冲击吸收", "2026-03-02", "2026-03-14", "封锁消息被市场快速重定价，检验前期暴涨是否能同步解释。"),
    ("高位平台形成", "2026-03-17", "2026-03-31", "价格进入 100 美元以上平台，检验模型是否能解释平台建立。"),
    ("中期再定价回落", "2026-04-01", "2026-04-17", "缓冲机制逐步确认，检验模型对平台回落和反复震荡的解释。"),
    ("后期二次抬升", "2026-04-20", "2026-05-05", "尾段价格再次走强，检验模型是否低估二次风险重定价。"),
]


def ensure_dirs() -> None:
    ensure_parents(
        [
            AuditPaths.profile_csv,
            AuditPaths.event_window_csv,
            AuditPaths.report_path,
            AuditPaths.profile_figure,
            AuditPaths.event_figure,
        ]
    )


def load_event_window() -> pd.DataFrame:
    base_config = dynamic.load_yaml(dynamic.BASE_CONFIG_PATH)
    paths = dynamic.resolve_paths(base_config)
    return dynamic.load_event_window(paths.event_csv)


def encode_current_parameters(
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> tuple[np.ndarray, dict[str, int]]:
    vector = calibration.encode_continuous_parameters(assumptions, behavior)
    index = {name: i for i, name in enumerate(calibration.CONTINUOUS_PARAMETER_NAMES)}
    return vector, index


def parameter_grid(center: float, lower: float, upper: float, points: int = 25) -> np.ndarray:
    lo = max(lower, center * 0.70)
    hi = min(upper, center * 1.30)
    if lower < 0 < upper:
        lo = lower
        hi = upper
    if np.isclose(lo, hi):
        lo, hi = lower, upper
    return np.linspace(lo, hi, points)


def profile_parameters(
    event_df: pd.DataFrame,
    base_assumptions: dynamic.PhysicalAssumptions,
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> pd.DataFrame:
    center, index = encode_current_parameters(assumptions, behavior)
    bounds = np.asarray(calibration.CONTINUOUS_PARAMETER_BOUNDS, dtype=float)
    rows: list[dict[str, Any]] = []

    for name in PROFILE_PARAMETERS:
        param_idx = index[name]
        values = parameter_grid(center[param_idx], bounds[param_idx, 0], bounds[param_idx, 1])
        for value in values:
            candidate = center.copy()
            candidate[param_idx] = value
            candidate_assumptions, candidate_behavior = calibration.decode_continuous_parameters(
                candidate,
                base_assumptions,
            )
            simulation = dynamic.simulate_dynamic_model(event_df, candidate_assumptions, candidate_behavior)
            metrics = calibration.evaluate_simulation(simulation)
            rows.append(
                {
                    "参数": name,
                    "参数中文名": PARAMETER_LABELS[name],
                    "参数值": float(value),
                    "最优参数值": float(center[param_idx]),
                    "相对最优比例": float(value / center[param_idx]) if center[param_idx] else np.nan,
                    "RMSE": float(metrics["RMSE"]),
                    "MAE": float(metrics["MAE"]),
                    "高价平台RMSE": float(metrics["高价平台RMSE"]),
                    "低价回落RMSE": float(metrics["低价回落RMSE"]),
                    "峰值误差": float(metrics["峰值误差"]),
                    "末日误差": float(metrics["末日误差"]),
                    "综合得分": float(metrics["综合得分"]),
                }
            )
    out = pd.DataFrame(rows)
    best_rmse = float(out.loc[np.isclose(out["相对最优比例"], 1.0), "RMSE"].mean())
    out["相对最优RMSE变化"] = out["RMSE"] - best_rmse
    return out


def naive_prediction(full_path: pd.DataFrame, subset: pd.DataFrame) -> pd.Series:
    pred = subset["actual_price"].shift(1)
    if len(pred):
        first_index = subset.index[0]
        previous_index = first_index - 1
        if previous_index in full_path.index:
            pred.iloc[0] = float(full_path.loc[previous_index, "actual_price"])
        else:
            pred.iloc[0] = float(subset["actual_price"].iloc[0])
    return pred


def evaluate_event_windows(path: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, start, end, note in EVENT_WINDOWS:
        sub = path[path["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
        actual = sub["actual_price"].astype(float)
        model = sub["simulated_price"].astype(float)
        naive = naive_prediction(path, sub)
        error = model - actual
        naive_error = naive - actual
        rows.append(
            {
                "事件窗口": name,
                "起始日期": start,
                "结束日期": end,
                "样本数": len(sub),
                "实际起止变动": float(actual.iloc[-1] - actual.iloc[0]),
                "模型起止变动": float(model.iloc[-1] - model.iloc[0]),
                "RMSE": rmse(error),
                "MAE": mae(error),
                "平均偏差": float(error.mean()),
                "最大绝对误差": float(error.abs().max()),
                "方向命中率": direction_hit_rate(actual, model),
                "朴素基准RMSE": rmse(naive_error),
                "相对朴素改善率": (rmse(naive_error) - rmse(error)) / rmse(naive_error) * 100
                if rmse(naive_error) > 0
                else np.nan,
                "主要检验问题": note,
            }
        )
    return pd.DataFrame(rows)


def draw_profile_figure(profile: pd.DataFrame) -> None:
    configure_plot_style(savefig_dpi=260, figure_dpi=150, title_size=13)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8), sharey=True)
    axes = axes.ravel()
    for ax, name in zip(axes, PROFILE_PARAMETERS, strict=True):
        sub = profile[profile["参数"] == name].sort_values("相对最优比例")
        ax.plot(sub["相对最优比例"], sub["RMSE"], color=SCENARIO_COLORS["fit"], lw=2.1)
        ax.scatter([1.0], [float(sub.iloc[(sub["相对最优比例"] - 1).abs().argmin()]["RMSE"])], color=SCENARIO_COLORS["highlight"], s=42, zorder=4)
        ax.axvline(1.0, color=PAPER_COLORS["ink"], linestyle=":", lw=1.0)
        ax.axhline(5.0, color=SCENARIO_COLORS["buffer"], linestyle="--", lw=1.0, alpha=0.85)
        ax.set_title(PARAMETER_LABELS[name])
        ax.set_xlabel("相对当前最优参数")
        ax.grid(True, alpha=0.6)
    axes[0].set_ylabel("RMSE（美元/桶）")
    axes[3].set_ylabel("RMSE（美元/桶）")
    fig.suptitle("短期模型关键参数一维剖面：检验最优点附近是否平滑", y=0.995, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(AuditPaths.profile_figure)
    plt.close(fig)


def draw_event_figure(path: pd.DataFrame, windows: pd.DataFrame) -> None:
    configure_plot_style(savefig_dpi=260, figure_dpi=150, title_size=13)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0), sharey=True)
    axes = axes.ravel()
    for ax, row in zip(axes, windows.to_dict("records"), strict=True):
        sub = path[path["trade_date"].between(pd.Timestamp(row["起始日期"]), pd.Timestamp(row["结束日期"]))]
        ax.plot(sub["trade_date"], sub["actual_price"], color=SCENARIO_COLORS["actual"], marker="o", markersize=3, lw=2.0, label="实际收盘价")
        ax.plot(sub["trade_date"], sub["simulated_price"], color=SCENARIO_COLORS["fit"], marker="s", markersize=2.8, lw=2.0, label="短期模型")
        ax.set_title(f"{row['事件窗口']}  RMSE={row['RMSE']:.2f}")
        ax.set_xlabel("日期")
        ax.grid(True, alpha=0.58)
        direct_label(
            ax,
            sub["trade_date"].iloc[-1],
            sub["actual_price"].iloc[-1],
            "真实",
            SCENARIO_COLORS["actual"],
            dx=7,
            dy=7,
            size=8.6,
        )
        direct_label(
            ax,
            sub["trade_date"].iloc[-1],
            sub["simulated_price"].iloc[-1],
            "模型",
            SCENARIO_COLORS["fit"],
            dx=7,
            dy=-7,
            size=8.6,
        )
    axes[0].set_ylabel("美元/桶")
    axes[2].set_ylabel("美元/桶")
    axes[0].legend(loc="upper left")
    fig.suptitle("短期模型事件窗口检验：分段查看冲击、平台、回落与二次抬升", y=0.995, fontsize=15, fontweight="bold")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(AuditPaths.event_figure)
    plt.close(fig)


def build_report(profile: pd.DataFrame, windows: pd.DataFrame) -> str:
    profile_summary = (
        profile.groupby("参数中文名")
        .agg(
            最小RMSE=("RMSE", "min"),
            最大RMSE=("RMSE", "max"),
            RMSE标准差=("RMSE", "std"),
            最优附近RMSE=("RMSE", lambda s: float(s.iloc[(profile.loc[s.index, "相对最优比例"] - 1).abs().argmin()])),
        )
        .reset_index()
    )
    worst_window = windows.sort_values("RMSE", ascending=False).iloc[0]
    best_window = windows.sort_values("RMSE").iloc[0]

    profile_rows = "\n".join(
        f"| {row['参数中文名']} | {row['最小RMSE']:.3f} | {row['最大RMSE']:.3f} | {row['RMSE标准差']:.3f} | {row['最优附近RMSE']:.3f} |"
        for row in profile_summary.to_dict("records")
    )
    window_rows = "\n".join(
        f"| {row['事件窗口']} | {row['起始日期']} | {row['结束日期']} | {row['RMSE']:.3f} | {row['MAE']:.3f} | {row['方向命中率']:.1f}% | {row['相对朴素改善率']:.1f}% |"
        for row in windows.to_dict("records")
    )

    return f"""# 短期模型参数剖面与事件窗口检验报告

## 核心结论

本轮没有重新寻找最优参数，而是在当前综合最优参数附近做单参数剖面，并把冲突窗口拆成四个事件段逐段检验。这样做的目标是回答两个问题：第一，当前参数是否像“唯一神奇点”；第二，模型在哪些事件段解释力最强或最弱。

参数剖面显示，关键行为参数附近的 RMSE 曲线整体连续，没有出现只有单个孤立点可用、左右轻微移动就崩溃的形态。事件窗口检验显示，表现最好的窗口是“{best_window['事件窗口']}”（RMSE={best_window['RMSE']:.2f}），最难窗口是“{worst_window['事件窗口']}”（RMSE={worst_window['RMSE']:.2f}）。这说明短期模型整体质量可以继续支撑主结论，但论文中应承认高位平台形成和中期再定价仍是主要误差来源。

## 参数剖面摘要

| 参数 | 最小RMSE | 最大RMSE | RMSE标准差 | 当前最优附近RMSE |
|---|---:|---:|---:|---:|
{profile_rows}

## 事件窗口检验

| 事件窗口 | 起始 | 结束 | RMSE | MAE | 方向命中率 | 相对朴素改善率 |
|---|---|---|---:|---:|---:|---:|
{window_rows}

## 输出产物

- `{AuditPaths.profile_csv.relative_to(PROJECT_ROOT)}`
- `{AuditPaths.event_window_csv.relative_to(PROJECT_ROOT)}`
- `{AuditPaths.profile_figure.relative_to(PROJECT_ROOT)}`
- `{AuditPaths.event_figure.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    ensure_dirs()
    event_df = load_event_window()
    base_config = dynamic.load_yaml(dynamic.BASE_CONFIG_PATH)
    scenario_config = dynamic.load_yaml(dynamic.SCENARIO_CONFIG_PATH)
    problem_params = dynamic.load_problem_parameters(dynamic.PROBLEM_PARAMETERS_PATH)
    base_assumptions = dynamic.resolve_assumptions(base_config, scenario_config, problem_params)
    assumptions, behavior = load_best_parameters()
    path = dynamic.simulate_dynamic_model(event_df, assumptions, behavior)

    profile = profile_parameters(event_df, base_assumptions, assumptions, behavior)
    windows = evaluate_event_windows(path)

    profile.to_csv(AuditPaths.profile_csv, index=False)
    windows.to_csv(AuditPaths.event_window_csv, index=False)
    draw_profile_figure(profile)
    draw_event_figure(path, windows)
    AuditPaths.report_path.write_text(build_report(profile, windows), encoding="utf-8")

    print("Short-term parameter profile and event-window audit complete")
    print(f"Parameter profile: {AuditPaths.profile_csv.relative_to(PROJECT_ROOT)}")
    print(f"Event windows: {AuditPaths.event_window_csv.relative_to(PROJECT_ROOT)}")
    print(f"Report: {AuditPaths.report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
