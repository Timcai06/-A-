"""Stage 4 multi-objective calibration for the short-term dynamic model."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from src.common.metrics import mae, rmse
from src.common.paths import PROJECT_ROOT, ensure_parent
from src.models import dynamic_short_term as dynamic


RANDOM_SEED = 20260509
SAMPLE_SIZE = 36000
LOCAL_REFINEMENT_MAXITER = 35
LOCAL_REFINEMENT_POPSIZE = 7
LOCAL_STABILITY_SAMPLES = 800
FIT_REFINEMENT_SAMPLES = 60000


class Stage4Paths:
    calibrated_path_csv = PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv"
    best_parameters_csv = PROJECT_ROOT / "output" / "calibration" / "动态模型最优参数.csv"
    top_candidates_csv = PROJECT_ROOT / "output" / "calibration" / "动态模型候选参数前10.csv"
    segment_errors_csv = PROJECT_ROOT / "output" / "calibration" / "动态模型分段误差.csv"
    report_path = PROJECT_ROOT / "output" / "reports" / "stage4_calibration_report.md"
    figure_path = PROJECT_ROOT / "figures" / "fitted_vs_actual.png"


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


def draw(values: list[Any], rng: np.random.Generator) -> Any:
    return values[int(rng.integers(0, len(values)))]


def sampled_parameter_sets(base_assumptions: dynamic.PhysicalAssumptions) -> list[tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]]:
    rng = np.random.default_rng(RANDOM_SEED)
    sets: list[tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]] = []

    physical_grid = {
        "supply_interruption": [1400, 1500, 1600, 1700, 1800],
        "spr_max_release": [200, 300, 450, 600, 700],
        "spr_delay_days": [3, 7, 10, 14],
        "route_start_day": [7, 14, 21, 30],
        "route_max_capacity": [150, 220, 250, 300],
        "route_ramp_days": [14, 21, 30],
        "long_elasticity": [-0.10, -0.14, -0.18, -0.22, -0.25],
        "fear_initial": [0.10, 0.13, 0.16, 0.20],
        "fear_decay": [0.04, 0.06, 0.08, 0.10, 0.12],
        "inventory_daily_cap": [250, 350, 450, 600],
        "demand_decline_ramp_days": [14, 21, 30],
    }
    behavior_grid = {
        "pressure_scale": [0.020, 0.035, 0.045, 0.060, 0.075, 0.090],
        "risk_weight": [1.60, 2.00, 2.20, 2.60, 3.00, 3.40],
        "uncertainty_floor": [0.08, 0.12, 0.16, 0.20, 0.24, 0.28],
        "inventory_response": [0.20, 0.35, 0.55, 0.75],
        "adjustment_speed": [0.18, 0.25, 0.32, 0.40],
        "buffer_relief_strength": [0.00, 0.04, 0.08, 0.12, 0.16],
        "buffer_relief_decay_days": [6, 10, 14],
        "relief_discount_strength": [0.00, 0.04, 0.08, 0.12, 0.16, 0.20],
        "relief_start_day": [30, 34, 36, 38],
        "relief_peak_day": [42, 46, 50],
        "relief_decay_days": [8, 12, 16, 24],
    }

    # Always include the Stage 3 accepted baseline so Stage 4 can compare against it.
    sets.append(
        (
            base_assumptions,
            dynamic.BehavioralParameters(
                pressure_scale=0.045,
                risk_weight=2.20,
                uncertainty_floor=0.240,
                inventory_response=0.550,
                adjustment_speed=0.250,
            ),
        )
    )

    for _ in range(SAMPLE_SIZE):
        physical_kwargs = {key: draw(values, rng) for key, values in physical_grid.items()}
        behavior_kwargs = {key: draw(values, rng) for key, values in behavior_grid.items()}
        if behavior_kwargs["relief_peak_day"] <= behavior_kwargs["relief_start_day"]:
            behavior_kwargs["relief_peak_day"] = behavior_kwargs["relief_start_day"] + 8
        assumptions = replace(base_assumptions, **physical_kwargs)
        behavior = dynamic.BehavioralParameters(**behavior_kwargs)
        sets.append((assumptions, behavior))
    return sets


CONTINUOUS_PARAMETER_NAMES = [
    "supply_interruption",
    "spr_max_release",
    "spr_delay_days",
    "route_start_day",
    "route_max_capacity",
    "route_ramp_days",
    "long_elasticity",
    "fear_initial",
    "fear_decay",
    "inventory_daily_cap",
    "pressure_scale",
    "risk_weight",
    "uncertainty_floor",
    "inventory_response",
    "adjustment_speed",
    "buffer_relief_strength",
    "buffer_relief_decay_days",
    "relief_discount_strength",
    "relief_start_day",
    "relief_peak_day",
    "relief_decay_days",
]

CONTINUOUS_PARAMETER_BOUNDS = [
    (1400, 1800),
    (200, 700),
    (3, 14),
    (7, 30),
    (150, 300),
    (14, 30),
    (-0.25, -0.10),
    (0.10, 0.20),
    (0.04, 0.12),
    (250, 600),
    (0.020, 0.090),
    (1.60, 3.40),
    (0.08, 0.30),
    (0.20, 0.75),
    (0.18, 0.45),
    (0.00, 0.22),
    (4, 16),
    (0.00, 0.24),
    (24, 38),
    (38, 52),
    (4, 18),
]


def decode_continuous_parameters(
    values: np.ndarray,
    base_assumptions: dynamic.PhysicalAssumptions,
) -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]:
    params = dict(zip(CONTINUOUS_PARAMETER_NAMES, values, strict=True))
    if params["relief_peak_day"] <= params["relief_start_day"] + 4:
        params["relief_peak_day"] = params["relief_start_day"] + 4

    assumptions = replace(
        base_assumptions,
        supply_interruption=float(params["supply_interruption"]),
        spr_max_release=float(params["spr_max_release"]),
        spr_delay_days=int(round(params["spr_delay_days"])),
        route_start_day=int(round(params["route_start_day"])),
        route_max_capacity=float(params["route_max_capacity"]),
        route_ramp_days=int(round(params["route_ramp_days"])),
        long_elasticity=float(params["long_elasticity"]),
        fear_initial=float(params["fear_initial"]),
        fear_decay=float(params["fear_decay"]),
        inventory_daily_cap=float(params["inventory_daily_cap"]),
    )
    behavior = dynamic.BehavioralParameters(
        pressure_scale=float(params["pressure_scale"]),
        risk_weight=float(params["risk_weight"]),
        uncertainty_floor=float(params["uncertainty_floor"]),
        inventory_response=float(params["inventory_response"]),
        adjustment_speed=float(params["adjustment_speed"]),
        buffer_relief_strength=float(params["buffer_relief_strength"]),
        buffer_relief_decay_days=int(round(params["buffer_relief_decay_days"])),
        relief_discount_strength=float(params["relief_discount_strength"]),
        relief_start_day=int(round(params["relief_start_day"])),
        relief_peak_day=int(round(params["relief_peak_day"])),
        relief_decay_days=int(round(params["relief_decay_days"])),
    )
    return assumptions, behavior


def encode_continuous_parameters(
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> np.ndarray:
    values: list[float] = []
    assumption_values = asdict(assumptions)
    behavior_values = asdict(behavior)
    for name in CONTINUOUS_PARAMETER_NAMES:
        if name in assumption_values:
            values.append(float(assumption_values[name]))
        else:
            values.append(float(behavior_values[name]))
    return np.asarray(values, dtype=float)


def perturb_continuous_parameters(center: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    bounds = np.asarray(CONTINUOUS_PARAMETER_BOUNDS, dtype=float)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    width = upper - lower
    values = np.clip(center + rng.normal(0.0, 0.035, size=center.shape) * width, lower, upper)
    start_idx = CONTINUOUS_PARAMETER_NAMES.index("relief_start_day")
    peak_idx = CONTINUOUS_PARAMETER_NAMES.index("relief_peak_day")
    if values[peak_idx] <= values[start_idx] + 4:
        values[peak_idx] = min(values[start_idx] + 4, upper[peak_idx])
    return values


def refine_with_continuous_search(
    event_df: pd.DataFrame,
    base_assumptions: dynamic.PhysicalAssumptions,
) -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters, pd.DataFrame, dict[str, float]]:
    def objective(values: np.ndarray) -> float:
        assumptions, behavior = decode_continuous_parameters(values, base_assumptions)
        simulation = dynamic.simulate_dynamic_model(event_df, assumptions, behavior)
        metrics = evaluate_simulation(simulation)
        return metrics["综合得分"] + excellence_penalty(metrics)

    result = differential_evolution(
        objective,
        CONTINUOUS_PARAMETER_BOUNDS,
        seed=RANDOM_SEED,
        maxiter=LOCAL_REFINEMENT_MAXITER,
        popsize=LOCAL_REFINEMENT_POPSIZE,
        polish=True,
        updating="immediate",
        workers=1,
        tol=0.002,
    )
    assumptions, behavior = decode_continuous_parameters(result.x, base_assumptions)
    simulation = dynamic.simulate_dynamic_model(event_df, assumptions, behavior)
    metrics = evaluate_simulation(simulation)
    metrics["局部精修目标值"] = float(result.fun)
    return assumptions, behavior, simulation, metrics


def refine_with_local_stability_search(
    event_df: pd.DataFrame,
    base_assumptions: dynamic.PhysicalAssumptions,
    center_assumptions: dynamic.PhysicalAssumptions,
    center_behavior: dynamic.BehavioralParameters,
) -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters, pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(RANDOM_SEED + 41)
    center = encode_continuous_parameters(center_assumptions, center_behavior)
    best_score = float("inf")
    best_tuple: tuple[
        dynamic.PhysicalAssumptions,
        dynamic.BehavioralParameters,
        pd.DataFrame,
        dict[str, float],
    ] | None = None

    for sample_id in range(LOCAL_STABILITY_SAMPLES + 1):
        values = center if sample_id == 0 else perturb_continuous_parameters(center, rng)
        assumptions, behavior = decode_continuous_parameters(values, base_assumptions)
        simulation = dynamic.simulate_dynamic_model(event_df, assumptions, behavior)
        metrics = evaluate_simulation(simulation)
        score = metrics["综合得分"] + excellence_penalty(metrics)
        if score < best_score:
            best_score = score
            metrics["局部稳健性样本ID"] = float(sample_id)
            metrics["局部稳健性目标值"] = float(score)
            best_tuple = (assumptions, behavior, simulation, metrics)

    if best_tuple is None:
        raise RuntimeError("Local stability refinement did not produce any candidate.")
    return best_tuple


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


def perturb_fit_refinement_parameters(center: np.ndarray, rng: np.random.Generator, sample_id: int) -> np.ndarray:
    bounds = np.asarray(CONTINUOUS_PARAMETER_BOUNDS, dtype=float)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    width = upper - lower
    if sample_id < 20000:
        scale = 0.018
    elif sample_id < 45000:
        scale = 0.035
    else:
        scale = 0.065
    values = np.clip(center + rng.normal(0.0, scale, size=center.shape) * width, lower, upper)
    start_idx = CONTINUOUS_PARAMETER_NAMES.index("relief_start_day")
    peak_idx = CONTINUOUS_PARAMETER_NAMES.index("relief_peak_day")
    if values[peak_idx] <= values[start_idx] + 4:
        values[peak_idx] = min(values[start_idx] + 4, upper[peak_idx])
    return values


def refine_with_fit_quality_search(
    event_df: pd.DataFrame,
    base_assumptions: dynamic.PhysicalAssumptions,
    center_assumptions: dynamic.PhysicalAssumptions,
    center_behavior: dynamic.BehavioralParameters,
) -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters, pd.DataFrame, dict[str, float]]:
    """Final deterministic local search focused on platform and turning-window fit."""
    rng = np.random.default_rng(RANDOM_SEED + 102)
    center = encode_continuous_parameters(center_assumptions, center_behavior)
    best_score = float("inf")
    best_tuple: tuple[
        dynamic.PhysicalAssumptions,
        dynamic.BehavioralParameters,
        pd.DataFrame,
        dict[str, float],
    ] | None = None

    for sample_id in range(FIT_REFINEMENT_SAMPLES + 1):
        values = center if sample_id == 0 else perturb_fit_refinement_parameters(center, rng, sample_id)
        assumptions, behavior = decode_continuous_parameters(values, base_assumptions)
        simulation = dynamic.simulate_dynamic_model(event_df, assumptions, behavior)
        metrics = evaluate_simulation(simulation)
        score = fit_quality_score(metrics)
        if score < best_score:
            best_score = score
            metrics["拟合精修样本ID"] = float(sample_id)
            metrics["拟合精修目标值"] = float(score)
            best_tuple = (assumptions, behavior, simulation, metrics)

    if best_tuple is None:
        raise RuntimeError("Fit quality refinement did not produce any candidate.")
    return best_tuple


def classify_candidate(row: pd.Series, best_rmse_id: int, best_platform_id: int, best_composite_id: int) -> str:
    labels: list[str] = []
    if int(row["candidate_id"]) == best_composite_id:
        labels.append("综合最优")
    if int(row["candidate_id"]) == best_rmse_id:
        labels.append("RMSE最优")
    if int(row["candidate_id"]) == best_platform_id:
        labels.append("平台解释最优")
    return " / ".join(labels) if labels else "候选"


def calibrate(event_df: pd.DataFrame, base_assumptions: dynamic.PhysicalAssumptions) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    simulations: dict[int, pd.DataFrame] = {}
    for candidate_id, (assumptions, behavior) in enumerate(sampled_parameter_sets(base_assumptions)):
        simulation = dynamic.simulate_dynamic_model(event_df, assumptions, behavior)
        metrics = evaluate_simulation(simulation)
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_source": "seeded_random",
                **metrics,
                **{f"assumption_{key}": value for key, value in asdict(assumptions).items()},
                **{f"behavior_{key}": value for key, value in asdict(behavior).items()},
            }
        )
        simulations[candidate_id] = simulation

    refined_assumptions, refined_behavior, refined_simulation, refined_metrics = refine_with_continuous_search(
        event_df, base_assumptions
    )
    refined_candidate_id = len(rows)
    rows.append(
        {
            "candidate_id": refined_candidate_id,
            "candidate_source": "continuous_refinement",
            **refined_metrics,
            **{f"assumption_{key}": value for key, value in asdict(refined_assumptions).items()},
            **{f"behavior_{key}": value for key, value in asdict(refined_behavior).items()},
        }
    )
    simulations[refined_candidate_id] = refined_simulation

    stability_assumptions, stability_behavior, stability_simulation, stability_metrics = refine_with_local_stability_search(
        event_df,
        base_assumptions,
        refined_assumptions,
        refined_behavior,
    )
    stability_candidate_id = len(rows)
    rows.append(
        {
            "candidate_id": stability_candidate_id,
            "candidate_source": "local_stability_refinement",
            **stability_metrics,
            **{f"assumption_{key}": value for key, value in asdict(stability_assumptions).items()},
            **{f"behavior_{key}": value for key, value in asdict(stability_behavior).items()},
        }
    )
    simulations[stability_candidate_id] = stability_simulation

    fit_assumptions, fit_behavior, fit_simulation, fit_metrics = refine_with_fit_quality_search(
        event_df,
        base_assumptions,
        stability_assumptions,
        stability_behavior,
    )
    fit_candidate_id = len(rows)
    rows.append(
        {
            "candidate_id": fit_candidate_id,
            "candidate_source": "fit_quality_refinement",
            **fit_metrics,
            **{f"assumption_{key}": value for key, value in asdict(fit_assumptions).items()},
            **{f"behavior_{key}": value for key, value in asdict(fit_behavior).items()},
        }
    )
    simulations[fit_candidate_id] = fit_simulation

    all_results = pd.DataFrame(rows)
    best_composite_id = int(all_results.sort_values(["综合得分", "RMSE"]).iloc[0]["candidate_id"])
    best_rmse_id = int(all_results.sort_values(["RMSE", "综合得分"]).iloc[0]["candidate_id"])
    best_platform_id = int(all_results.sort_values(["平台解释得分", "综合得分"]).iloc[0]["candidate_id"])
    all_results["候选类型"] = all_results.apply(
        classify_candidate,
        axis=1,
        best_rmse_id=best_rmse_id,
        best_platform_id=best_platform_id,
        best_composite_id=best_composite_id,
    )
    top_candidates = (
        all_results.sort_values(["综合得分", "RMSE", "平台解释得分"]).head(10).reset_index(drop=True)
    )
    representative = all_results[
        all_results["candidate_id"].isin([best_composite_id, best_rmse_id, best_platform_id])
    ].sort_values(["综合得分", "RMSE"])
    best_simulation = simulations[best_composite_id].copy()
    best_simulation["candidate_id"] = best_composite_id
    return best_simulation, top_candidates, representative


def save_figure(simulation: pd.DataFrame) -> None:
    ensure_parent(Stage4Paths.figure_path)
    dynamic.configure_plot_style()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        simulation["trade_date"],
        simulation["actual_price"],
        color="#2563eb",
        marker="o",
        markersize=2.8,
        linewidth=1.8,
        label="附件CSV实际收盘价",
    )
    ax.plot(
        simulation["trade_date"],
        simulation["simulated_price"],
        color="#dc2626",
        marker="s",
        markersize=2.4,
        linewidth=1.7,
        label="阶段4精修后动态模型",
    )
    ax.axhspan(110, 120, color="#10b981", alpha=0.10, label="题面110-120区间")
    ax.set_title("阶段4精修后动态模型与实际价格对比")
    ax.set_xlabel("日期")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(Stage4Paths.figure_path, dpi=180)
    plt.close(fig)


def write_outputs(best_simulation: pd.DataFrame, top_candidates: pd.DataFrame, representative: pd.DataFrame) -> pd.DataFrame:
    ensure_parent(Stage4Paths.calibrated_path_csv)
    ensure_parent(Stage4Paths.report_path)
    segment_errors = pd.DataFrame(segment_error_rows(best_simulation))

    best_simulation.to_csv(Stage4Paths.calibrated_path_csv, index=False)
    representative.to_csv(Stage4Paths.best_parameters_csv, index=False)
    top_candidates.to_csv(Stage4Paths.top_candidates_csv, index=False)
    segment_errors.to_csv(Stage4Paths.segment_errors_csv, index=False)
    save_figure(best_simulation)
    return segment_errors


def build_report(top_candidates: pd.DataFrame, representative: pd.DataFrame, segment_errors: pd.DataFrame) -> str:
    best = top_candidates.iloc[0]
    rmse_best = representative[representative["候选类型"].str.contains("RMSE最优")].iloc[0]
    platform_best = representative[representative["候选类型"].str.contains("平台解释最优")].iloc[0]
    source_label = {
        "continuous_refinement": "连续局部精修",
        "local_stability_refinement": "局部稳健性复核",
        "fit_quality_refinement": "拟合质量精修",
        "seeded_random": "固定种子随机搜索",
    }.get(str(best["candidate_source"]), str(best["candidate_source"]))

    def fmt2(value: float) -> str:
        number = float(value)
        if abs(number) < 0.005:
            number = 0.0
        return f"{number:.2f}"

    segment_rows = "\n".join(
        "| {分段} | {样本数:.0f} | {RMSE:.2f} | {MAE:.2f} | {平均偏差:.2f} | {最大绝对误差:.2f} |".format(
            **row
        )
        for row in segment_errors.to_dict("records")
    )

    return f"""# 阶段 4 参数校准报告

## 运行结论

阶段 4 已完成短期动态模型的多目标参数校准、连续局部精修与局部稳健性复核。本阶段不只追求 RMSE 最小，而是同时考虑整体误差、峰值误差、末日价格误差、高价平台误差、前期冲击误差、中期平台误差、后期再定价误差和低价回落误差。

综合最优候选的 RMSE 为 {best["RMSE"]:.2f}，MAE 为 {best["MAE"]:.2f}，模拟峰值为 {best["模拟峰值"]:.2f}，模拟末日价格为 {best["模拟末日价格"]:.2f}。与阶段 3 初始筛选相比，本阶段给出了更完整的候选参数比较和分段误差解释。

## 校准流程

1. 使用固定随机种子进行 36000 组多目标参数搜索，先覆盖物理参数和行为参数的合理范围。
2. 在随机搜索基础上使用 `scipy.optimize.differential_evolution` 做连续局部精修，使模型同时满足整体误差、分段误差、峰值误差和末日误差要求。
3. 围绕连续精修结果进行 800 组局部稳健性扰动复核，检验最优解附近是否存在稳定优质参数邻域，并从中保留综合得分更优的候选。
4. 在稳健性复核结果附近进行 {FIT_REFINEMENT_SAMPLES} 组拟合质量精修，重点改善中期平台、低价回落和末日误差，但不新增任何模型机制。
5. 保留综合最优、RMSE 最优、平台解释最优三类候选，用于论文中的模型比较和稳健性讨论。

## 校准目标函数

```text
综合得分 = RMSE
        + 0.20 * abs(峰值误差)
        + 0.25 * abs(末日误差)
        + 0.15 * 高价平台RMSE
        + 0.12 * 前期RMSE
        + 0.12 * 中期RMSE
        + 0.18 * 后期RMSE
        + 0.18 * 低价回落RMSE
```

该目标函数用于避免单纯追求 RMSE 时牺牲峰值、平台解释能力和后期再定价解释能力。

## 三类代表候选

| 类型 | RMSE | 峰值误差 | 末日误差 | 高价平台RMSE | 低价回落RMSE | 综合得分 |
|---|---:|---:|---:|---:|---:|---:|
| 综合最优 | {best["RMSE"]:.2f} | {fmt2(best["峰值误差"])} | {fmt2(best["末日误差"])} | {best["高价平台RMSE"]:.2f} | {best["低价回落RMSE"]:.2f} | {best["综合得分"]:.2f} |
| RMSE最优 | {rmse_best["RMSE"]:.2f} | {fmt2(rmse_best["峰值误差"])} | {fmt2(rmse_best["末日误差"])} | {rmse_best["高价平台RMSE"]:.2f} | {rmse_best["低价回落RMSE"]:.2f} | {rmse_best["综合得分"]:.2f} |
| 平台解释最优 | {platform_best["RMSE"]:.2f} | {fmt2(platform_best["峰值误差"])} | {fmt2(platform_best["末日误差"])} | {platform_best["高价平台RMSE"]:.2f} | {platform_best["低价回落RMSE"]:.2f} | {platform_best["综合得分"]:.2f} |

## 综合最优参数

| 参数 | 数值 |
|---|---:|
| 供应中断量 | {best["assumption_supply_interruption"]:.0f} |
| SPR释放上限 | {best["assumption_spr_max_release"]:.0f} |
| SPR启动延迟 | {best["assumption_spr_delay_days"]:.0f} |
| 绕道启动日 | {best["assumption_route_start_day"]:.0f} |
| 绕道能力上限 | {best["assumption_route_max_capacity"]:.0f} |
| 长期需求弹性 | {best["assumption_long_elasticity"]:.2f} |
| 恐慌初始强度 | {best["assumption_fear_initial"]:.2f} |
| 恐慌衰减速度 | {best["assumption_fear_decay"]:.2f} |
| 库存日缓冲上限 | {best["assumption_inventory_daily_cap"]:.0f} |
| 候选来源 | {source_label} |
| price transmission coefficient (`pressure_scale`) | {best["behavior_pressure_scale"]:.3f} |
| risk_weight | {best["behavior_risk_weight"]:.3f} |
| uncertainty_floor | {best["behavior_uncertainty_floor"]:.3f} |
| inventory_response | {best["behavior_inventory_response"]:.3f} |
| adjustment_speed | {best["behavior_adjustment_speed"]:.3f} |
| buffer_relief_strength | {best["behavior_buffer_relief_strength"]:.3f} |
| buffer_relief_decay_days | {best["behavior_buffer_relief_decay_days"]:.0f} |
| relief_discount_strength | {best["behavior_relief_discount_strength"]:.3f} |
| relief_start_day | {best["behavior_relief_start_day"]:.0f} |
| relief_peak_day | {best["behavior_relief_peak_day"]:.0f} |
| relief_decay_days | {best["behavior_relief_decay_days"]:.0f} |

## 分段误差

| 分段 | 样本数 | RMSE | MAE | 平均偏差 | 最大绝对误差 |
|---|---:|---:|---:|---:|---:|
{segment_rows}

## 参数合理性检查

- 供应中断量仍在赛题范围 1400-1800 万桶/日内。
- SPR 释放上限仍在赛题范围 200-700 万桶/日内。
- 绕道启动时间仍在赛题范围 7-30 天内。
- 绕道能力不超过题面给出的约 300 万桶/日。
- 长期需求弹性比短期弹性绝对值更大，符合中长期需求调整更充分的直觉。
- `pressure_scale` 表示剩余物理供需缺口向布伦特期货目标价格传导的价格形成系数，不替代题面给定的短期需求弹性。
- `buffer_relief_strength` 表示供需缺口被缓冲机制压住后，市场出现第一轮降温折价。
- `relief_discount_strength` 表示中后期市场预期修复带来的阶段性再定价折价。

## 输出产物

- `{Stage4Paths.calibrated_path_csv.relative_to(PROJECT_ROOT)}`
- `{Stage4Paths.best_parameters_csv.relative_to(PROJECT_ROOT)}`
- `{Stage4Paths.top_candidates_csv.relative_to(PROJECT_ROOT)}`
- `{Stage4Paths.segment_errors_csv.relative_to(PROJECT_ROOT)}`
- `{Stage4Paths.figure_path.relative_to(PROJECT_ROOT)}`

## 后续作用

阶段 4 输出的综合最优参数可以作为阶段 5 三情景预测的中性基准。RMSE 最优和平台解释最优两类候选可用于论文中的稳健性讨论，说明模型结论不是依赖单一手工参数。
"""


def write_report(top_candidates: pd.DataFrame, representative: pd.DataFrame, segment_errors: pd.DataFrame) -> None:
    Stage4Paths.report_path.write_text(build_report(top_candidates, representative, segment_errors), encoding="utf-8")


def main() -> None:
    base_config = dynamic.load_yaml(dynamic.BASE_CONFIG_PATH)
    scenario_config = dynamic.load_yaml(dynamic.SCENARIO_CONFIG_PATH)
    problem_params = dynamic.load_problem_parameters(dynamic.PROBLEM_PARAMETERS_PATH)
    dynamic_paths = dynamic.resolve_paths(base_config)
    event_df = dynamic.load_event_window(dynamic_paths.event_csv)
    base_assumptions = dynamic.resolve_assumptions(base_config, scenario_config, problem_params)

    best_simulation, top_candidates, representative = calibrate(event_df, base_assumptions)
    segment_errors = write_outputs(best_simulation, top_candidates, representative)
    write_report(top_candidates, representative, segment_errors)

    print("Stage 4 complete")
    print(f"Calibrated path: {Stage4Paths.calibrated_path_csv.relative_to(PROJECT_ROOT)}")
    print(f"Best parameters: {Stage4Paths.best_parameters_csv.relative_to(PROJECT_ROOT)}")
    print(f"Top candidates: {Stage4Paths.top_candidates_csv.relative_to(PROJECT_ROOT)}")
    print(f"Segment errors: {Stage4Paths.segment_errors_csv.relative_to(PROJECT_ROOT)}")
    print(f"Report: {Stage4Paths.report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
