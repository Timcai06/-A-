"""Search and refinement pipeline for Stage 4 calibration."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from src.calibration.evaluation import evaluate_simulation, excellence_penalty, fit_quality_score
from src.calibration.parameter_space import (
    CONTINUOUS_PARAMETER_BOUNDS,
    decode_continuous_parameters,
    encode_continuous_parameters,
    perturb_continuous_parameters,
    perturb_fit_refinement_parameters,
    sampled_parameter_sets,
)
from src.calibration.settings import (
    FIT_REFINEMENT_SAMPLES,
    LOCAL_REFINEMENT_MAXITER,
    LOCAL_REFINEMENT_POPSIZE,
    LOCAL_STABILITY_SAMPLES,
    RANDOM_SEED,
)
from src.models import dynamic_short_term as dynamic


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
    top_candidates = all_results.sort_values(["综合得分", "RMSE", "平台解释得分"]).head(10).reset_index(drop=True)
    representative = all_results[
        all_results["candidate_id"].isin([best_composite_id, best_rmse_id, best_platform_id])
    ].sort_values(["综合得分", "RMSE"])
    best_simulation = simulations[best_composite_id].copy()
    best_simulation["candidate_id"] = best_composite_id
    return best_simulation, top_candidates, representative
