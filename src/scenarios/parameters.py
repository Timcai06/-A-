"""Stage 5 scenario parameter loading and transformation."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import pandas as pd

from src.models import dynamic_short_term as dynamic
from src.scenarios.external_constraints import (
    ExternalConstraintFactors,
    apply_external_constraints,
    load_external_constraint_factors,
)
from src.scenarios.settings import BEST_PARAMETERS_PATH, SCENARIO_NAMES


def load_best_row() -> pd.Series:
    if not BEST_PARAMETERS_PATH.exists():
        raise FileNotFoundError(f"Missing Stage 4 best-parameter file: {BEST_PARAMETERS_PATH}")
    best = pd.read_csv(BEST_PARAMETERS_PATH)
    composite = best[best["候选类型"].astype(str).str.contains("综合最优")]
    if not composite.empty:
        return composite.iloc[0]
    return best.sort_values(["综合得分", "RMSE"]).iloc[0]


def calibrated_assumptions_and_behavior(best: pd.Series) -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]:
    assumptions = dynamic.PhysicalAssumptions(
        base_supply=float(best["assumption_base_supply"]),
        base_demand=float(best["assumption_base_demand"]),
        base_elasticity=float(best["assumption_base_elasticity"]),
        long_elasticity=float(best["assumption_long_elasticity"]),
        supply_interruption=float(best["assumption_supply_interruption"]),
        spr_max_release=float(best["assumption_spr_max_release"]),
        spr_delay_days=int(best["assumption_spr_delay_days"]),
        spr_ramp_days=int(best["assumption_spr_ramp_days"]),
        route_start_day=int(best["assumption_route_start_day"]),
        route_max_capacity=float(best["assumption_route_max_capacity"]),
        route_ramp_days=int(best["assumption_route_ramp_days"]),
        commercial_inventory=float(best["assumption_commercial_inventory"]),
        inventory_daily_cap=float(best["assumption_inventory_daily_cap"]),
        observed_demand_decline=float(best["assumption_observed_demand_decline"]),
        demand_decline_ramp_days=int(best["assumption_demand_decline_ramp_days"]),
        fear_initial=float(best["assumption_fear_initial"]),
        fear_decay=float(best["assumption_fear_decay"]),
    )
    behavior = dynamic.BehavioralParameters(
        pressure_scale=float(best["behavior_pressure_scale"]),
        risk_weight=float(best["behavior_risk_weight"]),
        uncertainty_floor=float(best["behavior_uncertainty_floor"]),
        inventory_response=float(best["behavior_inventory_response"]),
        adjustment_speed=float(best["behavior_adjustment_speed"]),
        buffer_relief_strength=float(best["behavior_buffer_relief_strength"]),
        buffer_relief_decay_days=int(best["behavior_buffer_relief_decay_days"]),
        relief_discount_strength=float(best["behavior_relief_discount_strength"]),
        relief_start_day=int(best["behavior_relief_start_day"]),
        relief_peak_day=int(best["behavior_relief_peak_day"]),
        relief_decay_days=int(best["behavior_relief_decay_days"]),
    )
    return assumptions, behavior


def build_scenario_parameters(
    base_assumptions: dynamic.PhysicalAssumptions,
    base_behavior: dynamic.BehavioralParameters,
    scenario_config: dict[str, Any],
    external_factors: ExternalConstraintFactors | None = None,
) -> dict[str, tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]]:
    scenarios = scenario_config["scenarios"]
    factors = external_factors or load_external_constraint_factors()

    optimistic_cfg = scenarios["optimistic"]
    pessimistic_cfg = scenarios["pessimistic"]

    optimistic = (
        replace(
            base_assumptions,
            supply_interruption=float(optimistic_cfg["supply_interruption"]),
            spr_max_release=float(optimistic_cfg["spr_max_release"]),
            spr_delay_days=int(optimistic_cfg["spr_delay_days"]),
            spr_ramp_days=6,
            route_start_day=int(optimistic_cfg["route_start_day"]),
            route_max_capacity=float(optimistic_cfg["route_max_capacity"]),
            route_ramp_days=18,
            long_elasticity=float(optimistic_cfg["demand_elasticity_long_term"]),
            inventory_daily_cap=max(base_assumptions.inventory_daily_cap, 350.0),
            observed_demand_decline=520.0,
            demand_decline_ramp_days=18,
            fear_decay=float(optimistic_cfg["fear_decay"]),
        ),
        replace(
            base_behavior,
            pressure_scale=base_behavior.pressure_scale * 0.82,
            risk_weight=base_behavior.risk_weight * 0.82,
            uncertainty_floor=base_behavior.uncertainty_floor * 0.70,
            adjustment_speed=min(base_behavior.adjustment_speed * 1.08, 0.45),
            buffer_relief_strength=min(base_behavior.buffer_relief_strength * 1.15, 0.28),
            relief_discount_strength=min(base_behavior.relief_discount_strength * 1.05, 0.25),
            relief_start_day=22,
            relief_peak_day=42,
            relief_decay_days=20,
        ),
    )

    neutral = (
        replace(
            base_assumptions,
            supply_interruption=float(base_assumptions.supply_interruption),
            spr_max_release=float(base_assumptions.spr_max_release),
            spr_delay_days=int(base_assumptions.spr_delay_days),
            route_start_day=int(base_assumptions.route_start_day),
            route_max_capacity=float(base_assumptions.route_max_capacity),
            long_elasticity=float(base_assumptions.long_elasticity),
            fear_decay=float(base_assumptions.fear_decay),
        ),
        base_behavior,
    )

    pessimistic = (
        replace(
            base_assumptions,
            supply_interruption=float(pessimistic_cfg["supply_interruption"]),
            spr_max_release=float(pessimistic_cfg["spr_max_release"]),
            spr_delay_days=int(pessimistic_cfg["spr_delay_days"]),
            spr_ramp_days=12,
            route_start_day=int(pessimistic_cfg["route_start_day"]),
            route_max_capacity=float(pessimistic_cfg["route_max_capacity"]),
            route_ramp_days=42,
            long_elasticity=float(pessimistic_cfg["demand_elasticity_long_term"]),
            inventory_daily_cap=180.0,
            observed_demand_decline=300.0,
            demand_decline_ramp_days=45,
            fear_decay=float(pessimistic_cfg["fear_decay"]),
        ),
        replace(
            base_behavior,
            pressure_scale=base_behavior.pressure_scale * 1.35,
            risk_weight=base_behavior.risk_weight * 1.35,
            uncertainty_floor=min(base_behavior.uncertainty_floor * 1.35, 0.38),
            adjustment_speed=max(base_behavior.adjustment_speed * 0.92, 0.20),
            buffer_relief_strength=base_behavior.buffer_relief_strength * 0.35,
            relief_discount_strength=base_behavior.relief_discount_strength * 0.25,
            relief_start_day=65,
            relief_peak_day=110,
            relief_decay_days=45,
        ),
    )

    raw_scenarios = {
        "optimistic": optimistic,
        "neutral": neutral,
        "pessimistic": pessimistic,
    }
    return {
        key: apply_external_constraints(key, assumptions, behavior, factors)
        for key, (assumptions, behavior) in raw_scenarios.items()
    }


def build_parameter_table(
    scenarios: dict[str, tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, (assumptions, behavior) in scenarios.items():
        rows.append(
            {
                "scenario": key,
                "情景": SCENARIO_NAMES[key],
                **{f"assumption_{name}": value for name, value in asdict(assumptions).items()},
                **{f"behavior_{name}": value for name, value in asdict(behavior).items()},
            }
        )
    return pd.DataFrame(rows)
