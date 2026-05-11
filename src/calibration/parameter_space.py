"""Parameter encoding, decoding, and perturbation helpers for calibration."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np

from src.calibration.settings import RANDOM_SEED, SAMPLE_SIZE
from src.models import dynamic_short_term as dynamic


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
