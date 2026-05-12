"""Scenario forecast for 60-180 day oil-price paths.

This module remains the public command entry point:

    python -m src.scenarios.scenario_forecast

The implementation is split into smaller modules for settings, parameters,
simulation, and reporting. Names used by downstream analysis scripts are
re-exported here for compatibility.
"""

from __future__ import annotations

import pandas as pd

from src.common.paths import PROJECT_ROOT
from src.models import dynamic_short_term as dynamic
from src.scenarios.parameters import (
    build_parameter_table,
    build_scenario_parameters,
    calibrated_assumptions_and_behavior,
    load_best_row,
)
from src.scenarios.external_constraints import load_external_constraint_factors
from src.scenarios.reporting import build_report, save_figures, write_outputs
from src.scenarios.settings import (
    BUFFER_FIGURE,
    CALIBRATED_PATH,
    BEST_PARAMETERS_PATH,
    FORECAST_END_DAY,
    MARKER_DAYS,
    REPORT_PATH,
    SCENARIO_METRICS_CSV,
    SCENARIO_NAMES,
    SCENARIO_PARAMS_CSV,
    SCENARIO_PRICE_FIGURE,
    SCENARIO_RESULT_CSV,
)
from src.scenarios.simulation import (
    build_forecast_frame,
    infer_gap_closure_day,
    load_calibrated_prefix,
    run_scenario,
    simulate_future_from_prefix,
    summarize_scenario,
)


__all__ = [
    "BEST_PARAMETERS_PATH",
    "BUFFER_FIGURE",
    "CALIBRATED_PATH",
    "FORECAST_END_DAY",
    "MARKER_DAYS",
    "PROJECT_ROOT",
    "REPORT_PATH",
    "SCENARIO_METRICS_CSV",
    "SCENARIO_NAMES",
    "SCENARIO_PARAMS_CSV",
    "SCENARIO_PRICE_FIGURE",
    "SCENARIO_RESULT_CSV",
    "build_forecast_frame",
    "build_parameter_table",
    "build_report",
    "build_scenario_parameters",
    "calibrated_assumptions_and_behavior",
    "dynamic",
    "infer_gap_closure_day",
    "load_best_row",
    "load_calibrated_prefix",
    "run_scenario",
    "save_figures",
    "simulate_future_from_prefix",
    "summarize_scenario",
    "write_outputs",
]


def main() -> None:
    base_config = dynamic.load_yaml(dynamic.BASE_CONFIG_PATH)
    scenario_config = dynamic.load_yaml(dynamic.SCENARIO_CONFIG_PATH)
    paths = dynamic.resolve_paths(base_config)
    event_df = dynamic.load_event_window(paths.event_csv)
    best = load_best_row()
    base_assumptions, base_behavior = calibrated_assumptions_and_behavior(best)
    external_factors = load_external_constraint_factors()
    scenarios = build_scenario_parameters(base_assumptions, base_behavior, scenario_config, external_factors)
    forecast_frame = build_forecast_frame(event_df)
    prefix = load_calibrated_prefix()

    simulations = [
        run_scenario(key, assumptions, behavior, forecast_frame, prefix)
        for key, (assumptions, behavior) in scenarios.items()
    ]
    result = pd.concat(simulations, ignore_index=True)
    metrics = pd.DataFrame(summarize_scenario(sim) for sim in simulations)
    params = build_parameter_table(scenarios)

    write_outputs(result, metrics, params, event_df)

    print("Stage 5 complete")
    print(f"Scenario paths: {SCENARIO_RESULT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Scenario metrics: {SCENARIO_METRICS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Scenario params: {SCENARIO_PARAMS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Price figure: {SCENARIO_PRICE_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Buffer figure: {BUFFER_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
