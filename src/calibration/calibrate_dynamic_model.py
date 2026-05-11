"""Stage 4 multi-objective calibration for the short-term dynamic model.

This module remains the public command entry point:

    python -m src.calibration.calibrate_dynamic_model

The implementation is split into smaller modules so analysis scripts can reuse
paths, parameter encoders, evaluation metrics, and report writers without
depending on one oversized file.
"""

from __future__ import annotations

from src.calibration.evaluation import evaluate_simulation, excellence_penalty, fit_quality_score, segment_error_rows
from src.calibration.parameter_space import (
    CONTINUOUS_PARAMETER_BOUNDS,
    CONTINUOUS_PARAMETER_NAMES,
    decode_continuous_parameters,
    encode_continuous_parameters,
    perturb_continuous_parameters,
    perturb_fit_refinement_parameters,
    sampled_parameter_sets,
)
from src.calibration.reporting import build_report, save_figure, write_outputs, write_report
from src.calibration.search import (
    calibrate,
    classify_candidate,
    refine_with_continuous_search,
    refine_with_fit_quality_search,
    refine_with_local_stability_search,
)
from src.calibration.settings import (
    FIT_REFINEMENT_SAMPLES,
    LOCAL_REFINEMENT_MAXITER,
    LOCAL_REFINEMENT_POPSIZE,
    LOCAL_STABILITY_SAMPLES,
    PROJECT_ROOT,
    RANDOM_SEED,
    SAMPLE_SIZE,
    Stage4Paths,
)
from src.models import dynamic_short_term as dynamic


__all__ = [
    "CONTINUOUS_PARAMETER_BOUNDS",
    "CONTINUOUS_PARAMETER_NAMES",
    "FIT_REFINEMENT_SAMPLES",
    "LOCAL_REFINEMENT_MAXITER",
    "LOCAL_REFINEMENT_POPSIZE",
    "LOCAL_STABILITY_SAMPLES",
    "PROJECT_ROOT",
    "RANDOM_SEED",
    "SAMPLE_SIZE",
    "Stage4Paths",
    "build_report",
    "calibrate",
    "classify_candidate",
    "decode_continuous_parameters",
    "encode_continuous_parameters",
    "evaluate_simulation",
    "excellence_penalty",
    "fit_quality_score",
    "perturb_continuous_parameters",
    "perturb_fit_refinement_parameters",
    "refine_with_continuous_search",
    "refine_with_fit_quality_search",
    "refine_with_local_stability_search",
    "sampled_parameter_sets",
    "save_figure",
    "segment_error_rows",
    "write_outputs",
    "write_report",
]


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
