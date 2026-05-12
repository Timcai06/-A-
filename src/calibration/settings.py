"""Stage 4 calibration constants and output paths."""

from __future__ import annotations

from src.common.paths import PROJECT_ROOT


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
    report_path = PROJECT_ROOT / "output" / "reports" / "短期模型参数校准报告.md"
    figure_path = PROJECT_ROOT / "figures" / "fitted_vs_actual.png"
