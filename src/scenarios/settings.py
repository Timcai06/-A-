"""Stage 5 scenario forecast constants and output paths."""

from __future__ import annotations

from src.common.paths import PROJECT_ROOT


BEST_PARAMETERS_PATH = PROJECT_ROOT / "output" / "calibration" / "动态模型最优参数.csv"
CALIBRATED_PATH = PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv"
SCENARIO_RESULT_CSV = PROJECT_ROOT / "output" / "scenarios" / "三情景预测结果.csv"
SCENARIO_METRICS_CSV = PROJECT_ROOT / "output" / "scenarios" / "三情景关键指标.csv"
SCENARIO_PARAMS_CSV = PROJECT_ROOT / "output" / "scenarios" / "三情景参数表.csv"
EXTERNAL_CONSTRAINTS_CSV = PROJECT_ROOT / "output" / "scenarios" / "官方外生约束参数因子.csv"
SCENARIO_PRICE_FIGURE = PROJECT_ROOT / "figures" / "scenario_price_paths.png"
BUFFER_FIGURE = PROJECT_ROOT / "figures" / "inventory_depletion_risk.png"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "长期三情景预测报告.md"
FORECAST_END_DAY = 180
MARKER_DAYS = [60, 90, 120, 180]

SCENARIO_NAMES = {
    "optimistic": "乐观情景",
    "neutral": "中性情景",
    "pessimistic": "悲观情景",
}
