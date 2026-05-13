#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BUILD_DIR="output/build/latex"
FINAL_DIR="output/final"
FINAL_PAPER_BASENAME="A题_霍尔木兹海峡封锁对国际原油价格影响_论文"
FINAL_PDF="${FINAL_DIR}/${FINAL_PAPER_BASENAME}.pdf"
FINAL_DOCX="${FINAL_DIR}/${FINAL_PAPER_BASENAME}.docx"

mkdir -p "${BUILD_DIR}" "${FINAL_DIR}"
mkdir -p "tmp/matplotlib" "tmp/cache"
export MPLCONFIGDIR="${ROOT_DIR}/tmp/matplotlib"
export XDG_CACHE_HOME="${ROOT_DIR}/tmp/cache"

if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [ -x /opt/homebrew/Caskroom/miniconda/base/envs/mathmodel-oil/bin/python3 ]; then
  PYTHON_BIN="/opt/homebrew/Caskroom/miniconda/base/envs/mathmodel-oil/bin/python3"
else
  PYTHON_BIN="python3"
fi

run_python_module() {
  "${PYTHON_BIN}" -m "$@"
}

require_artifact() {
  local artifact="$1"
  local hint="$2"
  if [ ! -s "${artifact}" ]; then
    echo "Required artifact is missing or empty: ${artifact}" >&2
    echo "${hint}" >&2
    exit 1
  fi
}

"${PYTHON_BIN}" -m src.pipeline.clean_data

if [ "${FULL_REBUILD:-0}" = "1" ]; then
  echo "FULL_REBUILD=1: rebuilding baseline, short-term dynamic model, and calibration outputs."
  run_python_module src.models.baseline_supply_demand
  run_python_module src.models.dynamic_short_term
  run_python_module src.calibration.calibrate_dynamic_model
else
  require_artifact "output/calibration/动态模型校准后路径.csv" \
    "Run FULL_REBUILD=1 scripts/build/build_final_paper.sh, or run python3 -m src.calibration.calibrate_dynamic_model first."
  require_artifact "output/calibration/动态模型最优参数.csv" \
    "Run FULL_REBUILD=1 scripts/build/build_final_paper.sh, or run python3 -m src.calibration.calibrate_dynamic_model first."
  require_artifact "output/calibration/动态模型分段误差.csv" \
    "Run FULL_REBUILD=1 scripts/build/build_final_paper.sh, or run python3 -m src.calibration.calibrate_dynamic_model first."
  require_artifact "output/calibration/动态模型候选参数前10.csv" \
    "Run FULL_REBUILD=1 scripts/build/build_final_paper.sh, or run python3 -m src.calibration.calibrate_dynamic_model first."
  require_artifact "figures/baseline_vs_actual.png" \
    "Run FULL_REBUILD=1 scripts/build/build_final_paper.sh, or run python3 -m src.models.baseline_supply_demand first."
fi

"${PYTHON_BIN}" -m src.data.fetch_opec_balance
"${PYTHON_BIN}" -m src.data.fetch_futures_term_structure
"${PYTHON_BIN}" -m src.analysis.market_risk_variables
"${PYTHON_BIN}" -m src.scenarios.scenario_forecast
"${PYTHON_BIN}" -m src.visualization.final_paper_figures
"${PYTHON_BIN}" -m src.visualization.short_term_paper_figures
"${PYTHON_BIN}" -m src.analysis.sensitivity_analysis
"${PYTHON_BIN}" -m src.analysis.calibration_weight_sensitivity
"${PYTHON_BIN}" -m src.analysis.historical_volatility_calibration
"${PYTHON_BIN}" -m src.analysis.monte_carlo_scenario_tree
"${PYTHON_BIN}" -m src.analysis.long_term_state_transition
"${PYTHON_BIN}" -m src.analysis.lagged_risk_index_audit
"${PYTHON_BIN}" -m src.analysis.historical_robustness_audit
"${PYTHON_BIN}" -m src.analysis.statistical_audit
"${PYTHON_BIN}" -m src.analysis.short_term_residual_correction
"${PYTHON_BIN}" -m src.analysis.short_term_model_quality_audit
"${PYTHON_BIN}" -m src.analysis.short_term_model_quality
"${PYTHON_BIN}" -m src.analysis.short_term_ml_features
"${PYTHON_BIN}" -m src.analysis.short_term_ml_residual_model
"${PYTHON_BIN}" -m src.analysis.short_term_rolling_validation
"${PYTHON_BIN}" -m src.analysis.short_term_final_quality_gate
"${PYTHON_BIN}" -m src.experiments.run_model_suite
"${PYTHON_BIN}" -m src.visualization.advanced_candidate_figures

if command -v Rscript >/dev/null 2>&1; then
  Rscript scripts/audit/r_econometric_audit.R || echo "R econometric audit failed; continuing Python-led paper build." >&2
  Rscript scripts/audit/r_academic_figures.R || echo "R academic figures failed; continuing Python-led paper build." >&2
else
  echo "Rscript not found; skipped optional R econometric audit and academic figures." >&2
fi

mkdir -p paper/figures
if [ "${RENDER_MERMAID:-0}" = "1" ] && command -v mmdc >/dev/null 2>&1; then
  mmdc -p paper/diagrams/puppeteer-config.json -i paper/diagrams/综合机制递推模型结构图.mmd -o paper/figures/综合机制递推模型结构图.png -b white -s 2 \
    || echo "Mermaid mechanism diagram render failed; using existing rendered figure." >&2
  mmdc -p paper/diagrams/puppeteer-config.json -i paper/diagrams/评审质疑防御框架图.mmd -o paper/figures/评审质疑防御框架图.png -b white -s 2 \
    || echo "Mermaid defense diagram render failed; using existing rendered figure." >&2
elif [ "${RENDER_MERMAID:-0}" = "1" ]; then
  echo "mmdc not found; using existing Mermaid-rendered figures if present." >&2
else
  echo "Skipped Mermaid rendering; set RENDER_MERMAID=1 to refresh diagrams." >&2
fi
cp figures/price_trend.png paper/figures/布伦特原油长期价格走势.png
cp figures/event_window_price.png paper/figures/冲突窗口价格走势.png
cp figures/return_volatility.png paper/figures/布伦特收益率与波动率.png
cp figures/baseline_vs_actual.png paper/figures/传统供需基准与真实价格对比.png
cp figures/monte_carlo_price_fan.png paper/figures/蒙特卡洛价格扇形区间.png
cp figures/蒙特卡洛情景树高级组合图.png paper/figures/蒙特卡洛情景树高级组合图.png
cp figures/传统蒙特卡洛路径云图.png paper/figures/传统蒙特卡洛路径云图.png
cp figures/long_term_state_transition_fan.png paper/figures/长期状态转移情景树.png
cp figures/inventory_depletion_risk.png paper/figures/库存与供需缺口风险.png
cp figures/sensitivity_tornado_180day.png paper/figures/参数敏感性龙卷风图.png
cp figures/monte_carlo_tail_risk.png paper/figures/蒙特卡洛尾部风险概率.png
cp figures/gpr_event_window_audit.png paper/figures/地缘风险指数滞后审计.png
cp figures/lagged_gpr_brent_return.png paper/figures/滞后GPR与布伦特收益关系.png
cp figures/OVX隐含波动率滞后检验.png paper/figures/OVX隐含波动率滞后检验.png
cp figures/historical_event_extremeness.png paper/figures/历史窗口极端性检验.png
cp figures/历史波动参数校准.png paper/figures/历史波动参数校准.png
cp figures/R短期误差学术诊断.png paper/figures/R短期误差学术诊断.png
cp figures/R长期状态转移扇形图.png paper/figures/R长期状态转移扇形图.png
cp figures/R历史基准误差分布图.png paper/figures/R历史基准误差分布图.png
cp figures/短期模型统计审计.png paper/figures/短期模型统计审计.png
cp output/candidate_figures/Python长期敏感性热力候选图.png paper/figures/长期敏感性扰动热力图.png
cp output/candidate_figures/Python候选模型权衡气泡候选图.png paper/figures/候选模型权衡气泡图.png

"${PYTHON_BIN}" -m src.analysis.paper_visual_audit

if ! command -v xelatex >/dev/null 2>&1; then
  if [ -x /Library/TeX/texbin/xelatex ]; then
    export PATH="/Library/TeX/texbin:${PATH}"
  else
    echo "xelatex not found. Install MacTeX or BasicTeX first." >&2
    exit 1
  fi
fi

xelatex -interaction=nonstopmode -halt-on-error -output-directory="${BUILD_DIR}" paper/总论文.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory="${BUILD_DIR}" paper/总论文.tex

cp "${BUILD_DIR}/总论文.pdf" "${FINAL_PDF}"
echo "Built ${FINAL_PDF}"

if command -v pandoc >/dev/null 2>&1; then
  pandoc \
    --from=latex \
    --to=docx \
    --toc \
    --toc-depth=2 \
    --resource-path=".:paper/figures:figures" \
    --output="${FINAL_DOCX}" \
    paper/总论文.tex
  "${PYTHON_BIN}" scripts/postprocess/fix_docx_toc_title.py "${FINAL_DOCX}"
  echo "Built ${FINAL_DOCX}"
else
  echo "pandoc not found; skipped DOCX export." >&2
fi
