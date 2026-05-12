#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BUILD_DIR="output/build/latex"
FINAL_DIR="output/final"

mkdir -p "${BUILD_DIR}" "${FINAL_DIR}"
mkdir -p "tmp/matplotlib" "tmp/cache"
export MPLCONFIGDIR="${ROOT_DIR}/tmp/matplotlib"
export XDG_CACHE_HOME="${ROOT_DIR}/tmp/cache"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x /opt/homebrew/Caskroom/miniconda/base/envs/mathmodel-oil/bin/python3 ]; then
  PYTHON_BIN="/opt/homebrew/Caskroom/miniconda/base/envs/mathmodel-oil/bin/python3"
fi

"${PYTHON_BIN}" -m src.pipeline.clean_data
"${PYTHON_BIN}" -m src.visualization.final_paper_figures
"${PYTHON_BIN}" -m src.visualization.short_term_paper_figures
"${PYTHON_BIN}" -m src.analysis.sensitivity_analysis
"${PYTHON_BIN}" -m src.analysis.calibration_weight_sensitivity
"${PYTHON_BIN}" -m src.analysis.monte_carlo_scenario_tree
"${PYTHON_BIN}" -m src.analysis.long_term_state_transition
"${PYTHON_BIN}" -m src.analysis.lagged_risk_index_audit
"${PYTHON_BIN}" -m src.analysis.historical_robustness_audit
"${PYTHON_BIN}" -m src.analysis.statistical_audit

if command -v Rscript >/dev/null 2>&1; then
  Rscript scripts/audit/r_econometric_audit.R || echo "R econometric audit failed; continuing Python-led paper build." >&2
  Rscript scripts/audit/r_academic_figures.R || echo "R academic figures failed; continuing Python-led paper build." >&2
else
  echo "Rscript not found; skipped optional R econometric audit and academic figures." >&2
fi

mkdir -p paper/figures
cp figures/price_trend.png paper/figures/布伦特原油长期价格走势.png
cp figures/event_window_price.png paper/figures/冲突窗口价格走势.png
cp figures/return_volatility.png paper/figures/布伦特收益率与波动率.png
cp figures/baseline_vs_actual.png paper/figures/传统供需基准与真实价格对比.png
cp figures/monte_carlo_price_fan.png paper/figures/蒙特卡洛价格扇形区间.png
cp figures/long_term_state_transition_fan.png paper/figures/长期状态转移情景树.png
cp figures/inventory_depletion_risk.png paper/figures/库存与供需缺口风险.png
cp figures/sensitivity_tornado_180day.png paper/figures/参数敏感性龙卷风图.png
cp figures/monte_carlo_tail_risk.png paper/figures/蒙特卡洛尾部风险概率.png
cp figures/gpr_event_window_audit.png paper/figures/地缘风险指数滞后审计.png
cp figures/lagged_gpr_brent_return.png paper/figures/滞后GPR与布伦特收益关系.png
cp figures/historical_event_extremeness.png paper/figures/历史窗口极端性检验.png
cp figures/R短期误差学术诊断.png paper/figures/R短期误差学术诊断.png
cp figures/R长期状态转移扇形图.png paper/figures/R长期状态转移扇形图.png
cp figures/R历史基准误差分布图.png paper/figures/R历史基准误差分布图.png
cp figures/短期模型统计审计.png paper/figures/短期模型统计审计.png

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

cp "${BUILD_DIR}/总论文.pdf" "${FINAL_DIR}/总论文.pdf"
echo "Built ${FINAL_DIR}/总论文.pdf"

if command -v pandoc >/dev/null 2>&1; then
  pandoc \
    --from=latex \
    --to=docx \
    --toc \
    --toc-depth=2 \
    --resource-path=".:paper/figures:figures" \
    --output="${FINAL_DIR}/总论文.docx" \
    paper/总论文.tex
  "${PYTHON_BIN}" scripts/postprocess/fix_docx_toc_title.py "${FINAL_DIR}/总论文.docx"
  echo "Built ${FINAL_DIR}/总论文.docx"
else
  echo "pandoc not found; skipped DOCX export." >&2
fi
