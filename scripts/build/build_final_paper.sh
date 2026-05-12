#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BUILD_DIR="output/build/latex"
FINAL_DIR="output/final"

mkdir -p "${BUILD_DIR}" "${FINAL_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x /opt/homebrew/Caskroom/miniconda/base/envs/mathmodel-oil/bin/python3 ]; then
  PYTHON_BIN="/opt/homebrew/Caskroom/miniconda/base/envs/mathmodel-oil/bin/python3"
fi

"${PYTHON_BIN}" -m src.visualization.final_paper_figures
"${PYTHON_BIN}" -m src.visualization.short_term_paper_figures
"${PYTHON_BIN}" -m src.analysis.monte_carlo_scenario_tree
"${PYTHON_BIN}" -m src.analysis.lagged_risk_index_audit
"${PYTHON_BIN}" -m src.analysis.historical_robustness_audit
"${PYTHON_BIN}" -m src.analysis.statistical_audit

if command -v Rscript >/dev/null 2>&1; then
  Rscript scripts/audit/r_econometric_audit.R || echo "R econometric audit failed; continuing Python-led paper build." >&2
else
  echo "Rscript not found; skipped optional R econometric audit." >&2
fi

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
