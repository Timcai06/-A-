#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BUILD_DIR="output/build/latex"
FINAL_DIR="output/final"

mkdir -p "${BUILD_DIR}" "${FINAL_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x /opt/homebrew/Caskroom/miniconda/base/envs/mathmodel-oil/bin/python3 ]; then
  PYTHON_BIN="/opt/homebrew/Caskroom/miniconda/base/envs/mathmodel-oil/bin/python3"
fi

"${PYTHON_BIN}" -m src.visualization.final_paper_figures

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
    --resource-path=".:paper/figures:figures" \
    --output="${FINAL_DIR}/总论文.docx" \
    paper/总论文.tex
  echo "Built ${FINAL_DIR}/总论文.docx"
else
  echo "pandoc not found; skipped DOCX export." >&2
fi
