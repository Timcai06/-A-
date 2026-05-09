#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p output/pdf

if ! command -v xelatex >/dev/null 2>&1; then
  if [ -x /Library/TeX/texbin/xelatex ]; then
    export PATH="/Library/TeX/texbin:${PATH}"
  else
    echo "xelatex not found. Install MacTeX or BasicTeX first." >&2
    exit 1
  fi
fi

xelatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf paper/短期动态模型论文.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf paper/短期动态模型论文.tex

echo "Built output/pdf/短期动态模型论文.pdf"
