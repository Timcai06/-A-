#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p output/pdf output/doc

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

if command -v pandoc >/dev/null 2>&1; then
  pandoc \
    --from=latex \
    --to=docx \
    --resource-path=".:paper/figures:figures" \
    --output=output/doc/短期动态模型论文.docx \
    paper/短期动态模型论文.tex
  echo "Built output/doc/短期动态模型论文.docx"
else
  echo "pandoc not found; skipped DOCX export." >&2
fi
