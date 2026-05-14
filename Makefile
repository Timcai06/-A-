.PHONY: pdf paper full-paper

TEX := /Library/TeX/texbin/xelatex
BUILD_DIR := output/build/latex
FINAL_DIR := output/final
FINAL_PDF := $(FINAL_DIR)/A题_霍尔木兹海峡封锁对国际原油价格影响_论文.pdf

pdf:
	mkdir -p $(BUILD_DIR) $(FINAL_DIR)
	$(TEX) -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD_DIR) paper/总论文.tex
	$(TEX) -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD_DIR) paper/总论文.tex
	cp $(BUILD_DIR)/总论文.pdf $(FINAL_PDF)
	@echo "Built $(FINAL_PDF)"

paper:
	scripts/build/build_final_paper.sh

full-paper:
	FULL_REBUILD=1 scripts/build/build_final_paper.sh
