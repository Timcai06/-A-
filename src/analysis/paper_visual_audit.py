"""Audit paper figures, tables, captions, and reference usefulness."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parents


PAPER_DIR = PROJECT_ROOT / "paper"
SECTION_DIR = PAPER_DIR / "sections"
FIGURE_DIR = PAPER_DIR / "figures"
FIGURE_CSV = PROJECT_ROOT / "output" / "quality" / "论文图表清单.csv"
TABLE_CSV = PROJECT_ROOT / "output" / "quality" / "论文表格清单.csv"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "论文图表功能审计报告.md"


@dataclass(frozen=True)
class TexSource:
    path: Path
    text: str

    @property
    def section_name(self) -> str:
        return self.path.stem


def load_sources() -> list[TexSource]:
    paths = [PAPER_DIR / "总论文.tex", *sorted(SECTION_DIR.glob("*.tex"))]
    return [TexSource(path, path.read_text(encoding="utf-8")) for path in paths if path.exists()]


def clean_tex(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    value = value.replace(r"\%", "%").replace("--", "-")
    return value


def label_refs(all_text: str) -> dict[str, int]:
    refs: dict[str, int] = {}
    for match in re.finditer(r"\\(?:ref|pageref|autoref)\{([^}]+)\}", all_text):
        refs[match.group(1)] = refs.get(match.group(1), 0) + 1
    return refs


def caption_quality(caption: str) -> tuple[str, str]:
    signal_words = ["用于", "说明", "证明", "显示", "解释", "刻画", "检验", "回应", "表明"]
    has_signal = any(word in caption for word in signal_words)
    if len(caption) >= 34 and has_signal:
        return "主旨明确", "图注同时说明图的内容和论文作用"
    if len(caption) >= 24:
        return "基本明确", "图注说明了图的内容，但作用表达可继续增强"
    return "偏短", "图注偏描述性，建议补一句该图证明什么"


def extract_figures(sources: list[TexSource], refs: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source in sources:
        for match in re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", source.text, flags=re.S):
            env = match.group(0)
            image_match = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", env)
            caption_match = re.search(r"\\caption\{(.+?)\}", env, flags=re.S)
            label_match = re.search(r"\\label\{([^}]+)\}", env)
            image = image_match.group(1) if image_match else ""
            caption = clean_tex(caption_match.group(1)) if caption_match else ""
            label = label_match.group(1) if label_match else ""
            quality, note = caption_quality(caption)
            image_path = FIGURE_DIR / image if image else None
            rows.append(
                {
                    "章节": source.section_name,
                    "图片文件": image,
                    "文件存在": bool(image_path and image_path.exists()),
                    "图注": caption,
                    "标签": label or "无",
                    "正文引用次数": refs.get(label, 0) if label else 0,
                    "图注质量": quality,
                    "审计说明": note,
                }
            )
    return pd.DataFrame(rows)


def extract_tables(sources: list[TexSource], refs: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    table_pattern = r"\\begin\{(?:table|longtable)\}.*?\\end\{(?:table|longtable)\}"
    for source in sources:
        for match in re.finditer(table_pattern, source.text, flags=re.S):
            env = match.group(0)
            caption_match = re.search(r"\\caption\*?\{(.+?)\}", env, flags=re.S)
            label_match = re.search(r"\\label\{([^}]+)\}", env)
            caption = clean_tex(caption_match.group(1)) if caption_match else ""
            label = label_match.group(1) if label_match else ""
            quality, note = caption_quality(caption)
            rows.append(
                {
                    "章节": source.section_name,
                    "表注": caption,
                    "标签": label or "无",
                    "正文引用次数": refs.get(label, 0) if label else 0,
                    "表注质量": quality,
                    "审计说明": note,
                }
            )
    return pd.DataFrame(rows)


def build_report(figures: pd.DataFrame, tables: pd.DataFrame) -> str:
    missing_figures = figures[~figures["文件存在"]]
    weak_captions = figures[figures["图注质量"] == "偏短"]
    unlabeled_figures = figures[figures["标签"] == "无"]
    unlabeled_tables = tables[tables["标签"] == "无"]
    referenced_figures = int((figures["正文引用次数"] > 0).sum())
    referenced_tables = int((tables["正文引用次数"] > 0).sum())

    main_figures = [
        "短期模型拟合效果.png",
        "蒙特卡洛情景树高级组合图.png",
        "长期状态转移情景树.png",
        "参数敏感性龙卷风图.png",
        "短期模型参数剖面图.png",
    ]
    main_status = []
    for figure in main_figures:
        row = figures[figures["图片文件"] == figure]
        if row.empty:
            main_status.append(f"| {figure} | 未在正文出现 | - |")
        else:
            item = row.iloc[0]
            main_status.append(f"| {figure} | {item['章节']} | {item['图注质量']} |")

    return f"""# 论文图表功能审计报告

## 总览

本报告检查总论文中的图、表、图注和引用状态，用于保证图表不是装饰，而是服务于模型证据链。

| 项目 | 数量 |
|---|---:|
| 正文图片数 | {len(figures)} |
| 正文表格数 | {len(tables)} |
| 已被正文交叉引用的图片 | {referenced_figures} |
| 已被正文交叉引用的表格 | {referenced_tables} |
| 缺失图片文件 | {len(missing_figures)} |
| 偏短图片图注 | {len(weak_captions)} |
| 无标签图片 | {len(unlabeled_figures)} |
| 无标签表格 | {len(unlabeled_tables)} |

## 主图状态

| 主图 | 所在章节 | 图注质量 |
|---|---|---|
{chr(10).join(main_status)}

## 审计解释

- 不是所有图片都必须在正文中用 `\\ref{{}}` 交叉引用；连续叙述中紧邻出现的图，也可以由段落和图注共同承担解释功能。
- 关键主图应满足两个条件：图注说明“图里有什么”，同时说明“该图证明什么”。
- 若后续继续精修，优先处理偏短图注和无标签的核心图表，而不是机械给所有辅助图都增加长引用。

## 输出

- `{FIGURE_CSV.relative_to(PROJECT_ROOT)}`
- `{TABLE_CSV.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    ensure_parents([FIGURE_CSV, TABLE_CSV, REPORT_PATH])
    sources = load_sources()
    all_text = "\n".join(source.text for source in sources)
    refs = label_refs(all_text)
    figures = extract_figures(sources, refs)
    tables = extract_tables(sources, refs)
    figures.to_csv(FIGURE_CSV, index=False)
    tables.to_csv(TABLE_CSV, index=False)
    REPORT_PATH.write_text(build_report(figures, tables), encoding="utf-8")
    print("Paper visual audit complete")
    print(f"Figures: {FIGURE_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Tables: {TABLE_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
