# paper 目录说明

本目录保存论文源码、参考文献和论文专用图片。

## 主线文件

| 文件 | 作用 |
|---|---|
| `总论文.tex` | 当前唯一最终论文 LaTeX 源码 |
| `参考文献.bib` | BibTeX 参考文献 |
| `参考文献与证据库.md` | 文献、数据来源和论文证据整理 |
| `figures_mapping.md` | 论文图表与数据来源映射 |
| `figures/` | 最终论文使用的高质量图片 |

## 历史素材

`短期动态模型论文.tex` 是早期短期模型论文素材稿。它保留用于追溯短期模型写作来源，但最终交付以 `总论文.tex` 和 `output/final/总论文.pdf`、`output/final/总论文.docx` 为准。

## 构建方式

从项目根目录运行：

```bash
source scripts/project_env.sh
./scripts/build_final_paper.sh
```
