# paper 目录说明

本目录保存论文源码、参考文献和论文专用图片。

## 主线文件

| 文件 | 作用 |
|---|---|
| `总论文.tex` | 当前最终论文 LaTeX 主控源码 |
| `sections/` | 当前最终论文分层源码：每个正文章节一个文件夹，每个小节一个独立 `.tex` 文件 |
| `参考文献.bib` | BibTeX 参考文献 |
| `参考文献与证据库.md` | 文献、数据来源和论文证据整理 |
| `figures_mapping.md` | 论文图表与数据来源映射 |
| `figures/` | 最终论文使用的高质量图片 |

## 历史素材

`短期冲击模型历史素材稿.tex` 是早期短期模型论文素材稿。它保留用于追溯短期模型写作来源，但最终交付以 `总论文.tex`、`sections/` 和 `output/final/A题_霍尔木兹海峡封锁对国际原油价格影响_论文.pdf`、`output/final/A题_霍尔木兹海峡封锁对国际原油价格影响_论文.docx` 为准。

## 章节源码结构

`sections/` 采用“章节文件夹 + 小节文件”的结构。每个章节文件夹都有一个 `index.tex`，只负责声明本章标题或汇总 `\input`；具体内容放在同目录下的小节文件中。正文主控文件 `总论文.tex` 只引用各章节的 `index.tex`。

例如：

```text
sections/
  03_模型假设/
    index.tex
    01_基本假设.tex
    02_参数来源与识别边界.tex
  04_符号说明/
    index.tex
    01_符号说明.tex
  05_模型建立与求解/
    index.tex
    01_数据来源与清洗口径.tex
    02_外部供需与风险约束.tex
    03_波动率特征.tex
    04_传统供需基准模型.tex
    05_短期冲击模型.tex
    06_参数求解与核心结果.tex
    07_中长期油价调节模型.tex
    08_敏感性分析与尾部风险.tex
```

后续修改论文时，优先修改对应小节文件，不在 `总论文.tex` 中直接写正文。

## 构建方式

从项目根目录运行：

```bash
source scripts/project_env.sh
./scripts/build/build_final_paper.sh
```

根目录的 `scripts/build_final_paper.sh` 是兼容转发入口。正式构建脚本位于 `scripts/build/build_final_paper.sh`，会用 `xelatex -shell-escape` 编译总论文，并通过 `minted`/Pygments 渲染附录代码高亮。
