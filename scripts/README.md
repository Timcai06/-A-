# scripts 目录说明

这里放置项目级命令脚本。脚本职责是“复现、构建、统计审计、轻量后处理”，不承载核心模型逻辑。

## 目录分类

| 目录 | 用途 | 代表脚本 |
|---|---|---|
| `build/` | 论文和交付物构建 | `build_final_paper.sh`、`build_short_term_paper.sh` |
| `env/` | 本地运行环境初始化 | `project_env.sh` |
| `audit/` | R 统计和计量审计 | `statistical_audit.R`、`r_econometric_audit.R` |
| `postprocess/` | 文档轻量后处理 | `fix_docx_toc_title.py` |

根目录保留兼容入口：

- `project_env.sh`：转发到 `env/project_env.sh`，用于 `source scripts/project_env.sh`。
- `build_final_paper.sh`：转发到 `build/build_final_paper.sh`。
- `build_short_term_paper.sh`：转发到 `build/build_short_term_paper.sh`，当前不是最终交付主线。

## 使用方式

```bash
source scripts/project_env.sh
./scripts/build/build_final_paper.sh
```

根目录的 `build_final_paper.sh` 是兼容转发入口。正式脚本会重新生成或检查清洗数据、OPEC 约束、期货期限结构模板、OVX/GPR 风险变量、长期三情景、敏感性、蒙特卡洛、短期质量门、论文图表和附录代码高亮，然后输出最终 PDF/DOCX。

常用开关：

```bash
FULL_REBUILD=1 ./scripts/build/build_final_paper.sh
RENDER_MERMAID=1 ./scripts/build/build_final_paper.sh
PYTHON_BIN=/path/to/python3 ./scripts/build/build_final_paper.sh
```

其中 `FULL_REBUILD=1` 会从基准模型、短期递推和参数校准开始重算；`RENDER_MERMAID=1` 会刷新 Mermaid 结构图；默认构建使用 `xelatex -shell-escape`，以便 `minted`/Pygments 渲染附录代码高亮。

数据库相关脚本当前不是主线；PostgreSQL 方案保留在 `sql/` 和文档中，用于后续扩展。

当前机器已经接通 `Rscript`。论文主线统计审计由 `src.analysis.statistical_audit` 的 Python 版本完成；R 版用于独立复核，运行方式为：

```bash
Rscript scripts/audit/statistical_audit.R
Rscript scripts/audit/r_econometric_audit.R
```
