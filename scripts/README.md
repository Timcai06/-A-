# scripts 目录说明

这里放置项目级命令脚本。脚本职责是“复现、构建、轻量后处理”，不承载核心模型逻辑。

## 已有脚本

- `project_env.sh`：运行建模脚本前 source，用于设置 Matplotlib/fontconfig 的项目本地缓存目录。
- `build_final_paper.sh`：最终论文 PDF/DOCX 构建入口，会生成论文图、蒙特卡洛结果和最终交付文件。
- `build_short_term_paper.sh`：历史短期模型素材稿构建入口，当前不是最终交付主线。
- `fix_docx_toc_title.py`：DOCX 后处理脚本，将 Pandoc 默认英文目录标题改为中文“目录”。

## 使用方式

```bash
source scripts/project_env.sh
./scripts/build_final_paper.sh
```

数据库相关脚本当前不是主线；PostgreSQL 方案保留在 `sql/` 和文档中，用于后续扩展。
