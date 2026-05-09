# scripts

这里放置项目级命令脚本。当前阶段只建立目录，不放置会写数据库或生成结果的脚本。

已有脚本：

- `project_env.sh`：运行建模脚本前 source，用于设置 Matplotlib/fontconfig 的项目本地缓存目录。

后续可能添加：

- `load_raw_to_db.py`：将清洗后的原始价格数据写入 PostgreSQL。
- `export_results.py`：从 PostgreSQL 导出论文用 CSV。
- `run_stage1.sh`：阶段 1 一键运行入口。
