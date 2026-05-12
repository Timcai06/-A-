# 阶段 0.6 环境与数据库准备报告

## 完成时间

2026-05-09

## 完成内容

- 新增 `environment.yml`，推荐使用 `mamba` 创建 `mathmodel-oil` 环境。
- 更新 `requirements.txt`，补充 PostgreSQL 连接相关依赖。
- 新增 `config/database.yml`，记录数据库名、连接方式、schema 边界和使用策略。
- 新增 `sql/001_init_schema.sql`，设计 PostgreSQL schema 和核心表。
- 新增 `sql/002_load_raw_brent_placeholder.sql`，作为后续导入参考，不直接执行。
- 新增 `docs/09_环境与数据库方案.md`，说明为什么建环境、是否使用 PostgreSQL、何时启用数据库。

## 当前决策

- mamba/conda 环境需要先建，避免系统 Python 版本过新带来的兼容风险。
- PostgreSQL 可以使用，但当前只作为设计方案，不立即建库、不写入数据。
- 真正启用数据库应放在阶段 1 数据清洗完成之后。

## 验收结论

阶段 0.6 文件准备已完成。下一步可以创建环境，随后进入阶段 1。
