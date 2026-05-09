# 当前状态

更新时间：2026-05-09

## 当前阶段

阶段 1：数据清洗和探索已完成。

## 当前主线

采用 `Python 动态递推模型 + 参数校准 + 三情景预测 + 敏感性分析 + 论文图表` 作为主路径。

`AI 新闻情绪抓取`、`C++ 多智能体仿真`、`React/ECharts Dashboard` 暂时作为后置增强项，不进入主路径验收。

## 已完成

- 阶段 0 文档目录已建立。
- 顶层目录已建立：`data/`、`src/`、`figures/`、`output/`、`paper/`、`dashboard/`。
- 原始 CSV 已备份到 `data/raw/`。
- 项目规划 PDF 已生成。
- 阶段 0.5 已补强配置层、数据层、源码层和输出层边界。
- 阶段 0.6 已新增 `environment.yml`、`config/database.yml` 和 PostgreSQL 初始化 SQL 设计。
- `mathmodel-oil` 环境已存在，并验证 Python 版本为 3.11.15，核心科学计算和 PostgreSQL 依赖可导入。
- 阶段 1 已实现并运行 `src/pipeline/clean_data.py`。
- 阶段 1 已生成清洗数据、冲突窗口数据、三张基础图和阶段报告。
- 阶段 1 已新增并运行 `src/pipeline/validate_stage1.py --rerun`，自动验收 33 项，全部通过。
- 阶段 1 已新增数据字典、自动验收报告、机器可读校验结果和 manifest。

## 下一步

进入阶段 2：传统供需基准模型。

阶段 2 的目标是建立一个只考虑供应缺口和低需求价格弹性的对照模型，用来证明传统静态供需逻辑会高估油价，从而为阶段 3 引入库存、战略储备、绕道运输和需求收缩等缓冲机制提供论证基础。

## 阻塞点

PostgreSQL 当前只设计，不建库、不写入。阶段 1 已完成后，可以在进入阶段 2 前决定是否先导入清洗后的价格数据，但这不是阶段 2 的必要条件。

备注：本机绘图依赖首次导入时出现 Matplotlib/fontconfig 缓存目录不可写提示，已添加 `.env.example`、`scripts/project_env.sh` 和项目本地 `.cache/` 目录约定。

## 阶段 1 验收目标

- `data/processed/布伦特原油期货主力合约价格数据_清洗后.csv`：已完成
- `data/processed/布伦特原油期货主力合约价格数据_冲突窗口.csv`：已完成
- `figures/price_trend.png`：已完成
- `figures/event_window_price.png`：已完成
- `figures/return_volatility.png`：已完成
- `output/reports/stage1_data_cleaning_report.md`：已完成
- `output/reports/stage1_validation_report.md`：已完成
- `output/reports/stage1_manifest.json`：已完成
- `data/metadata/stage1_data_dictionary.md`：已完成
