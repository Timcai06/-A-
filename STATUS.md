# 当前状态

更新时间：2026-05-09

## 当前阶段

阶段 2：传统供需基准模型已完成。

```mermaid
flowchart LR
    S0["阶段0-0.6<br/>项目骨架/环境/数据库设计"] --> S1["阶段1<br/>数据清洗与验收"]
    S1 --> S2["阶段2<br/>传统供需基准模型"]
    S2 --> S3["阶段3<br/>短期动态递推模型"]
    S3 --> S4["阶段4<br/>参数校准"]
    S4 --> S5["阶段5-7<br/>预测/敏感性/论文"]

    S0:::done
    S1:::done
    S2:::done
    S3:::next

    classDef done fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef next fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
```

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
- 阶段 2 已实现并运行 `src/models/baseline_supply_demand.py`。
- 阶段 2 已生成传统供需基准结果、对照图和阶段报告。
- 阶段 2 结论：在线性化短期弹性口径下，传统供需模型预测约 278-337 USD/barrel，明显高于附件 CSV 的冲突窗口最高收盘价 114.06 USD/barrel。

## 当前可直接引用的成果

| 类型 | 文件 | 用途 |
|---|---|---|
| 清洗数据 | `data/processed/布伦特原油期货主力合约价格数据_清洗后.csv` | 论文数据来源和后续模型输入 |
| 冲突窗口 | `data/processed/布伦特原油期货主力合约价格数据_冲突窗口.csv` | 阶段 3 和阶段 4 的真实价格对照 |
| 数据验收 | `output/reports/stage1_validation_report.md` | 证明阶段 1 数据可靠 |
| 基准结果 | `output/baseline/传统供需基准模型结果.csv` | 论文模型一结果表 |
| 基准图 | `figures/baseline_vs_actual.png` | 证明传统模型高估现实价格 |
| 阶段报告 | `output/reports/stage2_baseline_model_report.md` | 可直接改写进论文模型一小节 |

## 下一步

进入阶段 3：短期动态递推模型。

阶段 3 的目标是把库存、战略储备、绕道运输、需求收缩和恐慌衰减放进日度递推模型，解释现实油价为什么没有按静态供需模型一路冲到 200 美元以上，并为后续参数校准和 60-180 天情景预测提供主模型。

阶段 3 最小完成标准：

- 输出 0-60 天日度模拟价格路径。
- 每天记录有效供给、有效需求、库存缓冲、SPR 释放、绕道运输和恐慌因子。
- 至少生成一张“动态模型 vs 实际价格”的对比图。
- 给出 RMSE、峰值误差和末日价格误差。

## 阻塞点

PostgreSQL 当前只设计，不建库、不写入。阶段 2 已经证明当前 CSV 文件足够支撑前两阶段，数据库可以等进入多情景、多轮参数实验后再启用。

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

## 阶段 2 验收目标

- `src/models/baseline_supply_demand.py`：已完成
- `output/baseline/传统供需基准模型结果.csv`：已完成
- `figures/baseline_vs_actual.png`：已完成
- `output/reports/stage2_baseline_model_report.md`：已完成
- 传统模型高估现实价格的论文论证：已完成
