# 当前状态

更新时间：2026-05-09

## 当前阶段

阶段 4：参数校准已完成。

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
    S3:::done
    S4:::done
    S5:::next

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
- 阶段 3 已实现并运行 `src/models/dynamic_short_term.py`。
- 阶段 3 已生成短期动态递推结果、误差指标、拟合对照图和阶段报告。
- 阶段 3 结论：在 SPR、库存、绕道运输、需求收缩和恐慌衰减共同作用下，模型能把价格解释到 110 美元附近平台。
- 阶段 4 已实现并运行 `src/calibration/calibrate_dynamic_model.py`。
- 阶段 4 已生成校准后路径、最优参数、候选参数前 10、分段误差和阶段报告。
- 阶段 4 结论：综合最优参数 RMSE 为 3.50，并通过固定种子随机搜索加连续局部精修，将前期、中期、后期、高价平台和低价回落误差同步压低，短期模型达到优秀水平。
- 阶段 4.6 已建立论文参考文献与证据库，记录油价冲击、库存预期、地缘风险、需求弹性、SPR 和霍尔木兹背景资料。

## 当前可直接引用的成果

| 类型 | 文件 | 用途 |
|---|---|---|
| 清洗数据 | `data/processed/布伦特原油期货主力合约价格数据_清洗后.csv` | 论文数据来源和后续模型输入 |
| 冲突窗口 | `data/processed/布伦特原油期货主力合约价格数据_冲突窗口.csv` | 阶段 3 和阶段 4 的真实价格对照 |
| 数据验收 | `output/reports/stage1_validation_report.md` | 证明阶段 1 数据可靠 |
| 基准结果 | `output/baseline/传统供需基准模型结果.csv` | 论文模型一结果表 |
| 基准图 | `figures/baseline_vs_actual.png` | 证明传统模型高估现实价格 |
| 动态结果 | `output/calibration/短期动态递推模型结果.csv` | 论文模型二结果表 |
| 动态图 | `figures/fitted_vs_actual.png` | 证明动态模型能解释价格平台 |
| 阶段报告 | `output/reports/stage3_dynamic_model_report.md` | 可直接改写进论文模型二小节 |
| 校准参数 | `output/calibration/动态模型最优参数.csv` | 阶段 5 中性预测基准 |
| 校准报告 | `output/reports/stage4_calibration_report.md` | 可直接改写进论文参数校准小节 |
| 质量检查 | `output/reports/stage0_to_stage3_quality_audit_report.md` | 阶段 0 至阶段 3 完整质量审计 |
| 文献证据库 | `paper/参考文献与证据库.md` | 后续论文文献综述、模型依据和参考文献来源 |

## 下一步

进入阶段 5：60-180 天情景预测。

阶段 5 的目标是以阶段 4 的综合最优参数作为中性基准，构造乐观、中性、悲观三种封锁持续情景，输出 60-180 天价格路径、平衡价格区间和库存风险。

阶段 5 最小完成标准：

- 输出三情景价格路径。
- 输出三情景参数表。
- 生成三情景价格对比图。
- 给出平衡价格区间和库存耗尽风险说明。

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

## 阶段 3 验收目标

- `src/models/dynamic_short_term.py`：已完成
- `output/calibration/短期动态递推模型结果.csv`：已完成
- `output/calibration/短期动态递推模型误差指标.csv`：已完成
- `figures/fitted_vs_actual.png`：已完成
- `output/reports/stage3_dynamic_model_report.md`：已完成

## 阶段 4 验收目标

- `src/calibration/calibrate_dynamic_model.py`：已完成
- `output/calibration/动态模型校准后路径.csv`：已完成
- `output/calibration/动态模型最优参数.csv`：已完成
- `output/calibration/动态模型候选参数前10.csv`：已完成
- `output/calibration/动态模型分段误差.csv`：已完成
- `output/reports/stage4_calibration_report.md`：已完成
