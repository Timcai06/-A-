# 当前状态

更新时间：2026-05-10

## 当前阶段

阶段 5：60-180 天三情景预测已完成。短期模型评委质疑防御性检验已补强完成。下一步进入阶段 6 敏感性分析。

```mermaid
flowchart LR
    S0["阶段0-0.6<br/>项目骨架/环境/数据库设计"] --> S1["阶段1<br/>数据清洗与验收"]
    S1 --> S2["阶段2<br/>传统供需基准模型"]
    S2 --> S3["阶段3<br/>短期动态递推模型"]
    S3 --> S4["阶段4<br/>参数校准"]
    S4 --> P0["短期模型论文初稿<br/>LaTeX/PDF/DOCX"]
    P0 --> D0["短期模型防御检验<br/>基准/滞后/拐点"]
    D0 --> S5["阶段5<br/>三情景预测"]
    S5 --> S6["阶段6<br/>敏感性分析"]
    S6 --> S7["阶段7<br/>总论文"]

    S0:::done
    S1:::done
    S2:::done
    S3:::done
    S4:::done
    P0:::done
    D0:::done
    S5:::done
    S6:::next
    S7:::next

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
- 阶段 4 结论：综合最优参数 RMSE 为 3.47，MAE 为 2.89，并通过固定种子随机搜索、连续局部精修和局部稳健性复核，将前期、中期、后期、高价平台和低价回落误差同步压低，短期模型达到优秀水平。
- 阶段 4.7 已新增质量增强诊断，证明短期模型相对上一日价格朴素基准 RMSE 改善 22.4%，Theil U 为 0.776，并在 800 组局部参数扰动下保持稳定。
- 阶段 4.9 已新增评委质疑防御性检验：短期模型 RMSE 3.47，优于朴素上一日基准 4.47、滚动 ARIMA(1,1,0) 基准 4.53；MAPE 为 2.90%；模型曲线原始位置 RMSE 最低，向左平移 1 天后 RMSE 升至 3.95，排除了明显滞后复制风险；主要突变日中 8/10 同步捕捉、1/10 提前 1 日、1/10 未捕捉。
- 阶段 4.10 已新增致命质疑补充防御：承认 46 个交易日对应 21 个连续校准参数存在过拟合风险，并通过题面约束、多目标校准、\(\pm 15\%\) 压力测试、传统基准弹性敏感性和代码硬编码审计进行防御。压力测试中 96.9% 的样本峰值仍落在 105-125 美元/桶，代码审计确认模型未把 120 美元硬编码为上限。
- 已对原始 CSV 的其他历史高波动窗口做固定参数边界检验：霍尔木兹机制参数在非霍尔木兹窗口通常弱于朴素基准，说明模型是事件机制模型，不应包装成通用油价短线交易器。
- 阶段 4.8 已新增参数来源与可信度说明，明确当前短期模型没有使用爬虫数值数据，校准参数必须写成模型待估参数而非真实观测数据。
- 阶段 4.6 已建立论文参考文献与证据库，记录油价冲击、库存预期、地缘风险、需求弹性、SPR 和霍尔木兹背景资料。
- 短期动态模型论文初稿已生成，采用 `ctexart` + `xelatex` 编译，包含公式、参数表、误差评价、机制贡献图、候选模型对比图和参考文献。
- 本机已安装 BasicTeX，并补齐 `ctex`、中文字体和常用排版包；论文 PDF 与 DOCX 可用 `./scripts/build_short_term_paper.sh` 一键复现。
- 短期模型 PDF 已渲染检查为 13 页，DOCX 已通过 LibreOffice 转 PDF 后抽样检查，适合作为后续人工润色和队友批注版本。
- 阶段 5 已实现并运行 `src/scenarios/forecast_stage5.py`，基于阶段 4 综合最优参数构造乐观、中性、悲观三条 60-180 天路径。
- 阶段 5 结论：中性情景第 180 天约 104.27 USD/barrel；悲观情景第 180 天约 119.26 USD/barrel，且存在高二次跳涨风险。

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
| 短期论文源码 | `paper/短期动态模型论文.tex` | 短期模型章节的 LaTeX 初稿 |
| 短期论文 PDF | `output/final/短期动态模型论文.pdf` | 已渲染检查的 13 页论文初稿 |
| 短期论文 DOCX | `output/final/短期动态模型论文.docx` | 可编辑 Word 版本，已做基础版式检查 |
| 论文专用图 | `paper/figures/*.png` | 短期模型拟合、误差诊断、机制贡献和候选模型对比 |
| 质量增强报告 | `output/reports/短期模型质量增强报告.md` | 相对基准、残差诊断和局部扰动稳健性 |
| 防御检验报告 | `output/reports/短期模型评委质疑防御报告.md` | 回应 Random Walk、滞后曲线、预测步长、MAPE 和历史窗口边界质疑 |
| 致命质疑报告 | `output/reports/短期模型致命质疑补充防御报告.md` | 回应参数过拟合、恐慌内生性、200 美元稻草人和硬编码天花板质疑 |
| 预测步长说明 | `output/reports/短期模型预测步长说明.md` | 明确短期模型是条件机制递推，不是 T+1 日度交易预测器 |
| 参数来源说明 | `data/metadata/参数来源与可信度说明.md` | 区分附件数据、题面参数、文献依据和模型校准参数 |
| 三情景路径 | `output/scenarios/三情景预测结果.csv` | 阶段 5 乐观、中性、悲观价格路径 |
| 三情景指标 | `output/scenarios/三情景关键指标.csv` | 第 60/90/120/180 天价格、均价、二次跳涨风险 |
| 情景预测图 | `figures/scenario_price_paths.png` | 三情景 60-180 天路径图 |
| 库存风险图 | `figures/inventory_depletion_risk.png` | 商业库存和剩余供需缺口变化 |
| 阶段 5 报告 | `output/reports/stage5_scenario_forecast_report.md` | 可直接改写进论文情景预测小节 |

## 下一步

进入阶段 6：敏感性分析。

阶段 6 的目标是在阶段 5 三情景路径基础上，识别最影响第 180 天价格、外推期峰值和二次跳涨风险的参数。

阶段 6 最小完成标准：

- 输出敏感性分析结果表。
- 生成龙卷风图或关键参数排序图。
- 生成 2-3 个关键参数响应曲线。
- 给出“最关键影响因素”和“政策缓冲优先级”解释。

短期模型论文 PDF/DOCX 复现命令：

```bash
source scripts/project_env.sh
python3 -m src.visualization.short_term_paper_figures
python3 -m src.analysis.short_term_model_quality
python3 -m src.analysis.short_term_model_defense
python3 -m src.analysis.short_term_model_fatal_challenges
./scripts/build_short_term_paper.sh
```

阶段 5 复现命令：

```bash
source scripts/project_env.sh
/opt/homebrew/Caskroom/miniconda/base/envs/mathmodel-oil/bin/python3 -m src.scenarios.forecast_stage5
```

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
