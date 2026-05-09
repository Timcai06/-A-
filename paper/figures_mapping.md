# 论文图表映射表

本文档用于记录每张图如何进入论文正文，避免最后阶段重新整理材料。

当前 `figures/` 目录中实际已有 5 张图，对应图 1 至图 5。图 6 至图 8 是后续阶段的计划图，只有完成阶段 5 情景预测和阶段 6 敏感性分析后才会生成。

## 当前生成情况

| 状态 | 数量 | 图号 |
|---|---:|---|
| 已生成 | 5 | 图1-图5 |
| 待生成 | 3 | 图6-图8 |

## 图表清单

| 图号 | 图表文件 | 对应论文位置 | 支撑结论 | 数据或结果来源 | 状态 |
|---|---|---|---|---|---|
| 图1 | `figures/price_trend.png` | 数据说明 | 布伦特油价长期走势提供历史背景 | `data/processed/布伦特原油期货主力合约价格数据_清洗后.csv` | 已生成 |
| 图2 | `figures/event_window_price.png` | 数据说明 | 冲突窗口价格出现峰值和平台区间 | `data/processed/布伦特原油期货主力合约价格数据_冲突窗口.csv` | 已生成 |
| 图3 | `figures/return_volatility.png` | 数据说明 | 冲突窗口前后可结合收益率和波动率观察市场冲击 | `data/processed/布伦特原油期货主力合约价格数据_清洗后.csv` | 已生成 |
| 图4 | `figures/baseline_vs_actual.png` | 模型一结果 | 传统供需模型高估现实价格 | `output/baseline/传统供需基准模型结果.csv` | 已生成 |
| 图5 | `figures/fitted_vs_actual.png` | 模型二结果 | 校准后的动态模型能解释 110-120 美元平台 | `output/calibration/动态模型校准后路径.csv` | 已生成 |
| 图6 | `figures/scenario_price_paths.png` | 模型三结果 | 三情景下长期均衡价格不同 | `output/scenarios/三情景预测结果.csv` | 待生成 |
| 图7 | `figures/inventory_depletion_risk.png` | 模型三结果 | 库存耗尽可能触发二次跳涨风险 | `output/scenarios/三情景预测结果.csv` | 待生成 |
| 图8 | `figures/sensitivity_tornado.png` | 敏感性分析 | SPR、需求弹性等因素影响最大 | `output/sensitivity/敏感性分析结果.csv` | 待生成 |

## 待生成图表来源

| 图号 | 生成阶段 | 触发条件 |
|---|---|---|
| 图6 | 阶段 5 | `output/scenarios/三情景预测结果.csv` 生成后 |
| 图7 | 阶段 5 | 情景预测输出库存路径和库存风险后 |
| 图8 | 阶段 6 | `output/sensitivity/敏感性分析结果.csv` 生成后 |
