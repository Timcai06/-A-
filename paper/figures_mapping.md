# 论文图表映射表

本文档用于记录每张图如何进入论文正文，避免最后阶段重新整理材料。

当前 `figures/` 与 `paper/figures/` 中的核心图表已经进入或作为备用素材服务于 `paper/总论文.tex`，总论文是唯一主稿。

需要注意：图 1 和图 3 使用 2017-2026 全历史数据；图 2、图 4、图 5 聚焦 2026 冲突窗口，因此横轴只显示 2026 年附近的日期。

短期动态模型素材图存放在 `paper/figures/`。这些图已经择优迁移进总论文，用于支撑基准对比、滞后检验、拐点检验、机制消融、过拟合压力测试和稳健性带；部分重复图保留为备用素材，不再全部塞入正文。

## 当前生成情况

| 状态 | 数量 | 图号 |
|---|---:|---|
| 阶段图 | 9 | 数据、基准、三情景、敏感性分析 |
| 短期模型素材图 | 13 | 精选迁移进总论文，其余保留备用 |

## 图表清单

| 图号 | 图表文件 | 时间范围 | 对应论文位置 | 支撑结论 | 数据或结果来源 | 状态 |
|---|---|---|---|---|---|---|
| 图1 | `figures/price_trend.png` | 2017-09-01 至 2026-05-05 | 数据说明 | 布伦特油价长期走势提供历史背景 | `data/processed/布伦特原油期货主力合约价格数据_清洗后.csv` | 已生成 |
| 图2 | `figures/event_window_price.png` | 2026-03-02 至 2026-05-05 | 数据说明 | 冲突窗口价格出现峰值和平台区间 | `data/processed/布伦特原油期货主力合约价格数据_冲突窗口.csv` | 已生成 |
| 图3 | `figures/return_volatility.png` | 2017-09-01 至 2026-05-05 | 数据说明 | 冲突窗口前后可结合收益率和波动率观察市场冲击 | `data/processed/布伦特原油期货主力合约价格数据_清洗后.csv` | 已生成 |
| 图4 | `figures/baseline_vs_actual.png` | 2026-03-02 至 2026-05-05 | 模型一结果 | 传统供需模型高估现实价格 | `output/baseline/传统供需基准模型结果.csv` | 已生成 |
| 图5 | `figures/fitted_vs_actual.png` | 2026-03-02 至 2026-05-05 | 模型二结果 | 校准后的动态模型能解释 110-120 美元平台 | `output/calibration/动态模型校准后路径.csv` | 已生成 |
| 图6 | `figures/scenario_price_paths.png` | 2026-05-06 至外推第180天 | 模型三结果 | 三情景下长期均衡价格不同 | `output/scenarios/三情景预测结果.csv` | 已生成 |
| 图7 | `figures/inventory_depletion_risk.png` | 2026-05-06 至外推第180天 | 模型三结果 | 库存耗尽可能触发二次跳涨风险 | `output/scenarios/三情景预测结果.csv` | 已生成 |
| 图8 | `figures/sensitivity_tornado_180day.png` | 阶段 6 | 敏感性分析 | 不确定性与制度风险强度、地缘风险权重、供应中断量最敏感 | `output/sensitivity/阶段6_参数重要性排序.csv` | 已生成 |
| 图9 | `figures/sensitivity_parameter_response.png` | 阶段 6 | 敏感性分析 | 前三高敏感参数的响应曲线 | `output/sensitivity/阶段6_敏感性分析结果.csv` | 已生成 |
| 图10 | `figures/monte_carlo_price_fan.png` | 阶段 6.5 | 蒙特卡洛情景树 | 2000 条联合扰动路径形成价格概率区间 | `output/monte_carlo/蒙特卡洛路径分位数.csv` | 已生成 |
| 图11 | `figures/monte_carlo_tail_risk.png` | 阶段 6.5 | 蒙特卡洛情景树 | 量化突破 120/130 美元和高二次跳涨风险概率 | `output/monte_carlo/蒙特卡洛尾部风险摘要.csv` | 已生成 |

## 短期模型论文专用图

| 论文内编号 | 图表文件 | 时间范围 | 支撑结论 | 数据或结果来源 | 状态 |
|---|---|---|---|---|---|
| 总路线图 | `paper/figures/论文总体技术路线图.png` | 全论文 | 展示从附件数据、短期拟合到长期情景预测和敏感性分析的完整技术路线 | `src/visualization/final_paper_figures.py` | 已生成 |
| 初稿图4 | `paper/figures/短期模型拟合效果.png` | 2026-03-02 至 2026-05-05 | 校准模型在整体路径上贴合真实油价 | `output/calibration/动态模型校准后路径.csv` | 已生成 |
| 初稿图5 | `paper/figures/短期模型误差诊断.png` | 2026-03-02 至 2026-05-05 | 误差集中在局部阶段，整体无系统性漂移 | `output/calibration/动态模型校准后路径.csv`、`output/calibration/动态模型分段误差.csv` | 已生成 |
| 初稿图6 | `paper/figures/短期模型机制贡献.png` | 2026-03-02 至 2026-05-05 | 风险溢价推高价格，缓冲和预期修复压低价格 | `output/calibration/动态模型校准后路径.csv` | 已生成 |
| 初稿图7 | `paper/figures/短期模型机制消融实验.png` | 2026-03-02 至 2026-05-05 | 逐项关闭题面物理机制和市场价格形成机制，证明扩展项不是任意拟合补丁 | `output/calibration/短期模型机制消融实验.csv` | 已生成 |
| 初稿图8 | `paper/figures/候选模型误差对比.png` | 阶段 4 候选模型 | 综合最优模型同时控制整体误差与分段误差 | `output/calibration/动态模型候选参数前10.csv` | 已生成 |
| 初稿图9 | `paper/figures/短期模型基准对比.png` | 2026-03-02 至 2026-05-05 | 证明短期模型优于 Random Walk、三日均值、漂移随机游走和滚动 ARIMA 基准 | `output/calibration/短期模型基准对比.csv` | 已生成 |
| 初稿图10 | `paper/figures/短期模型滞后平移检验.png` | 2026-03-02 至 2026-05-05 | 检查预测曲线是否只是滞后复制真实曲线 | `output/calibration/短期模型滞后平移检验.csv` | 已生成 |
| 初稿图11 | `paper/figures/短期模型拐点局部检验.png` | 2026-03-02 至 2026-03-14；2026-04-22 至 2026-05-05 | 放大暴涨和反转窗口，验证主要拐点不是马后炮跟随 | `output/calibration/短期模型拐点检验.csv` | 已生成 |
| 初稿图12 | `paper/figures/短期模型过拟合压力测试.png` | 2026-03-02 至 2026-05-05 | 用 \(\pm 15\%\) 参数扰动检查价格平台是否依赖单一参数点 | `output/calibration/短期模型过拟合压力测试.csv` | 已生成 |
| 初稿图13 | `paper/figures/短期模型残差诊断增强.png` | 2026-03-02 至 2026-05-05 | 检查残差分布、滚动误差和自相关 | `output/calibration/动态模型校准后路径.csv`、`output/calibration/短期模型滚动误差.csv` | 已生成 |
| 初稿图14 | `paper/figures/短期模型稳健性带.png` | 2026-03-02 至 2026-05-05 | 展示优秀局部扰动样本形成的稳定价格带 | `output/calibration/短期模型稳健性区间.csv` | 已生成 |
| 总览图 | `paper/figures/短期拟合与长期预测总览.png` | 2026-03-02 至 2026-08-29 | 连接短期拟合与 60--180 天三情景外推，明确真实数据截止点 | `output/calibration/动态模型校准后路径.csv`、`output/scenarios/三情景预测结果.csv` | 已生成 |

## 说明

最终论文以 `paper/总论文.tex` 为准。短期模型论文素材稿可继续保留，但不再作为独立 PDF/DOCX 交付。
