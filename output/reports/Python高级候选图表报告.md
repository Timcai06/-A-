# Python高级候选图表报告

## 输出规则

本轮图表均为候选图，未进入论文引用前只保存在 `output/candidate_figures/`，不放入 `paper/figures/`。

## 候选图清单

- `output/candidate_figures/Python机制压力瀑布候选图.png`
- `output/candidate_figures/Python短期滚动拟合质量候选图.png`
- `output/candidate_figures/Python短期误差日历热力候选图.png`
- `output/candidate_figures/Python长期敏感性热力候选图.png`
- `output/candidate_figures/Python候选模型权衡气泡候选图.png`
- `output/candidate_figures/Seaborn滞后特征相关性热力候选图.png`
- `output/candidate_figures/Seaborn短期残差分段分布候选图.png`

## 初步判断

- 机制贡献瀑布图适合解释短期峰值价格由哪些机制项推高或压低。
- 滚动拟合质量图适合回答短期模型是否只在局部窗口表现较好。
- 误差日历热力图适合快速暴露误差集中在哪几周和哪些交易日。
- 长期敏感性热力图适合替代单一龙卷风图，展示不同扰动档位下的价格方向。
- 候选模型权衡气泡图适合说明主模型选择不是单一 RMSE 排序，而是多目标权衡。
- Seaborn 滞后特征相关性热力图适合说明历史机器学习特征的冗余结构和信息边界。
- Seaborn 短期残差分段分布图适合检查冲击启动、高位平台和回落修复三段的误差偏态。
