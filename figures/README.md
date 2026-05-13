# figures 目录说明

本目录保存阶段图表和通用模型图，主要由数据清洗、基准模型、情景预测、敏感性分析、蒙特卡洛和外部审计脚本生成。

## 与 paper/figures 的区别

| 目录 | 用途 |
|---|---|
| `figures/` | 阶段图表、通用图、模型运行图 |
| `paper/figures/` | 最终论文直接引用的高质量排版图 |
| `output/candidate_figures/` | 候选图、试样图、视觉方案对比图，不直接进入论文 |

最终论文优先引用 `paper/figures/`；`figures/` 主要用于阶段报告、过程检查和补充证据。

未被论文正文或附录引用的候选图不要放入 `paper/figures/`。候选图只有在确定进入论文后，才复制或迁移到 `paper/figures/` 并同步更新 `paper/figures_mapping.md`。

图表来源和论文编号见 `paper/figures_mapping.md`。
