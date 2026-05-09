# 阶段 0.5 工程骨架补强报告

## 完成时间

2026-05-09

## 完成内容

- 新增 `config/` 参数配置层。
- 新增 `data/external/`、`data/interim/`、`data/metadata/`，用于承接外部数据、中间数据和来源说明。
- 将 `src/` 拆为 `data/`、`models/`、`calibration/`、`analysis/`、`visualization/`、`pipeline/`。
- 将 `output/` 拆为 `baseline/`、`calibration/`、`scenarios/`、`sensitivity/`、`reports/`、`runs/`。
- 新增根目录 `README.md` 和 `STATUS.md`。
- 新增 `requirements.txt`，固定阶段 1 起步依赖。
- 新增 `docs/08_架构决策记录.md`，明确当前主线与后置增强项。
- 新增 `paper/figures_mapping.md`，提前建立图表到论文结论的映射。
- 标准化原始数据入口为 `data/raw/brent_daily.csv`。

## 当前主线

```text
Python 数据处理 -> 传统供需基准模型 -> 动态递推模型 -> 参数校准 -> 三情景预测 -> 敏感性分析 -> 论文图表
```

## 验收结论

阶段 0.5 已完成。下一步可以进入阶段 1：数据清洗和探索。
