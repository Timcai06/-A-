# src 目录说明

本目录保存项目可复现代码。

## 模块边界

| 目录 | 作用 |
|---|---|
| `pipeline/` | 数据清洗与自动验证 |
| `models/` | 传统供需基准模型、短期动态递推模型 |
| `calibration/` | 参数搜索、评价指标、校准报告 |
| `scenarios/` | 60--180 天长期三情景预测主模型 |
| `analysis/` | 敏感性、蒙特卡洛、历史稳健性、统计审计、质疑审计 |
| `visualization/` | 总论文和短期素材稿图表生成 |
| `common/` | 路径、指标、绘图等公共工具 |

## 当前主线

1. `src/pipeline/clean_data.py`
2. `src/models/baseline_supply_demand.py`
3. `src/models/dynamic_short_term.py`
4. `src/calibration/calibrate_dynamic_model.py`
5. `src/scenarios/scenario_forecast.py`
6. `src/analysis/sensitivity_analysis.py`
7. `src/analysis/monte_carlo_scenario_tree.py`
8. `src/analysis/statistical_audit.py`
9. `src/visualization/final_paper_figures.py`

最终论文构建入口是 `scripts/build_final_paper.sh`。
