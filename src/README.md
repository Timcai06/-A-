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

核心模型链路：

1. `src/pipeline/clean_data.py`
2. `src/models/baseline_supply_demand.py`
3. `src/models/dynamic_short_term.py`
4. `src/calibration/calibrate_dynamic_model.py`
5. `src/scenarios/scenario_forecast.py`
6. `src/analysis/sensitivity_analysis.py`
7. `src/analysis/monte_carlo_scenario_tree.py`
8. `src/analysis/statistical_audit.py`
9. `src/visualization/final_paper_figures.py`

论文构建链路还会调用外部约束、市场风险、短期质量门和图表审计模块，包括：

- `src/data/fetch_opec_balance.py`
- `src/data/fetch_futures_term_structure.py`
- `src/analysis/market_risk_variables.py`
- `src/analysis/short_term_residual_correction.py`
- `src/analysis/short_term_parameter_event_audit.py`
- `src/analysis/short_term_ml_features.py`
- `src/analysis/short_term_ml_residual_model.py`
- `src/analysis/short_term_error_attribution.py`
- `src/analysis/short_term_rolling_validation.py`
- `src/analysis/short_term_final_quality_gate.py`
- `src/analysis/paper_visual_audit.py`
- `src/visualization/advanced_candidate_figures.py`

最终论文构建入口是 `scripts/build/build_final_paper.sh`，根目录 `scripts/build_final_paper.sh` 仅作为兼容转发入口。
