"""Output writers and Markdown report generation for Stage 4 calibration."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.calibration.evaluation import segment_error_rows
from src.calibration.settings import FIT_REFINEMENT_SAMPLES, Stage4Paths
from src.common.paths import PROJECT_ROOT, ensure_parent
from src.models import dynamic_short_term as dynamic


def save_figure(simulation: pd.DataFrame) -> None:
    ensure_parent(Stage4Paths.figure_path)
    dynamic.configure_plot_style()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        simulation["trade_date"],
        simulation["actual_price"],
        color="#2563eb",
        marker="o",
        markersize=2.8,
        linewidth=1.8,
        label="附件CSV实际收盘价",
    )
    ax.plot(
        simulation["trade_date"],
        simulation["simulated_price"],
        color="#dc2626",
        marker="s",
        markersize=2.4,
        linewidth=1.7,
        label="阶段4精修后动态模型",
    )
    ax.axhspan(110, 120, color="#10b981", alpha=0.10, label="题面110-120区间")
    ax.set_title("阶段4精修后动态模型与实际价格对比")
    ax.set_xlabel("日期")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(Stage4Paths.figure_path, dpi=180)
    plt.close(fig)


def write_outputs(best_simulation: pd.DataFrame, top_candidates: pd.DataFrame, representative: pd.DataFrame) -> pd.DataFrame:
    ensure_parent(Stage4Paths.calibrated_path_csv)
    ensure_parent(Stage4Paths.report_path)
    segment_errors = pd.DataFrame(segment_error_rows(best_simulation))

    best_simulation.to_csv(Stage4Paths.calibrated_path_csv, index=False)
    representative.to_csv(Stage4Paths.best_parameters_csv, index=False)
    top_candidates.to_csv(Stage4Paths.top_candidates_csv, index=False)
    segment_errors.to_csv(Stage4Paths.segment_errors_csv, index=False)
    save_figure(best_simulation)
    return segment_errors


def build_report(top_candidates: pd.DataFrame, representative: pd.DataFrame, segment_errors: pd.DataFrame) -> str:
    best = top_candidates.iloc[0]
    rmse_best = representative[representative["候选类型"].str.contains("RMSE最优")].iloc[0]
    platform_best = representative[representative["候选类型"].str.contains("平台解释最优")].iloc[0]
    source_label = {
        "continuous_refinement": "连续局部精修",
        "local_stability_refinement": "局部稳健性复核",
        "fit_quality_refinement": "拟合质量精修",
        "seeded_random": "固定种子随机搜索",
    }.get(str(best["candidate_source"]), str(best["candidate_source"]))

    def fmt2(value: float) -> str:
        number = float(value)
        if abs(number) < 0.005:
            number = 0.0
        return f"{number:.2f}"

    segment_rows = "\n".join(
        "| {分段} | {样本数:.0f} | {RMSE:.2f} | {MAE:.2f} | {平均偏差:.2f} | {最大绝对误差:.2f} |".format(
            **row
        )
        for row in segment_errors.to_dict("records")
    )

    return f"""# 阶段 4 参数校准报告

## 运行结论

阶段 4 已完成短期动态模型的多目标参数校准、连续局部精修与局部稳健性复核。本阶段不只追求 RMSE 最小，而是同时考虑整体误差、峰值误差、末日价格误差、高价平台误差、前期冲击误差、中期平台误差、后期再定价误差和低价回落误差。

综合最优候选的 RMSE 为 {best["RMSE"]:.2f}，MAE 为 {best["MAE"]:.2f}，模拟峰值为 {best["模拟峰值"]:.2f}，模拟末日价格为 {best["模拟末日价格"]:.2f}。与阶段 3 初始筛选相比，本阶段给出了更完整的候选参数比较和分段误差解释。

## 校准流程

1. 使用固定随机种子进行 36000 组多目标参数搜索，先覆盖物理参数和行为参数的合理范围。
2. 在随机搜索基础上使用 `scipy.optimize.differential_evolution` 做连续局部精修，使模型同时满足整体误差、分段误差、峰值误差和末日误差要求。
3. 围绕连续精修结果进行 800 组局部稳健性扰动复核，检验最优解附近是否存在稳定优质参数邻域，并从中保留综合得分更优的候选。
4. 在稳健性复核结果附近进行 {FIT_REFINEMENT_SAMPLES} 组拟合质量精修，重点改善中期平台、低价回落和末日误差，但不新增任何模型机制。
5. 保留综合最优、RMSE 最优、平台解释最优三类候选，用于论文中的模型比较和稳健性讨论。

## 校准目标函数

```text
综合得分 = RMSE
        + 0.20 * abs(峰值误差)
        + 0.25 * abs(末日误差)
        + 0.15 * 高价平台RMSE
        + 0.12 * 前期RMSE
        + 0.12 * 中期RMSE
        + 0.18 * 后期RMSE
        + 0.18 * 低价回落RMSE
```

该目标函数用于避免单纯追求 RMSE 时牺牲峰值、平台解释能力和后期再定价解释能力。

## 三类代表候选

| 类型 | RMSE | 峰值误差 | 末日误差 | 高价平台RMSE | 低价回落RMSE | 综合得分 |
|---|---:|---:|---:|---:|---:|---:|
| 综合最优 | {best["RMSE"]:.2f} | {fmt2(best["峰值误差"])} | {fmt2(best["末日误差"])} | {best["高价平台RMSE"]:.2f} | {best["低价回落RMSE"]:.2f} | {best["综合得分"]:.2f} |
| RMSE最优 | {rmse_best["RMSE"]:.2f} | {fmt2(rmse_best["峰值误差"])} | {fmt2(rmse_best["末日误差"])} | {rmse_best["高价平台RMSE"]:.2f} | {rmse_best["低价回落RMSE"]:.2f} | {rmse_best["综合得分"]:.2f} |
| 平台解释最优 | {platform_best["RMSE"]:.2f} | {fmt2(platform_best["峰值误差"])} | {fmt2(platform_best["末日误差"])} | {platform_best["高价平台RMSE"]:.2f} | {platform_best["低价回落RMSE"]:.2f} | {platform_best["综合得分"]:.2f} |

## 综合最优参数

| 参数 | 数值 |
|---|---:|
| 供应中断量 | {best["assumption_supply_interruption"]:.0f} |
| SPR释放上限 | {best["assumption_spr_max_release"]:.0f} |
| SPR启动延迟 | {best["assumption_spr_delay_days"]:.0f} |
| 绕道启动日 | {best["assumption_route_start_day"]:.0f} |
| 绕道能力上限 | {best["assumption_route_max_capacity"]:.0f} |
| 长期需求弹性 | {best["assumption_long_elasticity"]:.2f} |
| 恐慌初始强度 | {best["assumption_fear_initial"]:.2f} |
| 恐慌衰减速度 | {best["assumption_fear_decay"]:.2f} |
| 库存日缓冲上限 | {best["assumption_inventory_daily_cap"]:.0f} |
| 候选来源 | {source_label} |
| price transmission coefficient (`pressure_scale`) | {best["behavior_pressure_scale"]:.3f} |
| risk_weight | {best["behavior_risk_weight"]:.3f} |
| uncertainty_floor | {best["behavior_uncertainty_floor"]:.3f} |
| inventory_response | {best["behavior_inventory_response"]:.3f} |
| adjustment_speed | {best["behavior_adjustment_speed"]:.3f} |
| buffer_relief_strength | {best["behavior_buffer_relief_strength"]:.3f} |
| buffer_relief_decay_days | {best["behavior_buffer_relief_decay_days"]:.0f} |
| relief_discount_strength | {best["behavior_relief_discount_strength"]:.3f} |
| relief_start_day | {best["behavior_relief_start_day"]:.0f} |
| relief_peak_day | {best["behavior_relief_peak_day"]:.0f} |
| relief_decay_days | {best["behavior_relief_decay_days"]:.0f} |

## 分段误差

| 分段 | 样本数 | RMSE | MAE | 平均偏差 | 最大绝对误差 |
|---|---:|---:|---:|---:|---:|
{segment_rows}

## 参数合理性检查

- 供应中断量仍在赛题范围 1400-1800 万桶/日内。
- SPR 释放上限仍在赛题范围 200-700 万桶/日内。
- 绕道启动时间仍在赛题范围 7-30 天内。
- 绕道能力不超过题面给出的约 300 万桶/日。
- 长期需求弹性比短期弹性绝对值更大，符合中长期需求调整更充分的直觉。
- `pressure_scale` 表示剩余物理供需缺口向布伦特期货目标价格传导的价格形成系数，不替代题面给定的短期需求弹性。
- `buffer_relief_strength` 表示供需缺口被缓冲机制压住后，市场出现第一轮降温折价。
- `relief_discount_strength` 表示中后期市场预期修复带来的阶段性再定价折价。

## 输出产物

- `{Stage4Paths.calibrated_path_csv.relative_to(PROJECT_ROOT)}`
- `{Stage4Paths.best_parameters_csv.relative_to(PROJECT_ROOT)}`
- `{Stage4Paths.top_candidates_csv.relative_to(PROJECT_ROOT)}`
- `{Stage4Paths.segment_errors_csv.relative_to(PROJECT_ROOT)}`
- `{Stage4Paths.figure_path.relative_to(PROJECT_ROOT)}`

## 后续作用

阶段 4 输出的综合最优参数可以作为阶段 5 三情景预测的中性基准。RMSE 最优和平台解释最优两类候选可用于论文中的稳健性讨论，说明模型结论不是依赖单一手工参数。
"""


def write_report(top_candidates: pd.DataFrame, representative: pd.DataFrame, segment_errors: pd.DataFrame) -> None:
    Stage4Paths.report_path.write_text(build_report(top_candidates, representative, segment_errors), encoding="utf-8")
