"""Stage 4 multi-objective calibration for the short-term dynamic model."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models import dynamic_short_term as dynamic


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANDOM_SEED = 20260509
SAMPLE_SIZE = 18000


class Stage4Paths:
    calibrated_path_csv = PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv"
    best_parameters_csv = PROJECT_ROOT / "output" / "calibration" / "动态模型最优参数.csv"
    top_candidates_csv = PROJECT_ROOT / "output" / "calibration" / "动态模型候选参数前10.csv"
    segment_errors_csv = PROJECT_ROOT / "output" / "calibration" / "动态模型分段误差.csv"
    report_path = PROJECT_ROOT / "output" / "reports" / "stage4_calibration_report.md"
    figure_path = PROJECT_ROOT / "figures" / "fitted_vs_actual.png"


def rmse(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))


def mae(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(np.mean(np.abs(values)))


def segment_error_rows(simulation: pd.DataFrame) -> list[dict[str, Any]]:
    df = simulation.copy()
    df["error"] = df["simulated_price"] - df["actual_price"]
    df["abs_error"] = df["error"].abs()
    segments = [
        ("全窗口", df.index == df.index),
        ("前期冲击_0到14天", df["day_index"] <= 14),
        ("中期平台_15到35天", (df["day_index"] > 14) & (df["day_index"] <= 35)),
        ("后期再定价_36天以后", df["day_index"] > 35),
        ("高价平台_actual大于等于110", df["actual_price"] >= 110),
        ("低价回落_actual小于100", df["actual_price"] < 100),
    ]
    rows: list[dict[str, Any]] = []
    for name, mask in segments:
        sub = df[mask].copy()
        rows.append(
            {
                "分段": name,
                "样本数": len(sub),
                "RMSE": rmse(sub["error"]),
                "MAE": mae(sub["error"]),
                "平均偏差": float(sub["error"].mean()) if len(sub) else float("nan"),
                "最大绝对误差": float(sub["abs_error"].max()) if len(sub) else float("nan"),
                "实际最低价": float(sub["actual_price"].min()) if len(sub) else float("nan"),
                "实际最高价": float(sub["actual_price"].max()) if len(sub) else float("nan"),
            }
        )
    return rows


def evaluate_simulation(simulation: pd.DataFrame) -> dict[str, float]:
    df = simulation.copy()
    df["error"] = df["simulated_price"] - df["actual_price"]
    peak_error = float(df["simulated_price"].max() - df["actual_price"].max())
    final_error = float(df["simulated_price"].iloc[-1] - df["actual_price"].iloc[-1])
    full_rmse = rmse(df["error"])
    full_mae = mae(df["error"])
    high_rmse = rmse(df.loc[df["actual_price"] >= 110, "error"])
    low_rmse = rmse(df.loc[df["actual_price"] < 100, "error"])
    late_rmse = rmse(df.loc[df["day_index"] > 35, "error"])
    mid_rmse = rmse(df.loc[(df["day_index"] > 14) & (df["day_index"] <= 35), "error"])

    composite_score = (
        full_rmse
        + 0.20 * abs(peak_error)
        + 0.25 * abs(final_error)
        + 0.15 * high_rmse
        + 0.10 * late_rmse
        + 0.10 * low_rmse
    )
    platform_score = high_rmse + 0.40 * abs(peak_error) + 0.40 * abs(final_error)
    return {
        "RMSE": full_rmse,
        "MAE": full_mae,
        "峰值误差": peak_error,
        "末日误差": final_error,
        "高价平台RMSE": high_rmse,
        "低价回落RMSE": low_rmse,
        "中期RMSE": mid_rmse,
        "后期RMSE": late_rmse,
        "模拟峰值": float(df["simulated_price"].max()),
        "实际峰值": float(df["actual_price"].max()),
        "模拟末日价格": float(df["simulated_price"].iloc[-1]),
        "实际末日价格": float(df["actual_price"].iloc[-1]),
        "综合得分": composite_score,
        "平台解释得分": platform_score,
    }


def draw(values: list[Any], rng: np.random.Generator) -> Any:
    return values[int(rng.integers(0, len(values)))]


def sampled_parameter_sets(base_assumptions: dynamic.PhysicalAssumptions) -> list[tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]]:
    rng = np.random.default_rng(RANDOM_SEED)
    sets: list[tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]] = []

    physical_grid = {
        "supply_interruption": [1400, 1500, 1600, 1700, 1800],
        "spr_max_release": [200, 300, 450, 600, 700],
        "spr_delay_days": [3, 7, 10, 14],
        "route_start_day": [7, 14, 21, 30],
        "route_max_capacity": [150, 220, 250, 300],
        "route_ramp_days": [14, 21, 30],
        "long_elasticity": [-0.10, -0.14, -0.18, -0.22, -0.25],
        "fear_initial": [0.10, 0.13, 0.16, 0.20],
        "fear_decay": [0.04, 0.06, 0.08, 0.10, 0.12],
        "inventory_daily_cap": [250, 350, 450, 600],
        "demand_decline_ramp_days": [14, 21, 30],
    }
    behavior_grid = {
        "pressure_scale": [0.020, 0.035, 0.045, 0.060, 0.075, 0.090],
        "risk_weight": [1.60, 2.00, 2.20, 2.60, 3.00, 3.40],
        "uncertainty_floor": [0.08, 0.12, 0.16, 0.20, 0.24, 0.28],
        "inventory_response": [0.20, 0.35, 0.55, 0.75],
        "adjustment_speed": [0.18, 0.25, 0.32, 0.40],
    }

    # Always include the Stage 3 accepted baseline so Stage 4 can compare against it.
    sets.append(
        (
            base_assumptions,
            dynamic.BehavioralParameters(
                pressure_scale=0.045,
                risk_weight=2.20,
                uncertainty_floor=0.240,
                inventory_response=0.550,
                adjustment_speed=0.250,
            ),
        )
    )

    for _ in range(SAMPLE_SIZE):
        physical_kwargs = {key: draw(values, rng) for key, values in physical_grid.items()}
        behavior_kwargs = {key: draw(values, rng) for key, values in behavior_grid.items()}
        assumptions = replace(base_assumptions, **physical_kwargs)
        behavior = dynamic.BehavioralParameters(**behavior_kwargs)
        sets.append((assumptions, behavior))
    return sets


def classify_candidate(row: pd.Series, best_rmse_id: int, best_platform_id: int, best_composite_id: int) -> str:
    labels: list[str] = []
    if int(row["candidate_id"]) == best_composite_id:
        labels.append("综合最优")
    if int(row["candidate_id"]) == best_rmse_id:
        labels.append("RMSE最优")
    if int(row["candidate_id"]) == best_platform_id:
        labels.append("平台解释最优")
    return " / ".join(labels) if labels else "候选"


def calibrate(event_df: pd.DataFrame, base_assumptions: dynamic.PhysicalAssumptions) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    simulations: dict[int, pd.DataFrame] = {}
    for candidate_id, (assumptions, behavior) in enumerate(sampled_parameter_sets(base_assumptions)):
        simulation = dynamic.simulate_dynamic_model(event_df, assumptions, behavior)
        metrics = evaluate_simulation(simulation)
        rows.append(
            {
                "candidate_id": candidate_id,
                **metrics,
                **{f"assumption_{key}": value for key, value in asdict(assumptions).items()},
                **{f"behavior_{key}": value for key, value in asdict(behavior).items()},
            }
        )
        simulations[candidate_id] = simulation

    all_results = pd.DataFrame(rows)
    best_composite_id = int(all_results.sort_values(["综合得分", "RMSE"]).iloc[0]["candidate_id"])
    best_rmse_id = int(all_results.sort_values(["RMSE", "综合得分"]).iloc[0]["candidate_id"])
    best_platform_id = int(all_results.sort_values(["平台解释得分", "综合得分"]).iloc[0]["candidate_id"])
    all_results["候选类型"] = all_results.apply(
        classify_candidate,
        axis=1,
        best_rmse_id=best_rmse_id,
        best_platform_id=best_platform_id,
        best_composite_id=best_composite_id,
    )
    top_candidates = (
        all_results.sort_values(["综合得分", "RMSE", "平台解释得分"]).head(10).reset_index(drop=True)
    )
    representative = all_results[
        all_results["candidate_id"].isin([best_composite_id, best_rmse_id, best_platform_id])
    ].sort_values(["综合得分", "RMSE"])
    best_simulation = simulations[best_composite_id].copy()
    best_simulation["candidate_id"] = best_composite_id
    return best_simulation, top_candidates, representative


def save_figure(simulation: pd.DataFrame) -> None:
    Stage4Paths.figure_path.parent.mkdir(parents=True, exist_ok=True)
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
        label="阶段4校准后动态模型",
    )
    ax.axhspan(110, 120, color="#10b981", alpha=0.10, label="题面110-120区间")
    ax.set_title("阶段4校准后动态模型与实际价格对比")
    ax.set_xlabel("日期")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(Stage4Paths.figure_path, dpi=180)
    plt.close(fig)


def write_outputs(best_simulation: pd.DataFrame, top_candidates: pd.DataFrame, representative: pd.DataFrame) -> pd.DataFrame:
    Stage4Paths.calibrated_path_csv.parent.mkdir(parents=True, exist_ok=True)
    Stage4Paths.report_path.parent.mkdir(parents=True, exist_ok=True)
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
    segment_rows = "\n".join(
        "| {分段} | {样本数:.0f} | {RMSE:.2f} | {MAE:.2f} | {平均偏差:.2f} | {最大绝对误差:.2f} |".format(
            **row
        )
        for row in segment_errors.to_dict("records")
    )

    return f"""# 阶段 4 参数校准报告

## 运行结论

阶段 4 已完成短期动态模型的多目标参数校准。本阶段不只追求 RMSE 最小，而是同时考虑整体误差、峰值误差、末日价格误差、高价平台误差、后期误差和低价回落误差。

综合最优候选的 RMSE 为 {best["RMSE"]:.2f}，MAE 为 {best["MAE"]:.2f}，模拟峰值为 {best["模拟峰值"]:.2f}，模拟末日价格为 {best["模拟末日价格"]:.2f}。与阶段 3 初始筛选相比，本阶段给出了更完整的候选参数比较和分段误差解释。

## 校准目标函数

```text
综合得分 = RMSE
        + 0.20 * abs(峰值误差)
        + 0.25 * abs(末日误差)
        + 0.15 * 高价平台RMSE
        + 0.10 * 后期RMSE
        + 0.10 * 低价回落RMSE
```

该目标函数用于避免单纯追求 RMSE 时牺牲峰值和平台解释能力。

## 三类代表候选

| 类型 | RMSE | 峰值误差 | 末日误差 | 高价平台RMSE | 低价回落RMSE | 综合得分 |
|---|---:|---:|---:|---:|---:|---:|
| 综合最优 | {best["RMSE"]:.2f} | {best["峰值误差"]:.2f} | {best["末日误差"]:.2f} | {best["高价平台RMSE"]:.2f} | {best["低价回落RMSE"]:.2f} | {best["综合得分"]:.2f} |
| RMSE最优 | {rmse_best["RMSE"]:.2f} | {rmse_best["峰值误差"]:.2f} | {rmse_best["末日误差"]:.2f} | {rmse_best["高价平台RMSE"]:.2f} | {rmse_best["低价回落RMSE"]:.2f} | {rmse_best["综合得分"]:.2f} |
| 平台解释最优 | {platform_best["RMSE"]:.2f} | {platform_best["峰值误差"]:.2f} | {platform_best["末日误差"]:.2f} | {platform_best["高价平台RMSE"]:.2f} | {platform_best["低价回落RMSE"]:.2f} | {platform_best["综合得分"]:.2f} |

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
| pressure_scale | {best["behavior_pressure_scale"]:.3f} |
| risk_weight | {best["behavior_risk_weight"]:.3f} |
| uncertainty_floor | {best["behavior_uncertainty_floor"]:.3f} |
| inventory_response | {best["behavior_inventory_response"]:.3f} |
| adjustment_speed | {best["behavior_adjustment_speed"]:.3f} |

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


def main() -> None:
    base_config = dynamic.load_yaml(dynamic.BASE_CONFIG_PATH)
    scenario_config = dynamic.load_yaml(dynamic.SCENARIO_CONFIG_PATH)
    problem_params = dynamic.load_problem_parameters(dynamic.PROBLEM_PARAMETERS_PATH)
    dynamic_paths = dynamic.resolve_paths(base_config)
    event_df = dynamic.load_event_window(dynamic_paths.event_csv)
    base_assumptions = dynamic.resolve_assumptions(base_config, scenario_config, problem_params)

    best_simulation, top_candidates, representative = calibrate(event_df, base_assumptions)
    segment_errors = write_outputs(best_simulation, top_candidates, representative)
    write_report(top_candidates, representative, segment_errors)

    print("Stage 4 complete")
    print(f"Calibrated path: {Stage4Paths.calibrated_path_csv.relative_to(PROJECT_ROOT)}")
    print(f"Best parameters: {Stage4Paths.best_parameters_csv.relative_to(PROJECT_ROOT)}")
    print(f"Top candidates: {Stage4Paths.top_candidates_csv.relative_to(PROJECT_ROOT)}")
    print(f"Segment errors: {Stage4Paths.segment_errors_csv.relative_to(PROJECT_ROOT)}")
    print(f"Report: {Stage4Paths.report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
