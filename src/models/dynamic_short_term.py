"""Stage 3 short-term dynamic recursion model.

The model explains why the event-window Brent price stayed far below the
traditional static supply-demand benchmark. It keeps five mechanisms explicit:
strategic reserve release, commercial inventory buffering, reroute recovery,
demand contraction, and panic decay.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.common.metrics import mae as calc_mae
from src.common.metrics import rmse as calc_rmse
from src.common.paths import PROJECT_ROOT, ensure_parent
from src.common.plotting import configure_plot_style as apply_plot_style

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base.yml"
SCENARIO_CONFIG_PATH = PROJECT_ROOT / "config" / "scenarios.yml"
PROBLEM_PARAMETERS_PATH = PROJECT_ROOT / "data" / "metadata" / "题面参数表.csv"

# Daily decay speed for blockade risk premium. It is separated from the
# calibrated risk weight so sensitivity analysis can audit whether the long-run
# price path depends on one hidden exponential constant.
BLOCKADE_RISK_DECAY = 0.004


@dataclass(frozen=True)
class DynamicPaths:
    event_csv: Path
    result_csv: Path
    metrics_csv: Path
    figure_path: Path
    report_path: Path


@dataclass(frozen=True)
class PhysicalAssumptions:
    base_supply: float
    base_demand: float
    base_elasticity: float
    long_elasticity: float
    supply_interruption: float
    spr_max_release: float
    spr_delay_days: int
    spr_ramp_days: int
    route_start_day: int
    route_max_capacity: float
    route_ramp_days: int
    commercial_inventory: float
    inventory_daily_cap: float
    observed_demand_decline: float
    demand_decline_ramp_days: int
    fear_initial: float
    fear_decay: float


@dataclass(frozen=True)
class BehavioralParameters:
    # Price-transmission coefficient from residual physical gap to Brent futures target price.
    # It does not replace demand elasticity; elasticity is still used in effective demand.
    pressure_scale: float
    risk_weight: float
    uncertainty_floor: float
    inventory_response: float
    adjustment_speed: float
    buffer_relief_strength: float = 0.0
    buffer_relief_decay_days: int = 10
    relief_discount_strength: float = 0.0
    relief_start_day: int = 35
    relief_peak_day: int = 46
    relief_decay_days: int = 16


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_problem_parameters(path: Path) -> dict[str, float]:
    params = pd.read_csv(path)
    values: dict[str, float] = {}
    for _, row in params.iterrows():
        raw_value = row.get("数值")
        lower = row.get("下限")
        upper = row.get("上限")
        if pd.notna(raw_value):
            values[str(row["参数名"])] = float(raw_value)
        elif pd.notna(lower) and pd.notna(upper):
            values[str(row["参数名"])] = (float(lower) + float(upper)) / 2
    return values


def resolve_paths(base_config: dict[str, Any]) -> DynamicPaths:
    paths = base_config["paths"]
    return DynamicPaths(
        event_csv=PROJECT_ROOT / paths["processed_event_window_csv"],
        result_csv=PROJECT_ROOT / "output" / "calibration" / "短期动态递推模型结果.csv",
        metrics_csv=PROJECT_ROOT / "output" / "calibration" / "短期动态递推模型误差指标.csv",
        figure_path=PROJECT_ROOT / "figures" / "fitted_vs_actual.png",
        report_path=PROJECT_ROOT / "output" / "reports" / "短期动态递推模型报告.md",
    )


def resolve_assumptions(
    base_config: dict[str, Any],
    scenario_config: dict[str, Any],
    problem_params: dict[str, float],
) -> PhysicalAssumptions:
    base = base_config["baseline_assumptions"]
    neutral = scenario_config["scenarios"]["neutral"]
    return PhysicalAssumptions(
        base_supply=float(base["pre_war_supply"]),
        base_demand=float(base["pre_war_demand"]),
        base_elasticity=float(base["short_term_price_elasticity"]),
        long_elasticity=float(neutral["demand_elasticity_long_term"]),
        supply_interruption=float(neutral["supply_interruption"]),
        spr_max_release=float(neutral["spr_max_release"]),
        spr_delay_days=int(neutral["spr_delay_days"]),
        spr_ramp_days=7,
        route_start_day=int(neutral["route_start_day"]),
        route_max_capacity=float(neutral["route_max_capacity"]),
        route_ramp_days=21,
        commercial_inventory=float(problem_params.get("commercial_inventory", 58000)),
        inventory_daily_cap=450.0,
        observed_demand_decline=float(problem_params.get("observed_demand_decline", 430)),
        demand_decline_ramp_days=21,
        fear_initial=0.16,
        fear_decay=float(neutral["fear_decay"]),
    )


def load_event_window(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Event-window CSV not found: {path}")
    return pd.read_csv(path, parse_dates=["trade_date"]).sort_values("trade_date").reset_index(drop=True)


def ramp(day: float, start_day: float, ramp_days: float, max_value: float) -> float:
    if day < start_day:
        return 0.0
    if ramp_days <= 0:
        return max_value
    return max_value * min((day - start_day + 1) / ramp_days, 1.0)


def interpolate_elasticity(day: float, assumptions: PhysicalAssumptions, horizon_days: float = 60.0) -> float:
    weight = min(max(day / horizon_days, 0.0), 1.0)
    return assumptions.base_elasticity + (assumptions.long_elasticity - assumptions.base_elasticity) * weight


def expectation_relief_discount(day: float, base_price: float, behavior: BehavioralParameters) -> float:
    """Temporary discount after markets observe buffers and rerouting taking effect."""
    if behavior.relief_discount_strength <= 0 or day < behavior.relief_start_day:
        return 0.0

    ramp_days = max(behavior.relief_peak_day - behavior.relief_start_day, 1)
    buildup = min((day - behavior.relief_start_day) / ramp_days, 1.0)
    fade = np.exp(-max(day - behavior.relief_peak_day, 0) / max(behavior.relief_decay_days, 1))
    return float(base_price * behavior.relief_discount_strength * buildup * fade)


def buffer_confirmation_discount(
    day: float,
    gap_closure_day: float | None,
    base_price: float,
    behavior: BehavioralParameters,
) -> float:
    """Short-lived discount after the residual supply gap is visibly buffered."""
    if behavior.buffer_relief_strength <= 0 or gap_closure_day is None or day < gap_closure_day:
        return 0.0

    days_since_closure = day - gap_closure_day
    buildup = 1 - np.exp(-days_since_closure / 3)
    fade = np.exp(-days_since_closure / max(behavior.buffer_relief_decay_days, 1))
    return float(base_price * behavior.buffer_relief_strength * buildup * fade)


def simulate_dynamic_model(
    event_df: pd.DataFrame,
    assumptions: PhysicalAssumptions,
    behavior: BehavioralParameters,
    time_axis: str = "calendar",
) -> pd.DataFrame:
    first_date = event_df["trade_date"].min()
    base_price = float(event_df.iloc[0]["pre_close"])
    previous_price = base_price
    previous_fear_excess = assumptions.fear_initial
    inventory_remaining = assumptions.commercial_inventory
    gap_closure_day: int | None = None
    rows: list[dict[str, Any]] = []

    for step_index, actual_row in enumerate(event_df.itertuples(index=False)):
        trade_date = actual_row.trade_date
        if time_axis == "calendar":
            day_index = int((trade_date - first_date).days)
        elif time_axis == "trading_day":
            day_index = step_index
        else:
            raise ValueError(f"Unsupported time_axis: {time_axis}")
        elasticity = interpolate_elasticity(day_index, assumptions)
        price_ratio = max(previous_price / base_price, 0.1)

        price_adjusted_demand = assumptions.base_demand * (price_ratio**elasticity)
        demand_decline = ramp(day_index, 0, assumptions.demand_decline_ramp_days, assumptions.observed_demand_decline)
        effective_demand = max(price_adjusted_demand - demand_decline, assumptions.base_demand * 0.70)

        spr_release = ramp(
            day_index,
            assumptions.spr_delay_days,
            assumptions.spr_ramp_days,
            assumptions.spr_max_release,
        )
        route_supply = ramp(
            day_index,
            assumptions.route_start_day,
            assumptions.route_ramp_days,
            assumptions.route_max_capacity,
        )
        supply_without_inventory = assumptions.base_supply - assumptions.supply_interruption + spr_release + route_supply
        raw_gap = max(effective_demand - supply_without_inventory, 0.0)

        inventory_buffer = min(
            raw_gap * behavior.inventory_response,
            assumptions.inventory_daily_cap,
            inventory_remaining,
        )
        inventory_remaining -= inventory_buffer
        effective_supply = supply_without_inventory + inventory_buffer
        residual_gap = max(effective_demand - effective_supply, 0.0)
        if gap_closure_day is None and residual_gap <= assumptions.base_demand * 0.005:
            gap_closure_day = day_index

        fear_excess = assumptions.fear_initial * np.exp(-assumptions.fear_decay * day_index)
        shortage_pressure = (
            base_price
            * behavior.pressure_scale
            * (residual_gap / assumptions.base_demand)
            / max(abs(elasticity), 0.01)
        )
        blockade_risk_premium = (
            base_price
            * behavior.risk_weight
            * (assumptions.supply_interruption / assumptions.base_demand)
            * (1 - np.exp(-day_index / 7))
            * np.exp(-BLOCKADE_RISK_DECAY * day_index)
        )
        uncertainty_premium = base_price * behavior.uncertainty_floor * (1 - np.exp(-day_index / 18))
        panic_premium = base_price * 0.45 * fear_excess
        buffer_discount = buffer_confirmation_discount(day_index, gap_closure_day, base_price, behavior)
        relief_discount = expectation_relief_discount(day_index, base_price, behavior)
        target_price = (
            base_price
            + shortage_pressure
            + blockade_risk_premium
            + uncertainty_premium
            + panic_premium
            - buffer_discount
            - relief_discount
        )

        simulated_price = previous_price + behavior.adjustment_speed * (target_price - previous_price)
        simulated_price += 2.5 * (fear_excess - previous_fear_excess)
        simulated_price = float(np.clip(simulated_price, base_price * 0.75, 180.0))

        rows.append(
            {
                "day_index": day_index,
                "step_index": step_index,
                "time_axis": time_axis,
                "trade_date": trade_date,
                "actual_price": float(actual_row.close_price),
                "simulated_price": simulated_price,
                "effective_supply": effective_supply,
                "effective_demand": effective_demand,
                "supply_gap": residual_gap,
                "spr_release": spr_release,
                "route_supply": route_supply,
                "inventory_buffer": inventory_buffer,
                "inventory_remaining": inventory_remaining,
                "demand_decline": demand_decline,
                "demand_elasticity": elasticity,
                "fear_factor": 1 + fear_excess,
                "shortage_pressure": shortage_pressure,
                "blockade_risk_premium": blockade_risk_premium,
                "uncertainty_premium": uncertainty_premium,
                "panic_premium": panic_premium,
                "buffer_confirmation_discount": buffer_discount,
                "expectation_relief_discount": relief_discount,
            }
        )
        previous_price = simulated_price
        previous_fear_excess = fear_excess

    return pd.DataFrame(rows)


def compute_metrics(simulated: pd.DataFrame, behavior: BehavioralParameters) -> dict[str, float]:
    error = simulated["simulated_price"] - simulated["actual_price"]
    rmse = calc_rmse(error)
    mae = calc_mae(error)
    peak_price_error = float(simulated["simulated_price"].max() - simulated["actual_price"].max())
    final_price_error = float(simulated["simulated_price"].iloc[-1] - simulated["actual_price"].iloc[-1])
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "peak_price_error": peak_price_error,
        "final_price_error": final_price_error,
        "simulated_peak_price": float(simulated["simulated_price"].max()),
        "actual_peak_price": float(simulated["actual_price"].max()),
        "simulated_final_price": float(simulated["simulated_price"].iloc[-1]),
        "actual_final_price": float(simulated["actual_price"].iloc[-1]),
        "pressure_scale": behavior.pressure_scale,
        "risk_weight": behavior.risk_weight,
        "uncertainty_floor": behavior.uncertainty_floor,
        "inventory_response": behavior.inventory_response,
        "adjustment_speed": behavior.adjustment_speed,
        "buffer_relief_strength": behavior.buffer_relief_strength,
        "buffer_relief_decay_days": behavior.buffer_relief_decay_days,
        "relief_discount_strength": behavior.relief_discount_strength,
        "relief_start_day": behavior.relief_start_day,
        "relief_peak_day": behavior.relief_peak_day,
        "relief_decay_days": behavior.relief_decay_days,
    }
    metrics["selection_score"] = rmse + 0.20 * abs(peak_price_error) + 0.35 * abs(final_price_error)
    return metrics


def candidate_behaviors() -> list[BehavioralParameters]:
    candidates: list[BehavioralParameters] = []
    for pressure_scale, risk_weight, uncertainty_floor, inventory_response, adjustment_speed in product(
        [0.045, 0.060, 0.075, 0.090],
        [2.2, 2.6, 3.0, 3.4],
        [0.08, 0.12, 0.16, 0.20, 0.24, 0.28],
        [0.25, 0.40, 0.55],
        [0.25, 0.35, 0.45],
    ):
        candidates.append(
            BehavioralParameters(
                pressure_scale=pressure_scale,
                risk_weight=risk_weight,
                uncertainty_floor=uncertainty_floor,
                inventory_response=inventory_response,
                adjustment_speed=adjustment_speed,
            )
        )
    return candidates


def select_initial_behavior(
    event_df: pd.DataFrame,
    assumptions: PhysicalAssumptions,
) -> tuple[BehavioralParameters, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, float]] = []
    best_behavior: BehavioralParameters | None = None
    best_simulation: pd.DataFrame | None = None
    best_score = float("inf")

    for behavior in candidate_behaviors():
        simulation = simulate_dynamic_model(event_df, assumptions, behavior)
        metrics = compute_metrics(simulation, behavior)
        metric_rows.append(metrics)
        if metrics["selection_score"] < best_score:
            best_score = metrics["selection_score"]
            best_behavior = behavior
            best_simulation = simulation

    if best_behavior is None or best_simulation is None:
        raise RuntimeError("No dynamic-model behavior candidate was evaluated.")

    metrics_df = pd.DataFrame(metric_rows).sort_values(["selection_score", "rmse", "mae"]).reset_index(drop=True)
    return best_behavior, best_simulation, metrics_df


def configure_plot_style() -> None:
    apply_plot_style(savefig_dpi=180, figure_dpi=150, title_size=13)


def save_outputs(simulation: pd.DataFrame, metrics_df: pd.DataFrame, paths: DynamicPaths) -> None:
    ensure_parent(paths.result_csv)
    ensure_parent(paths.metrics_csv)
    simulation.to_csv(paths.result_csv, index=False)
    metrics_df.head(20).to_csv(paths.metrics_csv, index=False)


def save_figure(simulation: pd.DataFrame, paths: DynamicPaths) -> None:
    ensure_parent(paths.figure_path)
    configure_plot_style()

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
        label="短期动态递推模型",
    )
    ax.axhspan(110, 120, color="#10b981", alpha=0.10, label="题面110-120区间")
    ax.set_title("短期动态递推模型与实际价格对比")
    ax.set_xlabel("日期")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(paths.figure_path, dpi=180)
    plt.close(fig)


def build_report(
    event_df: pd.DataFrame,
    simulation: pd.DataFrame,
    metrics_df: pd.DataFrame,
    assumptions: PhysicalAssumptions,
    behavior: BehavioralParameters,
    paths: DynamicPaths,
) -> str:
    best = metrics_df.iloc[0]
    event_start = event_df["trade_date"].min().date()
    event_end = event_df["trade_date"].max().date()
    coverage_days = int((event_df["trade_date"].max() - event_df["trade_date"].min()).days)
    average_components = simulation[
        ["spr_release", "route_supply", "inventory_buffer", "demand_decline", "supply_gap", "fear_factor"]
    ].mean()

    return f"""# 短期动态递推模型报告

## 运行结论

阶段 3 已建立短期动态递推模型。模型纳入战略储备释放、商业库存缓冲、绕道运输恢复、需求收缩和恐慌衰减五类机制，并使用附件 CSV 的冲突窗口价格进行初步对照。

当前最优初始参数组合的 RMSE 为 {best["rmse"]:.2f} USD/barrel，MAE 为 {best["mae"]:.2f} USD/barrel。模拟峰值为 {best["simulated_peak_price"]:.2f} USD/barrel，实际最高收盘价为 {best["actual_peak_price"]:.2f} USD/barrel。

本阶段采用平衡选择分数：

```text
selection_score = RMSE + 0.20 * abs(峰值误差) + 0.35 * abs(末日误差)
```

这样做是为了同时保证整体路径、峰值和末日平台价格都不过度偏离。完整参数校准留到阶段 4 继续扩大搜索。

## 输入数据

- 冲突窗口价格：`{paths.event_csv.relative_to(PROJECT_ROOT)}`
- 题面参数表：`{PROBLEM_PARAMETERS_PATH.relative_to(PROJECT_ROOT)}`
- 实际交易窗口：{event_start} 至 {event_end}
- 日历跨度：{coverage_days} 天
- 样本交易日数：{len(event_df)}

## 机制设定

| 机制 | 当前设定 |
|---|---:|
| 战前供给 | {assumptions.base_supply:.0f} 万桶/日 |
| 战前需求 | {assumptions.base_demand:.0f} 万桶/日 |
| 供应中断 | {assumptions.supply_interruption:.0f} 万桶/日 |
| SPR 释放上限 | {assumptions.spr_max_release:.0f} 万桶/日 |
| SPR 启动延迟 | {assumptions.spr_delay_days} 天 |
| 绕道运输上限 | {assumptions.route_max_capacity:.0f} 万桶/日 |
| 绕道启动日 | 第 {assumptions.route_start_day} 天 |
| 商业库存 | {assumptions.commercial_inventory:.0f} 万桶 |
| 需求下降上限 | {assumptions.observed_demand_decline:.0f} 万桶/日 |
| 恐慌初始强度 | {assumptions.fear_initial:.2f} |
| 恐慌衰减速度 | {assumptions.fear_decay:.2f} |

## 初始行为参数

| 参数 | 数值 |
|---|---:|
| pressure_scale | {behavior.pressure_scale:.3f} |
| risk_weight | {behavior.risk_weight:.3f} |
| uncertainty_floor | {behavior.uncertainty_floor:.3f} |
| inventory_response | {behavior.inventory_response:.3f} |
| adjustment_speed | {behavior.adjustment_speed:.3f} |

## 平均机制贡献

| 变量 | 平均值 |
|---|---:|
| SPR 释放 | {average_components["spr_release"]:.2f} 万桶/日 |
| 绕道运输 | {average_components["route_supply"]:.2f} 万桶/日 |
| 库存缓冲 | {average_components["inventory_buffer"]:.2f} 万桶/日 |
| 需求收缩 | {average_components["demand_decline"]:.2f} 万桶/日 |
| 剩余供需缺口 | {average_components["supply_gap"]:.2f} 万桶/日 |
| 恐慌因子 | {average_components["fear_factor"]:.3f} |

## 误差指标

| 指标 | 数值 |
|---|---:|
| RMSE | {best["rmse"]:.2f} |
| MAE | {best["mae"]:.2f} |
| 峰值误差 | {best["peak_price_error"]:.2f} |
| 末日误差 | {best["final_price_error"]:.2f} |
| 平衡选择分数 | {best["selection_score"]:.2f} |

## 输出产物

- `{paths.result_csv.relative_to(PROJECT_ROOT)}`
- `{paths.metrics_csv.relative_to(PROJECT_ROOT)}`
- `{paths.figure_path.relative_to(PROJECT_ROOT)}`

## 后续作用

- 阶段 3 已经把传统供需基准模型升级为含缓冲机制的动态解释模型。
- 阶段 4 可以在此基础上扩大参数搜索范围，系统校准 `pressure_scale`、`risk_weight`、`inventory_response` 等行为参数。
- 阶段 5 的 60-180 天预测可以复用这里的机制函数，只需切换三情景参数和预测长度。
"""


def write_report(
    event_df: pd.DataFrame,
    simulation: pd.DataFrame,
    metrics_df: pd.DataFrame,
    assumptions: PhysicalAssumptions,
    behavior: BehavioralParameters,
    paths: DynamicPaths,
) -> None:
    ensure_parent(paths.report_path)
    report = build_report(event_df, simulation, metrics_df, assumptions, behavior, paths)
    paths.report_path.write_text(report, encoding="utf-8")


def main() -> None:
    base_config = load_yaml(BASE_CONFIG_PATH)
    scenario_config = load_yaml(SCENARIO_CONFIG_PATH)
    problem_params = load_problem_parameters(PROBLEM_PARAMETERS_PATH)
    paths = resolve_paths(base_config)
    assumptions = resolve_assumptions(base_config, scenario_config, problem_params)
    event_df = load_event_window(paths.event_csv)

    behavior, simulation, metrics_df = select_initial_behavior(event_df, assumptions)
    save_outputs(simulation, metrics_df, paths)
    save_figure(simulation, paths)
    write_report(event_df, simulation, metrics_df, assumptions, behavior, paths)

    print("Stage 3 complete")
    print(f"Results: {paths.result_csv.relative_to(PROJECT_ROOT)}")
    print(f"Metrics: {paths.metrics_csv.relative_to(PROJECT_ROOT)}")
    print(f"Figure: {paths.figure_path.relative_to(PROJECT_ROOT)}")
    print(f"Report: {paths.report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
