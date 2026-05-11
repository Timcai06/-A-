"""Stage 5 scenario forecast for 60-180 day oil-price paths.

The forecast reuses the calibrated Stage 4 dynamic model. Observed attachment
prices are used up to the available event-window end date; later dates are
scenario extrapolations, not fabricated observations.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parent
from src.models import dynamic_short_term as dynamic


BEST_PARAMETERS_PATH = PROJECT_ROOT / "output" / "calibration" / "动态模型最优参数.csv"
CALIBRATED_PATH = PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv"
SCENARIO_RESULT_CSV = PROJECT_ROOT / "output" / "scenarios" / "三情景预测结果.csv"
SCENARIO_METRICS_CSV = PROJECT_ROOT / "output" / "scenarios" / "三情景关键指标.csv"
SCENARIO_PARAMS_CSV = PROJECT_ROOT / "output" / "scenarios" / "三情景参数表.csv"
SCENARIO_PRICE_FIGURE = PROJECT_ROOT / "figures" / "scenario_price_paths.png"
BUFFER_FIGURE = PROJECT_ROOT / "figures" / "inventory_depletion_risk.png"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "stage5_scenario_forecast_report.md"
FORECAST_END_DAY = 180
MARKER_DAYS = [60, 90, 120, 180]

SCENARIO_NAMES = {
    "optimistic": "乐观情景",
    "neutral": "中性情景",
    "pessimistic": "悲观情景",
}


def load_best_row() -> pd.Series:
    if not BEST_PARAMETERS_PATH.exists():
        raise FileNotFoundError(f"Missing Stage 4 best-parameter file: {BEST_PARAMETERS_PATH}")
    best = pd.read_csv(BEST_PARAMETERS_PATH)
    composite = best[best["候选类型"].astype(str).str.contains("综合最优")]
    if not composite.empty:
        return composite.iloc[0]
    return best.sort_values(["综合得分", "RMSE"]).iloc[0]


def calibrated_assumptions_and_behavior(best: pd.Series) -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]:
    assumptions = dynamic.PhysicalAssumptions(
        base_supply=float(best["assumption_base_supply"]),
        base_demand=float(best["assumption_base_demand"]),
        base_elasticity=float(best["assumption_base_elasticity"]),
        long_elasticity=float(best["assumption_long_elasticity"]),
        supply_interruption=float(best["assumption_supply_interruption"]),
        spr_max_release=float(best["assumption_spr_max_release"]),
        spr_delay_days=int(best["assumption_spr_delay_days"]),
        spr_ramp_days=int(best["assumption_spr_ramp_days"]),
        route_start_day=int(best["assumption_route_start_day"]),
        route_max_capacity=float(best["assumption_route_max_capacity"]),
        route_ramp_days=int(best["assumption_route_ramp_days"]),
        commercial_inventory=float(best["assumption_commercial_inventory"]),
        inventory_daily_cap=float(best["assumption_inventory_daily_cap"]),
        observed_demand_decline=float(best["assumption_observed_demand_decline"]),
        demand_decline_ramp_days=int(best["assumption_demand_decline_ramp_days"]),
        fear_initial=float(best["assumption_fear_initial"]),
        fear_decay=float(best["assumption_fear_decay"]),
    )
    behavior = dynamic.BehavioralParameters(
        pressure_scale=float(best["behavior_pressure_scale"]),
        risk_weight=float(best["behavior_risk_weight"]),
        uncertainty_floor=float(best["behavior_uncertainty_floor"]),
        inventory_response=float(best["behavior_inventory_response"]),
        adjustment_speed=float(best["behavior_adjustment_speed"]),
        buffer_relief_strength=float(best["behavior_buffer_relief_strength"]),
        buffer_relief_decay_days=int(best["behavior_buffer_relief_decay_days"]),
        relief_discount_strength=float(best["behavior_relief_discount_strength"]),
        relief_start_day=int(best["behavior_relief_start_day"]),
        relief_peak_day=int(best["behavior_relief_peak_day"]),
        relief_decay_days=int(best["behavior_relief_decay_days"]),
    )
    return assumptions, behavior


def build_scenario_parameters(
    base_assumptions: dynamic.PhysicalAssumptions,
    base_behavior: dynamic.BehavioralParameters,
    scenario_config: dict[str, Any],
) -> dict[str, tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]]:
    scenarios = scenario_config["scenarios"]

    optimistic_cfg = scenarios["optimistic"]
    neutral_cfg = scenarios["neutral"]
    pessimistic_cfg = scenarios["pessimistic"]

    optimistic = (
        replace(
            base_assumptions,
            supply_interruption=float(optimistic_cfg["supply_interruption"]),
            spr_max_release=float(optimistic_cfg["spr_max_release"]),
            spr_delay_days=int(optimistic_cfg["spr_delay_days"]),
            spr_ramp_days=6,
            route_start_day=int(optimistic_cfg["route_start_day"]),
            route_max_capacity=float(optimistic_cfg["route_max_capacity"]),
            route_ramp_days=18,
            long_elasticity=float(optimistic_cfg["demand_elasticity_long_term"]),
            inventory_daily_cap=max(base_assumptions.inventory_daily_cap, 350.0),
            observed_demand_decline=520.0,
            demand_decline_ramp_days=18,
            fear_decay=float(optimistic_cfg["fear_decay"]),
        ),
        replace(
            base_behavior,
            pressure_scale=base_behavior.pressure_scale * 0.82,
            risk_weight=base_behavior.risk_weight * 0.82,
            uncertainty_floor=base_behavior.uncertainty_floor * 0.70,
            adjustment_speed=min(base_behavior.adjustment_speed * 1.08, 0.45),
            buffer_relief_strength=min(base_behavior.buffer_relief_strength * 1.15, 0.28),
            relief_discount_strength=min(base_behavior.relief_discount_strength * 1.05, 0.25),
            relief_start_day=22,
            relief_peak_day=42,
            relief_decay_days=20,
        ),
    )

    # The neutral scenario keeps the Stage 4 calibrated parameters as the
    # central path, only preserving the config description for documentation.
    neutral = (
        replace(
            base_assumptions,
            supply_interruption=float(base_assumptions.supply_interruption),
            spr_max_release=float(base_assumptions.spr_max_release),
            spr_delay_days=int(base_assumptions.spr_delay_days),
            route_start_day=int(base_assumptions.route_start_day),
            route_max_capacity=float(base_assumptions.route_max_capacity),
            long_elasticity=float(base_assumptions.long_elasticity),
            fear_decay=float(base_assumptions.fear_decay),
        ),
        base_behavior,
    )

    pessimistic = (
        replace(
            base_assumptions,
            supply_interruption=float(pessimistic_cfg["supply_interruption"]),
            spr_max_release=float(pessimistic_cfg["spr_max_release"]),
            spr_delay_days=int(pessimistic_cfg["spr_delay_days"]),
            spr_ramp_days=12,
            route_start_day=int(pessimistic_cfg["route_start_day"]),
            route_max_capacity=float(pessimistic_cfg["route_max_capacity"]),
            route_ramp_days=42,
            long_elasticity=float(pessimistic_cfg["demand_elasticity_long_term"]),
            inventory_daily_cap=180.0,
            observed_demand_decline=300.0,
            demand_decline_ramp_days=45,
            fear_decay=float(pessimistic_cfg["fear_decay"]),
        ),
        replace(
            base_behavior,
            pressure_scale=base_behavior.pressure_scale * 1.35,
            risk_weight=base_behavior.risk_weight * 1.35,
            uncertainty_floor=min(base_behavior.uncertainty_floor * 1.35, 0.38),
            adjustment_speed=max(base_behavior.adjustment_speed * 0.92, 0.20),
            buffer_relief_strength=base_behavior.buffer_relief_strength * 0.35,
            relief_discount_strength=base_behavior.relief_discount_strength * 0.25,
            relief_start_day=65,
            relief_peak_day=110,
            relief_decay_days=45,
        ),
    )

    return {
        "optimistic": optimistic,
        "neutral": neutral,
        "pessimistic": pessimistic,
    }


def build_forecast_frame(event_df: pd.DataFrame) -> pd.DataFrame:
    event_start = event_df["trade_date"].min()
    last_day_index = int((event_df["trade_date"].max() - event_start).days)
    future_dates = pd.date_range(
        event_df["trade_date"].max() + pd.Timedelta(days=1),
        event_start + pd.Timedelta(days=FORECAST_END_DAY),
        freq="D",
    )
    future_rows = pd.DataFrame(
        {
            "trade_date": future_dates,
            "close_price": np.nan,
            "pre_close": np.nan,
        }
    )
    frame = pd.concat([event_df[["trade_date", "close_price", "pre_close"]], future_rows], ignore_index=True)
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    frame["阶段"] = np.where(
        frame["trade_date"] <= event_df["trade_date"].max(),
        "附件观测期",
        "情景外推期",
    )
    frame["day_index"] = (frame["trade_date"] - event_start).dt.days.astype(int)
    frame.loc[frame.index[0], "pre_close"] = float(event_df.iloc[0]["pre_close"])
    if int(frame["day_index"].max()) != FORECAST_END_DAY:
        raise RuntimeError("Forecast frame did not reach the configured end day.")
    if last_day_index >= FORECAST_END_DAY:
        raise RuntimeError("Observed event window already exceeds forecast horizon.")
    return frame


def load_calibrated_prefix() -> pd.DataFrame:
    if not CALIBRATED_PATH.exists():
        raise FileNotFoundError(f"Missing Stage 4 calibrated path: {CALIBRATED_PATH}")
    prefix = pd.read_csv(CALIBRATED_PATH, parse_dates=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    prefix["阶段"] = "附件观测期"
    prefix["is_observed_price"] = True
    prefix["forecast_price"] = prefix["simulated_price"]
    return prefix


def infer_gap_closure_day(prefix: pd.DataFrame, assumptions: dynamic.PhysicalAssumptions) -> int | None:
    closed = prefix[prefix["supply_gap"] <= assumptions.base_demand * 0.005]
    if closed.empty:
        return None
    return int(closed.iloc[0]["day_index"])


def simulate_future_from_prefix(
    prefix: pd.DataFrame,
    future_frame: pd.DataFrame,
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
    base_price: float,
) -> pd.DataFrame:
    previous_price = float(prefix.iloc[-1]["simulated_price"])
    previous_day = int(prefix.iloc[-1]["day_index"])
    previous_fear_excess = assumptions.fear_initial * np.exp(-assumptions.fear_decay * previous_day)
    inventory_remaining = float(prefix.iloc[-1]["inventory_remaining"])
    gap_closure_day = infer_gap_closure_day(prefix, assumptions)
    rows: list[dict[str, Any]] = []

    for _, row in future_frame.iterrows():
        trade_date = row["trade_date"]
        day_index = int(row["day_index"])
        elasticity = dynamic.interpolate_elasticity(day_index, assumptions)
        price_ratio = max(previous_price / base_price, 0.1)

        price_adjusted_demand = assumptions.base_demand * (price_ratio**elasticity)
        demand_decline = dynamic.ramp(
            day_index,
            0,
            assumptions.demand_decline_ramp_days,
            assumptions.observed_demand_decline,
        )
        effective_demand = max(price_adjusted_demand - demand_decline, assumptions.base_demand * 0.70)

        spr_release = dynamic.ramp(
            day_index,
            assumptions.spr_delay_days,
            assumptions.spr_ramp_days,
            assumptions.spr_max_release,
        )
        route_supply = dynamic.ramp(
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
            * np.exp(-0.004 * day_index)
        )
        uncertainty_premium = base_price * behavior.uncertainty_floor * (1 - np.exp(-day_index / 18))
        panic_premium = base_price * 0.45 * fear_excess
        buffer_discount = dynamic.buffer_confirmation_discount(day_index, gap_closure_day, base_price, behavior)
        relief_discount = dynamic.expectation_relief_discount(day_index, base_price, behavior)
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
                "trade_date": trade_date,
                "actual_price": np.nan,
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
                "阶段": "情景外推期",
                "is_observed_price": False,
                "forecast_price": simulated_price,
            }
        )
        previous_price = simulated_price
        previous_fear_excess = fear_excess

    return pd.DataFrame(rows)


def run_scenario(
    scenario_key: str,
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
    forecast_frame: pd.DataFrame,
    prefix: pd.DataFrame,
) -> pd.DataFrame:
    future_frame = forecast_frame[forecast_frame["阶段"] == "情景外推期"].copy()
    base_price = float(forecast_frame.iloc[0]["pre_close"])
    future = simulate_future_from_prefix(prefix, future_frame, assumptions, behavior, base_price)
    simulation = pd.concat([prefix, future], ignore_index=True)
    simulation["scenario"] = scenario_key
    simulation["情景"] = SCENARIO_NAMES[scenario_key]
    return simulation


def summarize_scenario(simulation: pd.DataFrame) -> dict[str, Any]:
    scenario_key = str(simulation.iloc[0]["scenario"])
    forecast_period = simulation[simulation["阶段"] == "情景外推期"]
    all_period = simulation.copy()
    marker_prices = {
        f"第{day}天价格": float(all_period.loc[all_period["day_index"] == day, "forecast_price"].iloc[0])
        for day in MARKER_DAYS
    }
    post_observed_peak = float(forecast_period["forecast_price"].max())
    post_observed_min = float(forecast_period["forecast_price"].min())
    final_price = float(all_period.loc[all_period["day_index"] == FORECAST_END_DAY, "forecast_price"].iloc[0])
    final_inventory = float(all_period.loc[all_period["day_index"] == FORECAST_END_DAY, "inventory_remaining"].iloc[0])
    final_gap = float(all_period.loc[all_period["day_index"] == FORECAST_END_DAY, "supply_gap"].iloc[0])
    max_gap = float(forecast_period["supply_gap"].max())
    cutoff_price = float(all_period[all_period["阶段"] == "附件观测期"]["forecast_price"].iloc[-1])
    second_jump = max(post_observed_peak - cutoff_price, 0.0)
    risk_level = "低"
    if final_gap > 500 or second_jump > 15:
        risk_level = "高"
    elif final_gap > 150 or second_jump > 8:
        risk_level = "中"

    return {
        "scenario": scenario_key,
        "情景": SCENARIO_NAMES[scenario_key],
        **marker_prices,
        "外推期最高价": post_observed_peak,
        "外推期最低价": post_observed_min,
        "第180天价格": final_price,
        "外推期均价": float(forecast_period["forecast_price"].mean()),
        "第180天剩余商业库存": final_inventory,
        "第180天剩余供需缺口": final_gap,
        "外推期最大供需缺口": max_gap,
        "二次跳涨幅度": float(second_jump),
        "二次跳涨风险": risk_level,
    }


def build_parameter_table(
    scenarios: dict[str, tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, (assumptions, behavior) in scenarios.items():
        rows.append(
            {
                "scenario": key,
                "情景": SCENARIO_NAMES[key],
                **{f"assumption_{name}": value for name, value in asdict(assumptions).items()},
                **{f"behavior_{name}": value for name, value in asdict(behavior).items()},
            }
        )
    return pd.DataFrame(rows)


def save_figures(result: pd.DataFrame) -> None:
    dynamic.configure_plot_style()
    ensure_parent(SCENARIO_PRICE_FIGURE)

    colors = {
        "optimistic": "#059669",
        "neutral": "#2563eb",
        "pessimistic": "#dc2626",
    }
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    observed = result[(result["scenario"] == "neutral") & (result["is_observed_price"])]
    observed_end = observed["trade_date"].max()
    ax.plot(
        observed["trade_date"],
        observed["actual_price"],
        color="#111827",
        linewidth=1.8,
        marker="o",
        markersize=2.5,
        label="附件真实收盘价",
    )
    for key, label in SCENARIO_NAMES.items():
        sub = result[result["scenario"] == key]
        plot_sub = sub[sub["trade_date"] >= observed_end]
        ax.plot(
            plot_sub["trade_date"],
            plot_sub["forecast_price"],
            color=colors[key],
            linewidth=2.0,
            label=label,
        )
    ax.axvline(observed_end, color="#6b7280", linestyle="--", linewidth=1.0, label="附件数据截止")
    ax.axhspan(110, 120, color="#f59e0b", alpha=0.10, label="110-120美元/桶参考区间")
    ax.set_title("阶段5：60-180天三情景原油价格路径")
    ax.set_xlabel("日期")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper left", ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(SCENARIO_PRICE_FIGURE, dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True)
    for key, label in SCENARIO_NAMES.items():
        sub = result[result["scenario"] == key]
        plot_sub = sub[sub["trade_date"] >= observed_end]
        axes[0].plot(plot_sub["trade_date"], plot_sub["inventory_remaining"], color=colors[key], linewidth=2.0, label=label)
        axes[1].plot(plot_sub["trade_date"], plot_sub["supply_gap"], color=colors[key], linewidth=2.0, label=label)
    axes[0].set_title("商业库存剩余量")
    axes[0].set_ylabel("万桶")
    axes[0].legend(loc="upper right")
    axes[1].set_title("剩余供需缺口")
    axes[1].set_xlabel("日期")
    axes[1].set_ylabel("万桶/日")
    for ax in axes:
        ax.axvline(observed_end, color="#6b7280", linestyle="--", linewidth=1.0)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(BUFFER_FIGURE, dpi=190)
    plt.close(fig)


def build_report(metrics: pd.DataFrame, params: pd.DataFrame, event_df: pd.DataFrame) -> str:
    event_start = event_df["trade_date"].min().date()
    observed_end = event_df["trade_date"].max().date()
    forecast_end = (event_df["trade_date"].min() + pd.Timedelta(days=FORECAST_END_DAY)).date()

    metric_rows = "\n".join(
        "| {情景} | {第60天价格:.2f} | {第90天价格:.2f} | {第120天价格:.2f} | {第180天价格:.2f} | {外推期最高价:.2f} | {外推期均价:.2f} | {二次跳涨风险} |".format(
            **row
        )
        for row in metrics.to_dict("records")
    )
    param_rows = "\n".join(
        "| {情景} | {assumption_supply_interruption:.0f} | {assumption_spr_max_release:.0f} | {assumption_route_max_capacity:.0f} | {assumption_long_elasticity:.2f} | {assumption_fear_decay:.2f} | {behavior_risk_weight:.2f} | {behavior_uncertainty_floor:.2f} |".format(
            **row
        )
        for row in params.to_dict("records")
    )

    return f"""# 阶段 5 三情景预测报告

## 运行结论

阶段 5 已基于阶段 4 综合最优短期动态模型，构建乐观、中性、悲观三条 60-180 天价格路径。预测口径为：从冲突窗口起点 {event_start} 推进到第 {FORECAST_END_DAY} 天，即 {forecast_end}；附件真实价格只覆盖到 {observed_end}，之后均为情景外推结果，不作为真实观测数据。

本阶段没有新增爬虫数据，也没有编造未来真实价格。模型继续使用附件 CSV 的历史价格作为校准基础，情景差异来自赛题范围内的供应中断、SPR 释放、绕道恢复、需求弹性、恐慌衰减和市场预期修复强度。

## 关键预测结果

| 情景 | 第60天 | 第90天 | 第120天 | 第180天 | 外推期最高价 | 外推期均价 | 二次跳涨风险 |
|---|---:|---:|---:|---:|---:|---:|---|
{metric_rows}

## 情景参数差异

| 情景 | 供应中断 | SPR释放上限 | 绕道能力 | 长期需求弹性 | 恐慌衰减 | 风险权重 | 不确定性平台 |
|---|---:|---:|---:|---:|---:|---:|---:|
{param_rows}

## 解释

- 乐观情景假设 SPR 释放更早更强、绕道运输恢复更快、需求收缩更充分，因此外推期价格逐步向低位收敛。
- 中性情景沿用阶段 4 的综合最优校准参数，是后续论文和敏感性分析的主基准路径。
- 悲观情景假设 SPR 释放弱、绕道运输慢、需求收缩不足且恐慌衰减较慢，因此价格维持更高平台，并存在更强的二次跳涨风险。

## 输出产物

- `{SCENARIO_RESULT_CSV.relative_to(PROJECT_ROOT)}`
- `{SCENARIO_METRICS_CSV.relative_to(PROJECT_ROOT)}`
- `{SCENARIO_PARAMS_CSV.relative_to(PROJECT_ROOT)}`
- `{SCENARIO_PRICE_FIGURE.relative_to(PROJECT_ROOT)}`
- `{BUFFER_FIGURE.relative_to(PROJECT_ROOT)}`

## 后续作用

阶段 5 已回答“封锁持续 60-180 天时油价路径如何演化”的预测问题。阶段 6 可以在此基础上围绕供应中断、SPR 释放、需求弹性、风险权重和预期修复强度做敏感性分析。
"""


def main() -> None:
    base_config = dynamic.load_yaml(dynamic.BASE_CONFIG_PATH)
    scenario_config = dynamic.load_yaml(dynamic.SCENARIO_CONFIG_PATH)
    paths = dynamic.resolve_paths(base_config)
    event_df = dynamic.load_event_window(paths.event_csv)
    best = load_best_row()
    base_assumptions, base_behavior = calibrated_assumptions_and_behavior(best)
    scenarios = build_scenario_parameters(base_assumptions, base_behavior, scenario_config)
    forecast_frame = build_forecast_frame(event_df)
    prefix = load_calibrated_prefix()

    simulations = [
        run_scenario(key, assumptions, behavior, forecast_frame, prefix)
        for key, (assumptions, behavior) in scenarios.items()
    ]
    result = pd.concat(simulations, ignore_index=True)
    metrics = pd.DataFrame(summarize_scenario(sim) for sim in simulations)
    params = build_parameter_table(scenarios)

    ensure_parent(SCENARIO_RESULT_CSV)
    ensure_parent(SCENARIO_METRICS_CSV)
    ensure_parent(SCENARIO_PARAMS_CSV)
    ensure_parent(REPORT_PATH)
    result.to_csv(SCENARIO_RESULT_CSV, index=False)
    metrics.to_csv(SCENARIO_METRICS_CSV, index=False)
    params.to_csv(SCENARIO_PARAMS_CSV, index=False)
    save_figures(result)
    REPORT_PATH.write_text(build_report(metrics, params, event_df), encoding="utf-8")

    print("Stage 5 complete")
    print(f"Scenario paths: {SCENARIO_RESULT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Scenario metrics: {SCENARIO_METRICS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Scenario params: {SCENARIO_PARAMS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Price figure: {SCENARIO_PRICE_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Buffer figure: {BUFFER_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
