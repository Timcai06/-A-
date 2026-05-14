"""Lag and smoothing experiments for the short-term mechanism model.

This is an engineering experiment only.  It does not update the official
short-term model artifacts or the paper.  The aim is to test whether delayed
and smoothed buffer/expectation discounts improve the weakest event windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.short_term_model_quality_audit import load_best_parameters
from src.common.metrics import mae, rmse
from src.common.paths import PROJECT_ROOT, ensure_parents
from src.common.plotting import PAPER_COLORS, SCENARIO_COLORS, configure_plot_style
from src.models import dynamic_short_term as dynamic


@dataclass(frozen=True)
class ExperimentPaths:
    summary_csv: Path = PROJECT_ROOT / "output" / "experiments" / "短期模型滞后平滑实验汇总.csv"
    best_path_csv: Path = PROJECT_ROOT / "output" / "experiments" / "短期模型滞后平滑实验最佳路径.csv"
    report_path: Path = PROJECT_ROOT / "output" / "reports" / "短期模型滞后平滑实验报告.md"
    figure_path: Path = PROJECT_ROOT / "output" / "candidate_figures" / "短期模型滞后平滑实验_最佳候选.png"


@dataclass(frozen=True)
class Variant:
    name: str
    buffer_lag_days: int = 0
    relief_lag_days: int = 0
    buffer_alpha: float = 1.0
    relief_alpha: float = 1.0
    risk_alpha: float = 1.0


EVENT_WINDOWS = [
    ("初始冲击吸收", "2026-03-02", "2026-03-14"),
    ("高位平台形成", "2026-03-17", "2026-03-31"),
    ("中期再定价回落", "2026-04-01", "2026-04-17"),
    ("后期二次抬升", "2026-04-20", "2026-05-05"),
]


def load_event_window() -> pd.DataFrame:
    base_config = dynamic.load_yaml(dynamic.BASE_CONFIG_PATH)
    paths = dynamic.resolve_paths(base_config)
    return dynamic.load_event_window(paths.event_csv)


def lagged(values: list[float], lag_days: int) -> float:
    if lag_days <= 0:
        return values[-1]
    if len(values) <= lag_days:
        return 0.0
    return values[-1 - lag_days]


def smooth(previous: float, current: float, alpha: float) -> float:
    bounded = float(np.clip(alpha, 0.0, 1.0))
    return bounded * current + (1 - bounded) * previous


def simulate_variant(
    event_df: pd.DataFrame,
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
    variant: Variant,
    time_axis: str = "calendar",
) -> pd.DataFrame:
    first_date = event_df["trade_date"].min()
    base_price = float(event_df.iloc[0]["pre_close"])
    previous_price = base_price
    previous_fear_excess = assumptions.fear_initial
    previous_buffer_coverage_ratio = 0.0
    inventory_remaining = assumptions.commercial_inventory
    gap_closure_day: int | None = None
    raw_buffer_values: list[float] = []
    raw_relief_values: list[float] = []
    smoothed_buffer_discount = 0.0
    smoothed_relief_discount = 0.0
    smoothed_risk_premium = 0.0
    rows: list[dict[str, object]] = []

    for step_index, actual_row in enumerate(event_df.itertuples(index=False)):
        trade_date = actual_row.trade_date
        if time_axis == "calendar":
            day_index = int((trade_date - first_date).days)
        elif time_axis == "trading_day":
            day_index = step_index
        else:
            raise ValueError(f"Unsupported time_axis: {time_axis}")

        elasticity = dynamic.interpolate_elasticity(day_index, assumptions)
        price_ratio = max(previous_price / base_price, 0.1)
        price_adjusted_demand = assumptions.base_demand * (price_ratio**elasticity)
        demand_decline = dynamic.ramp(day_index, 0, assumptions.demand_decline_ramp_days, assumptions.observed_demand_decline)
        effective_demand = max(price_adjusted_demand - demand_decline, assumptions.base_demand * 0.70)

        spr_release = dynamic.ramp(day_index, assumptions.spr_delay_days, assumptions.spr_ramp_days, assumptions.spr_max_release)
        route_supply = dynamic.ramp(day_index, assumptions.route_start_day, assumptions.route_ramp_days, assumptions.route_max_capacity)
        gross_shortage = max(effective_demand - (assumptions.base_supply - assumptions.supply_interruption), 0.0)
        supply_without_inventory = assumptions.base_supply - assumptions.supply_interruption + spr_release + route_supply
        raw_gap = max(effective_demand - supply_without_inventory, 0.0)
        inventory_buffer = min(raw_gap * behavior.inventory_response, assumptions.inventory_daily_cap, inventory_remaining)
        inventory_remaining -= inventory_buffer
        effective_supply = supply_without_inventory + inventory_buffer
        residual_gap = max(effective_demand - effective_supply, 0.0)
        total_buffer_supply = spr_release + route_supply + inventory_buffer
        buffer_coverage_ratio = float(np.clip(total_buffer_supply / max(gross_shortage, 1.0), 0.0, 1.5))
        coverage_momentum = buffer_coverage_ratio - previous_buffer_coverage_ratio
        if gap_closure_day is None and residual_gap <= assumptions.base_demand * 0.005:
            gap_closure_day = day_index

        fear_excess = assumptions.fear_initial * np.exp(-assumptions.fear_decay * day_index)
        shortage_pressure = (
            base_price
            * behavior.pressure_scale
            * (residual_gap / assumptions.base_demand)
            / max(abs(elasticity), 0.01)
        )
        raw_risk_premium = (
            base_price
            * behavior.risk_weight
            * (assumptions.supply_interruption / assumptions.base_demand)
            * (1 - np.exp(-day_index / 7))
            * np.exp(-dynamic.BLOCKADE_RISK_DECAY * day_index)
        )
        smoothed_risk_premium = smooth(smoothed_risk_premium, raw_risk_premium, variant.risk_alpha)
        uncertainty_premium = base_price * behavior.uncertainty_floor * (1 - np.exp(-day_index / 18))
        panic_premium = base_price * 0.45 * fear_excess

        raw_buffer_discount = dynamic.buffer_confirmation_discount(day_index, gap_closure_day, base_price, behavior, buffer_coverage_ratio)
        raw_relief_discount = dynamic.expectation_relief_discount(
            day_index,
            base_price,
            behavior,
            buffer_coverage_ratio,
            coverage_momentum,
        )
        raw_buffer_values.append(raw_buffer_discount)
        raw_relief_values.append(raw_relief_discount)
        lagged_buffer_discount = lagged(raw_buffer_values, variant.buffer_lag_days)
        lagged_relief_discount = lagged(raw_relief_values, variant.relief_lag_days)
        smoothed_buffer_discount = smooth(smoothed_buffer_discount, lagged_buffer_discount, variant.buffer_alpha)
        smoothed_relief_discount = smooth(smoothed_relief_discount, lagged_relief_discount, variant.relief_alpha)

        target_price = (
            base_price
            + shortage_pressure
            + smoothed_risk_premium
            + uncertainty_premium
            + panic_premium
            - smoothed_buffer_discount
            - smoothed_relief_discount
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
                "gross_shortage": gross_shortage,
                "supply_gap": residual_gap,
                "spr_release": spr_release,
                "route_supply": route_supply,
                "inventory_buffer": inventory_buffer,
                "total_buffer_supply": total_buffer_supply,
                "buffer_coverage_ratio": buffer_coverage_ratio,
                "buffer_coverage_momentum": coverage_momentum,
                "inventory_remaining": inventory_remaining,
                "demand_decline": demand_decline,
                "demand_elasticity": elasticity,
                "fear_factor": 1 + fear_excess,
                "shortage_pressure": shortage_pressure,
                "blockade_risk_premium": smoothed_risk_premium,
                "raw_blockade_risk_premium": raw_risk_premium,
                "uncertainty_premium": uncertainty_premium,
                "panic_premium": panic_premium,
                "buffer_confirmation_discount": smoothed_buffer_discount,
                "raw_buffer_confirmation_discount": raw_buffer_discount,
                "expectation_relief_discount": smoothed_relief_discount,
                "raw_expectation_relief_discount": raw_relief_discount,
                "variant": variant.name,
            }
        )
        previous_price = simulated_price
        previous_fear_excess = fear_excess
        previous_buffer_coverage_ratio = buffer_coverage_ratio

    return pd.DataFrame(rows)


def event_segment_metrics(simulation: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, start, end in EVENT_WINDOWS:
        sub = simulation[simulation["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        error = sub["simulated_price"] - sub["actual_price"]
        out[f"{name}_RMSE"] = rmse(error)
        out[f"{name}_MAE"] = mae(error)
    return out


def evaluate_variant(simulation: pd.DataFrame, variant: Variant) -> dict[str, object]:
    error = simulation["simulated_price"] - simulation["actual_price"]
    segment = event_segment_metrics(simulation)
    return {
        "方案": variant.name,
        "buffer_lag_days": variant.buffer_lag_days,
        "relief_lag_days": variant.relief_lag_days,
        "buffer_alpha": variant.buffer_alpha,
        "relief_alpha": variant.relief_alpha,
        "risk_alpha": variant.risk_alpha,
        "RMSE": rmse(error),
        "MAE": mae(error),
        "平均误差": float(error.mean()),
        "最大绝对误差": float(error.abs().max()),
        "峰值误差": float(simulation["simulated_price"].max() - simulation["actual_price"].max()),
        "末日误差": float(simulation["simulated_price"].iloc[-1] - simulation["actual_price"].iloc[-1]),
        **segment,
    }


def build_variants() -> list[Variant]:
    variants = [Variant(name="当前机制主模型")]
    for buffer_lag in [0, 1, 2, 3]:
        for relief_lag in [0, 1, 2, 3, 4]:
            for buffer_alpha in [1.0, 0.75, 0.55]:
                for relief_alpha in [1.0, 0.75, 0.55, 0.40]:
                    if buffer_lag == 0 and relief_lag == 0 and buffer_alpha == 1.0 and relief_alpha == 1.0:
                        continue
                    variants.append(
                        Variant(
                            name=f"buffer_lag{buffer_lag}_relief_lag{relief_lag}_ba{buffer_alpha}_ra{relief_alpha}",
                            buffer_lag_days=buffer_lag,
                            relief_lag_days=relief_lag,
                            buffer_alpha=buffer_alpha,
                            relief_alpha=relief_alpha,
                        )
                    )
    for risk_alpha in [0.85, 0.70]:
        variants.append(Variant(name=f"risk_smooth_{risk_alpha}", risk_alpha=risk_alpha))
    return variants


def objective(row: pd.Series) -> float:
    return float(
        row["RMSE"]
        + 0.20 * row["中期再定价回落_RMSE"]
        + 0.08 * row["高位平台形成_RMSE"]
        + 0.05 * abs(row["末日误差"])
    )


def select_best(summary: pd.DataFrame) -> pd.Series:
    scored = summary.copy()
    baseline = scored[scored["方案"] == "当前机制主模型"].iloc[0]
    scored["优化目标"] = scored.apply(objective, axis=1)
    eligible = scored[
        (scored["RMSE"] <= baseline["RMSE"] + 0.03)
        & (scored["最大绝对误差"] <= baseline["最大绝对误差"] + 0.25)
        & (scored["中期再定价回落_RMSE"] < baseline["中期再定价回落_RMSE"])
    ].copy()
    if eligible.empty:
        eligible = scored.copy()
    return eligible.sort_values(["优化目标", "RMSE"]).iloc[0]


def draw_figure(baseline: pd.DataFrame, best: pd.DataFrame, best_row: pd.Series) -> None:
    configure_plot_style(savefig_dpi=260, figure_dpi=150, title_size=13)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})
    ax = axes[0]
    ax.plot(baseline["trade_date"], baseline["actual_price"], color=SCENARIO_COLORS["actual"], lw=2.2, label="实际收盘价")
    ax.plot(baseline["trade_date"], baseline["simulated_price"], color=SCENARIO_COLORS["fit"], lw=1.8, label="当前机制主模型")
    ax.plot(best["trade_date"], best["simulated_price"], color=SCENARIO_COLORS["buffer"], lw=2.0, linestyle="--", label="滞后平滑候选")
    ax.axhspan(110, 120, color=SCENARIO_COLORS["band_outer"], alpha=0.45, label="110-120美元平台")
    ax.set_title(f"短期滞后平滑候选：{best_row['方案']}")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="lower right", ncol=2)

    ax = axes[1]
    baseline_error = baseline["simulated_price"] - baseline["actual_price"]
    best_error = best["simulated_price"] - best["actual_price"]
    ax.plot(baseline["trade_date"], baseline_error, color=SCENARIO_COLORS["fit"], lw=1.6, label="当前机制误差")
    ax.plot(best["trade_date"], best_error, color=SCENARIO_COLORS["buffer"], lw=1.7, linestyle="--", label="候选误差")
    ax.axhline(0, color=PAPER_COLORS["ink"], lw=0.9)
    ax.set_title("逐日误差对比")
    ax.set_ylabel("美元/桶")
    ax.set_xlabel("日期")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ExperimentPaths.figure_path)
    plt.close(fig)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def build_report(summary: pd.DataFrame, best_row: pd.Series) -> str:
    baseline = summary[summary["方案"] == "当前机制主模型"].iloc[0]
    top = summary.sort_values("优化目标").head(8)
    top_rows = "\n".join(
        "| {name} | {rmse} | {mae} | {mid} | {platform} | {maxerr} | {objective_score} |".format(
            name=row["方案"],
            rmse=fmt(row["RMSE"]),
            mae=fmt(row["MAE"]),
            mid=fmt(row["中期再定价回落_RMSE"]),
            platform=fmt(row["高位平台形成_RMSE"]),
            maxerr=fmt(row["最大绝对误差"]),
            objective_score=fmt(row["优化目标"]),
        )
        for row in top.to_dict("records")
    )
    rmse_delta = best_row["RMSE"] - baseline["RMSE"]
    mid_gain = (baseline["中期再定价回落_RMSE"] - best_row["中期再定价回落_RMSE"]) / baseline["中期再定价回落_RMSE"] * 100
    full_gain = (baseline["RMSE"] - best_row["RMSE"]) / baseline["RMSE"] * 100
    return f"""# 短期模型滞后平滑实验报告

> 本报告只服务于模型优化决策，不写入论文正文，不覆盖当前官方短期模型输出。

## 结论

本轮测试了缓冲确认折价、预期修复折价的 0-4 天滞后和指数平滑组合。当前最优候选为 `{best_row['方案']}`。

- 当前机制主模型 RMSE：{baseline['RMSE']:.3f}
- 候选方案 RMSE：{best_row['RMSE']:.3f}，相对变化 {rmse_delta:+.3f}，相对改善 {full_gain:.2f}%
- 当前中期再定价回落 RMSE：{baseline['中期再定价回落_RMSE']:.3f}
- 候选中期再定价回落 RMSE：{best_row['中期再定价回落_RMSE']:.3f}，改善 {mid_gain:.2f}%
- 候选参数：buffer_lag={int(best_row['buffer_lag_days'])}，relief_lag={int(best_row['relief_lag_days'])}，buffer_alpha={best_row['buffer_alpha']:.2f}，relief_alpha={best_row['relief_alpha']:.2f}

解释上，这个方向是合理的：缓冲和预期修复不是当天完全进入价格，而是经过市场确认后逐步反映。若候选只改善中期回落、但显著破坏全窗口 RMSE 或最大误差，则不应纳入主模型。

## 排名前 8 的候选

| 方案 | RMSE | MAE | 中期RMSE | 高位平台RMSE | 最大绝对误差 | 优化目标 |
|---|---:|---:|---:|---:|---:|---:|
{top_rows}

## 下一步判断

1. 如果只追求拟合，候选方案可以继续进入局部参数校准。
2. 如果追求论文主模型的稳健解释，只有当候选同时改善全窗口 RMSE、中期 RMSE 和最大误差时，才值得替换当前机制主模型。
3. 本轮实验没有引入新外部数据，也没有增加新的自由机制变量，只是在现有折价项上加入更现实的滞后和平滑。

## 输出文件

- `{ExperimentPaths.summary_csv.relative_to(PROJECT_ROOT)}`
- `{ExperimentPaths.best_path_csv.relative_to(PROJECT_ROOT)}`
- `{ExperimentPaths.figure_path.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    ensure_parents([ExperimentPaths.summary_csv, ExperimentPaths.best_path_csv, ExperimentPaths.report_path, ExperimentPaths.figure_path])
    event_df = load_event_window()
    assumptions, behavior = load_best_parameters()
    variants = build_variants()
    rows: list[dict[str, object]] = []
    simulations: dict[str, pd.DataFrame] = {}
    for variant in variants:
        simulation = simulate_variant(event_df, assumptions, behavior, variant)
        rows.append(evaluate_variant(simulation, variant))
        simulations[variant.name] = simulation

    summary = pd.DataFrame(rows)
    summary["优化目标"] = summary.apply(objective, axis=1)
    summary = summary.sort_values(["优化目标", "RMSE"]).reset_index(drop=True)
    best_row = select_best(summary)
    baseline = simulations["当前机制主模型"]
    best = simulations[str(best_row["方案"])]
    best.to_csv(ExperimentPaths.best_path_csv, index=False)
    summary.to_csv(ExperimentPaths.summary_csv, index=False)
    draw_figure(baseline, best, best_row)
    ExperimentPaths.report_path.write_text(build_report(summary, best_row), encoding="utf-8")
    print(ExperimentPaths.report_path.relative_to(PROJECT_ROOT))
    print(ExperimentPaths.summary_csv.relative_to(PROJECT_ROOT))
    print(ExperimentPaths.figure_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
