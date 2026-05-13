"""Monte Carlo scenario tree for long-horizon tail-risk analysis."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parent
from src.analysis.historical_volatility_calibration import HistoricalModelFactors, load_historical_model_factors
from src.scenarios import scenario_forecast as scenario
from src.scenarios.external_constraints import (
    ExternalConstraintFactors,
    apply_external_constraints,
    load_external_constraint_factors,
)


RANDOM_SEED = 20260511
N_SAMPLES = 2000
OUTPUT_DIR = PROJECT_ROOT / "output" / "monte_carlo"
SAMPLE_METRICS_CSV = OUTPUT_DIR / "蒙特卡洛样本指标.csv"
QUANTILE_CSV = OUTPUT_DIR / "蒙特卡洛路径分位数.csv"
TAIL_RISK_CSV = OUTPUT_DIR / "蒙特卡洛尾部风险摘要.csv"
PATH_SAMPLE_CSV = OUTPUT_DIR / "蒙特卡洛路径云样本.csv"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "蒙特卡洛情景树报告.md"
FAN_FIGURE = PROJECT_ROOT / "figures" / "monte_carlo_price_fan.png"
TAIL_FIGURE = PROJECT_ROOT / "figures" / "monte_carlo_tail_risk.png"
ACADEMIC_PANEL_FIGURE = PROJECT_ROOT / "figures" / "蒙特卡洛情景树高级组合图.png"
PATH_CLOUD_FIGURE = PROJECT_ROOT / "figures" / "传统蒙特卡洛路径云图.png"
MARKER_DAYS = [90, 120, 150, 180]


def load_baseline() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    scenario.dynamic.PhysicalAssumptions,
    scenario.dynamic.BehavioralParameters,
]:
    base_config = scenario.dynamic.load_yaml(scenario.dynamic.BASE_CONFIG_PATH)
    scenario_config = scenario.dynamic.load_yaml(scenario.dynamic.SCENARIO_CONFIG_PATH)
    paths = scenario.dynamic.resolve_paths(base_config)
    event_df = scenario.dynamic.load_event_window(paths.event_csv)
    best = scenario.load_best_row()
    assumptions, behavior = scenario.calibrated_assumptions_and_behavior(best)
    assumptions, behavior = scenario.build_scenario_parameters(assumptions, behavior, scenario_config)["neutral"]
    forecast_frame = scenario.build_forecast_frame(event_df)
    prefix = scenario.load_calibrated_prefix()
    return event_df, forecast_frame, prefix, assumptions, behavior


def sample_parameters(
    rng: np.random.Generator,
    sample_id: int,
    base_assumptions: scenario.dynamic.PhysicalAssumptions,
    base_behavior: scenario.dynamic.BehavioralParameters,
    external_factors: ExternalConstraintFactors,
    historical_factors: HistoricalModelFactors,
) -> tuple[scenario.dynamic.PhysicalAssumptions, scenario.dynamic.BehavioralParameters, dict[str, Any]]:
    raw_stress = float(rng.beta(2.1, 2.3))
    stress = float(np.clip(raw_stress + rng.normal(0.0, historical_factors.monte_carlo_stress_noise_sigma), 0.0, 1.0))
    supply_interruption = float(np.clip(1400 + 400 * stress + rng.normal(0, 35), 1400, 1800))
    spr_max_release = float(np.clip(700 - 500 * (stress**1.08) + rng.normal(0, 30), 200, 700))
    route_max_capacity = float(np.clip(360 - 210 * stress + rng.normal(0, 22), 150, 420))
    long_elasticity = float(np.clip(-0.35 + 0.25 * stress + rng.normal(0, 0.018), -0.35, -0.10))
    fear_decay = float(np.clip(0.16 - 0.12 * stress + rng.normal(0, 0.01), 0.035, 0.17))
    observed_demand_decline = float(np.clip(540 - 250 * stress + rng.normal(0, 25), 260, 560))

    risk_weight = float(np.clip(base_behavior.risk_weight * (0.74 + 0.62 * stress + rng.normal(0, 0.06)), 1.65, 3.35))
    uncertainty_floor = float(
        np.clip(base_behavior.uncertainty_floor * (0.64 + 0.78 * stress + rng.normal(0, 0.06)), 0.13, 0.36)
    )
    pressure_scale = float(np.clip(base_behavior.pressure_scale * rng.lognormal(mean=0.0, sigma=0.12), 0.018, 0.045))
    relief_strength = float(np.clip(base_behavior.relief_discount_strength * (1.20 - 0.72 * stress), 0.04, 0.28))
    buffer_strength = float(np.clip(base_behavior.buffer_relief_strength * (1.25 - 0.72 * stress), 0.04, 0.30))

    assumptions = replace(
        base_assumptions,
        supply_interruption=supply_interruption,
        spr_max_release=spr_max_release,
        spr_delay_days=int(np.clip(round(1 + 13 * stress + rng.normal(0, 1.1)), 0, 14)),
        spr_ramp_days=int(np.clip(round(6 + 8 * stress), 5, 14)),
        route_start_day=int(np.clip(round(6 + 22 * stress + rng.normal(0, 1.5)), 5, 32)),
        route_max_capacity=route_max_capacity,
        route_ramp_days=int(np.clip(round(16 + 44 * stress + rng.normal(0, 2)), 14, 64)),
        long_elasticity=long_elasticity,
        observed_demand_decline=observed_demand_decline,
        demand_decline_ramp_days=int(np.clip(round(16 + 34 * stress), 14, 55)),
        fear_decay=fear_decay,
    )
    behavior = replace(
        base_behavior,
        pressure_scale=pressure_scale,
        risk_weight=risk_weight,
        uncertainty_floor=uncertainty_floor,
        adjustment_speed=float(np.clip(base_behavior.adjustment_speed * (1.06 - 0.18 * stress), 0.20, 0.44)),
        buffer_relief_strength=buffer_strength,
        relief_discount_strength=relief_strength,
        relief_start_day=int(np.clip(round(22 + 44 * stress), 20, 70)),
        relief_peak_day=int(np.clip(round(42 + 68 * stress), 38, 115)),
        relief_decay_days=int(np.clip(round(20 + 28 * stress), 16, 55)),
    )
    label = "缓和路径" if stress < 0.33 else "中性附近" if stress < 0.66 else "高压尾部"
    constraint_key = "optimistic" if stress < 0.33 else "neutral" if stress < 0.66 else "pessimistic"
    assumptions, behavior = apply_external_constraints(
        constraint_key,
        assumptions,
        behavior,
        external_factors,
    )
    meta = {
        "sample_id": sample_id,
        "stress_index": stress,
        "raw_stress_index": raw_stress,
        "historical_stress_noise_sigma": historical_factors.monte_carlo_stress_noise_sigma,
        "historical_event_volatility_percentile": historical_factors.event_volatility_percentile,
        "路径类别": label,
        "supply_interruption": supply_interruption,
        "spr_max_release": spr_max_release,
        "route_max_capacity": route_max_capacity,
        "long_elasticity": long_elasticity,
        "fear_decay": fear_decay,
        "risk_weight": risk_weight,
        "uncertainty_floor": uncertainty_floor,
        "pressure_scale": pressure_scale,
        "relief_discount_strength": relief_strength,
        "buffer_relief_strength": buffer_strength,
    }
    return assumptions, behavior, meta


def simulate_one_sample(
    forecast_frame: pd.DataFrame,
    prefix: pd.DataFrame,
    assumptions: scenario.dynamic.PhysicalAssumptions,
    behavior: scenario.dynamic.BehavioralParameters,
) -> pd.DataFrame:
    future_frame = forecast_frame[forecast_frame["阶段"] == "情景外推期"].copy()
    base_price = float(forecast_frame.iloc[0]["pre_close"])
    return scenario.simulate_future_from_prefix(prefix, future_frame, assumptions, behavior, base_price)


def risk_label(final_gap: float, second_jump: float) -> str:
    if final_gap > 500 or second_jump > 15:
        return "高"
    if final_gap > 150 or second_jump > 8:
        return "中"
    return "低"


def summarize_sample(path: pd.DataFrame, prefix: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    cutoff_price = float(prefix.iloc[-1]["forecast_price"])
    marker_prices = {
        f"第{day}天价格": float(path.loc[path["day_index"] == day, "forecast_price"].iloc[0])
        for day in MARKER_DAYS
    }
    peak_price = float(path["forecast_price"].max())
    final_row = path.loc[path["day_index"] == 180].iloc[0]
    second_jump = max(peak_price - cutoff_price, 0.0)
    final_gap = float(final_row["supply_gap"])
    row = {
        **meta,
        **marker_prices,
        "外推期最高价": peak_price,
        "外推期最低价": float(path["forecast_price"].min()),
        "外推期均价": float(path["forecast_price"].mean()),
        "第180天价格": float(final_row["forecast_price"]),
        "第180天剩余供需缺口": final_gap,
        "第180天SPR实际释放": float(final_row["spr_release"]),
        "第180天制度风险溢价": float(final_row["regime_risk_premium"]),
        "第180天冲击不确定性溢价": float(final_row["shock_uncertainty_premium"]),
        "二次跳涨幅度": float(second_jump),
    }
    row["二次跳涨风险"] = risk_label(final_gap, second_jump)
    row["是否突破120"] = peak_price >= 120
    row["是否突破130"] = peak_price >= 130
    row["是否高缺口"] = final_gap > 500
    return row


def build_quantiles(paths: list[pd.DataFrame]) -> pd.DataFrame:
    combined = build_path_frame(paths)
    quantiles = (
        combined.groupby(["day_index", "trade_date"])["forecast_price"]
        .quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        .unstack()
        .reset_index()
        .rename(
            columns={
                0.05: "p05",
                0.10: "p10",
                0.25: "p25",
                0.50: "p50",
                0.75: "p75",
                0.90: "p90",
                0.95: "p95",
            }
        )
    )
    return quantiles


def build_path_frame(paths: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(
        [
            path[["day_index", "trade_date", "forecast_price"]].assign(sample_id=i)
            for i, path in enumerate(paths, start=1)
        ],
        ignore_index=True,
    )


def build_tail_risk(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("外推期最高价突破120", float(metrics["是否突破120"].mean())),
        ("外推期最高价突破130", float(metrics["是否突破130"].mean())),
        ("第180天剩余缺口超过500", float(metrics["是否高缺口"].mean())),
        ("二次跳涨风险为高", float((metrics["二次跳涨风险"] == "高").mean())),
        ("第180天价格P05", float(metrics["第180天价格"].quantile(0.05))),
        ("第180天价格P50", float(metrics["第180天价格"].quantile(0.50))),
        ("第180天价格P95", float(metrics["第180天价格"].quantile(0.95))),
        ("外推期最高价P95", float(metrics["外推期最高价"].quantile(0.95))),
    ]
    return pd.DataFrame(rows, columns=["指标", "数值"])


def run_monte_carlo(n_samples: int = N_SAMPLES) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, forecast_frame, prefix, base_assumptions, base_behavior = load_baseline()
    external_factors = load_external_constraint_factors(write_output=False)
    historical_factors = load_historical_model_factors()
    rng = np.random.default_rng(RANDOM_SEED)
    paths: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    for sample_id in range(1, n_samples + 1):
        assumptions, behavior, meta = sample_parameters(
            rng,
            sample_id,
            base_assumptions,
            base_behavior,
            external_factors,
            historical_factors,
        )
        path = simulate_one_sample(forecast_frame, prefix, assumptions, behavior)
        paths.append(path)
        metric_rows.append(summarize_sample(path, prefix, meta))

    metrics = pd.DataFrame(metric_rows)
    quantiles = build_quantiles(paths)
    tail_risk = build_tail_risk(metrics)
    path_frame = build_path_frame(paths)
    return metrics, quantiles, tail_risk, path_frame


def _format_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.grid(True, axis="x", color="#f3f4f6", linewidth=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")


def _select_path_samples(metrics: pd.DataFrame, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    shares = {"缓和路径": 0.30, "中性附近": 0.45, "高压尾部": 0.25}
    for label, share in shares.items():
        ids = metrics.loc[metrics["路径类别"] == label, "sample_id"].to_numpy()
        if len(ids) == 0:
            continue
        take = min(len(ids), max(1, int(round(n * share))))
        selected.extend(rng.choice(ids, size=take, replace=False).astype(int).tolist())
    if len(selected) < n:
        rest = metrics.loc[~metrics["sample_id"].isin(selected), "sample_id"].to_numpy()
        take = min(len(rest), n - len(selected))
        if take > 0:
            selected.extend(rng.choice(rest, size=take, replace=False).astype(int).tolist())
    return selected[:n]


def save_figures(metrics: pd.DataFrame, quantiles: pd.DataFrame, path_frame: pd.DataFrame) -> None:
    scenario.dynamic.configure_plot_style()
    ensure_parent(FAN_FIGURE)
    ensure_parent(ACADEMIC_PANEL_FIGURE)

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    dates = pd.to_datetime(quantiles["trade_date"])
    ax.fill_between(dates, quantiles["p05"], quantiles["p95"], color="#bfdbfe", alpha=0.45, label="5%-95% 区间")
    ax.fill_between(dates, quantiles["p25"], quantiles["p75"], color="#60a5fa", alpha=0.35, label="25%-75% 区间")
    ax.plot(dates, quantiles["p50"], color="#1d4ed8", linewidth=2.2, label="中位数路径")
    ax.axhspan(110, 120, color="#f59e0b", alpha=0.10, label="110-120美元平台")
    ax.axhline(120, color="#dc2626", linestyle="--", linewidth=1.0, label="120美元风险线")
    ax.set_title("蒙特卡洛情景树：60-180天价格扇形区间")
    ax.set_xlabel("日期")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper right", ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FAN_FIGURE, dpi=200)
    plt.close(fig)

    risk_probs = pd.DataFrame(
        {
            "风险事件": ["最高价>=120", "最高价>=130", "180天缺口>500", "高二次跳涨风险"],
            "概率": [
                metrics["是否突破120"].mean(),
                metrics["是否突破130"].mean(),
                metrics["是否高缺口"].mean(),
                (metrics["二次跳涨风险"] == "高").mean(),
            ],
        }
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bars = ax.bar(risk_probs["风险事件"], risk_probs["概率"] * 100, color=["#2563eb", "#dc2626", "#f59e0b", "#7c3aed"])
    ax.set_title("蒙特卡洛尾部风险概率")
    ax.set_ylabel("概率（%）")
    ax.set_ylim(0, max(10, float((risk_probs["概率"] * 100).max()) + 8))
    for bar, value in zip(bars, risk_probs["概率"] * 100, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{value:.1f}%", ha="center", va="bottom")
    ax.tick_params(axis="x", labelrotation=12)
    fig.tight_layout()
    fig.savefig(TAIL_FIGURE, dpi=200)
    plt.close(fig)

    dates = pd.to_datetime(quantiles["trade_date"])
    path_frame = path_frame.merge(metrics[["sample_id", "路径类别"]], on="sample_id", how="left")
    path_frame["trade_date"] = pd.to_datetime(path_frame["trade_date"])
    selected_ids = _select_path_samples(metrics, n=120, seed=RANDOM_SEED + 7)
    selected_paths = path_frame[path_frame["sample_id"].isin(selected_ids)]
    category_color = {"缓和路径": "#2563eb", "中性附近": "#64748b", "高压尾部": "#b91c1c"}

    fig = plt.figure(figsize=(12.8, 6.8), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[2.9, 0.06, 1.15], height_ratios=[1.1, 0.9])
    ax_main = fig.add_subplot(gs[:, 0])
    ax_dist = fig.add_subplot(gs[0, 2], sharey=ax_main)
    ax_tail = fig.add_subplot(gs[1, 2])

    ax_main.axhspan(110, 120, color="#fef3c7", alpha=0.45, zorder=0)
    for _, group in selected_paths.groupby("sample_id"):
        label = str(group["路径类别"].iloc[0])
        ax_main.plot(
            group["trade_date"],
            group["forecast_price"],
            color=category_color.get(label, "#64748b"),
            alpha=0.075,
            linewidth=0.7,
            zorder=1,
        )
    ax_main.fill_between(dates, quantiles["p05"], quantiles["p95"], color="#dbeafe", alpha=0.74, zorder=2)
    ax_main.fill_between(dates, quantiles["p25"], quantiles["p75"], color="#93c5fd", alpha=0.58, zorder=3)
    ax_main.plot(dates, quantiles["p50"], color="#0f172a", linewidth=2.35, zorder=4)
    ax_main.axhline(120, color="#991b1b", linestyle=(0, (5, 3)), linewidth=1.2, zorder=5)
    ax_main.text(dates.iloc[-1], 120.7, "120 美元风险线", color="#991b1b", ha="right", va="bottom", fontsize=9)
    ax_main.text(dates.iloc[4], 116.7, "110-120 美元平台", color="#92400e", ha="left", va="center", fontsize=9)
    ax_main.set_ylabel("布伦特价格（美元/桶）")
    ax_main.set_xlabel("外推日期")
    ax_main.set_ylim(84, max(132, float(quantiles["p95"].max()) + 2))
    ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    _format_axis(ax_main)

    final_prices = metrics["第180天价格"].to_numpy()
    bins = np.linspace(float(final_prices.min()) - 1, float(final_prices.max()) + 1, 22)
    ax_dist.hist(final_prices, bins=bins, orientation="horizontal", color="#cbd5e1", edgecolor="white", linewidth=0.6)
    q05, q50, q95 = np.quantile(final_prices, [0.05, 0.50, 0.95])
    for q, color, text in [(q05, "#2563eb", "P05"), (q50, "#0f172a", "P50"), (q95, "#2563eb", "P95")]:
        ax_dist.axhline(q, color=color, linewidth=1.1)
        ax_dist.text(ax_dist.get_xlim()[1] * 0.96, q, text, color=color, ha="right", va="bottom", fontsize=8)
    ax_dist.axhline(120, color="#991b1b", linestyle=(0, (5, 3)), linewidth=1.0)
    ax_dist.set_xlabel("样本数")
    ax_dist.set_title("第180天分布", fontsize=10, pad=7)
    ax_dist.tick_params(axis="y", labelleft=False)
    _format_axis(ax_dist)

    thresholds = np.arange(95, 141, 1)
    exceedance = [(metrics["外推期最高价"] >= threshold).mean() * 100 for threshold in thresholds]
    ax_tail.plot(thresholds, exceedance, color="#0f172a", linewidth=2.0)
    ax_tail.fill_between(thresholds, exceedance, color="#bfdbfe", alpha=0.55)
    prob120 = float((metrics["外推期最高价"] >= 120).mean() * 100)
    prob130 = float((metrics["外推期最高价"] >= 130).mean() * 100)
    ax_tail.scatter([120, 130], [prob120, prob130], color=["#991b1b", "#991b1b"], s=32, zorder=3)
    ax_tail.text(120, prob120 + 2.0, f"120: {prob120:.1f}%", color="#991b1b", ha="center", fontsize=8.5)
    ax_tail.text(130, prob130 + 2.0, f"130: {prob130:.1f}%", color="#991b1b", ha="center", fontsize=8.5)
    ax_tail.set_xlabel("最高价阈值（美元/桶）")
    ax_tail.set_ylabel("突破概率（%）")
    ax_tail.set_ylim(0, max(35, max(exceedance) + 4))
    ax_tail.set_title("尾部突破概率曲线", fontsize=10, pad=7)
    _format_axis(ax_tail)

    legend_handles = [
        Patch(facecolor="#dbeafe", alpha=0.74, label="5%-95% 区间"),
        Patch(facecolor="#93c5fd", alpha=0.58, label="25%-75% 区间"),
        Line2D([0], [0], color="#0f172a", linewidth=2.2, label="中位数路径"),
        Line2D([0], [0], color="#64748b", alpha=0.45, linewidth=1.2, label="抽样路径云"),
        Line2D([0], [0], color="#991b1b", linestyle=(0, (5, 3)), linewidth=1.1, label="风险阈值"),
    ]
    ax_main.legend(handles=legend_handles, loc="upper right", frameon=False, ncol=2)
    fig.savefig(ACADEMIC_PANEL_FIGURE, dpi=260)
    plt.close(fig)

    cloud_ids = _select_path_samples(metrics, n=320, seed=RANDOM_SEED + 13)
    cloud = path_frame[path_frame["sample_id"].isin(cloud_ids)]
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    ax.axhspan(110, 120, color="#fef3c7", alpha=0.45, zorder=0)
    for _, group in cloud.groupby("sample_id"):
        label = str(group["路径类别"].iloc[0])
        ax.plot(
            group["trade_date"],
            group["forecast_price"],
            color=category_color.get(label, "#64748b"),
            alpha=0.055,
            linewidth=0.65,
            zorder=1,
        )
    ax.plot(dates, quantiles["p50"], color="#0f172a", linewidth=2.35, label="中位数路径", zorder=3)
    ax.plot(dates, quantiles["p05"], color="#2563eb", linewidth=1.0, linestyle="--", label="P05/P95", zorder=3)
    ax.plot(dates, quantiles["p95"], color="#2563eb", linewidth=1.0, linestyle="--", zorder=3)
    ax.axhline(120, color="#991b1b", linestyle=(0, (5, 3)), linewidth=1.1, label="120 美元风险线")
    ax.set_xlabel("外推日期")
    ax.set_ylabel("布伦特价格（美元/桶）")
    ax.set_ylim(84, max(132, float(quantiles["p95"].max()) + 2))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    _format_axis(ax)
    ax.legend(loc="upper right", frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(PATH_CLOUD_FIGURE, dpi=240)
    plt.close(fig)


def build_report(metrics: pd.DataFrame, tail_risk: pd.DataFrame) -> str:
    risk_lookup = dict(zip(tail_risk["指标"], tail_risk["数值"], strict=False))
    category_rows = "\n".join(
        "| {路径类别} | {样本数} | {第180天价格中位数:.2f} | {外推最高价P95:.2f} | {突破120概率:.1%} |".format(**row)
        for row in (
            metrics.groupby("路径类别")
            .agg(
                样本数=("sample_id", "count"),
                第180天价格中位数=("第180天价格", "median"),
                外推最高价P95=("外推期最高价", lambda x: float(x.quantile(0.95))),
                突破120概率=("是否突破120", "mean"),
            )
            .reset_index()
            .to_dict("records")
        )
    )
    stress_noise = float(metrics["historical_stress_noise_sigma"].iloc[0])
    event_vol_pct = float(metrics["historical_event_volatility_percentile"].iloc[0])
    return f"""# 蒙特卡洛情景树报告

## 运行结论

蒙特卡洛情景树在三情景模型基础上进行 {len(metrics)} 次联合扰动模拟。扰动变量包括供应中断量、SPR释放上限、SPR启动延迟、绕道运输能力、长期需求弹性、恐慌衰减速度、地缘风险权重、不确定性与制度风险强度、预期修复强度等。

扰动均限定在赛题边界或敏感性分析使用过的合理区间内；压力指数的噪声项由 2017--2025 附件历史波动分位数校准，当前标准差为 {stress_noise:.3f}，冲突窗口波动位于历史约 {event_vol_pct:.1%} 分位。本阶段没有引入爬虫数据，也没有生成任何未来真实价格。输出结果只用于刻画条件情景下的概率区间和尾部风险。

需要说明：本模型没有假设所有参数相互独立，而是用综合压力指数将供应中断、SPR 可用量、绕道恢复、长期弹性、风险权重和制度风险强度联动起来。该相关结构是基于赛题机制和能源市场逻辑设定的条件先验，不是从未来真实样本估计出的协方差矩阵。因此，突破 120 或 130 美元的概率应解释为本文模型设定下的条件尾部概率，而不是无条件历史频率。

## 核心概率结果

| 指标 | 数值 |
|---|---:|
| 外推期最高价突破120概率 | {risk_lookup["外推期最高价突破120"]:.1%} |
| 外推期最高价突破130概率 | {risk_lookup["外推期最高价突破130"]:.1%} |
| 第180天剩余缺口超过500概率 | {risk_lookup["第180天剩余缺口超过500"]:.1%} |
| 二次跳涨风险为高概率 | {risk_lookup["二次跳涨风险为高"]:.1%} |
| 第180天价格P05 | {risk_lookup["第180天价格P05"]:.2f} |
| 第180天价格P50 | {risk_lookup["第180天价格P50"]:.2f} |
| 第180天价格P95 | {risk_lookup["第180天价格P95"]:.2f} |
| 外推期最高价P95 | {risk_lookup["外推期最高价P95"]:.2f} |

## 路径类别摘要

| 路径类别 | 样本数 | 第180天价格中位数 | 外推最高价P95 | 突破120概率 |
|---|---:|---:|---:|---:|
{category_rows}

## 论文可用表述

蒙特卡洛情景树表明，三情景结论不是单点参数假设的产物。在多参数联合扰动下，第180天价格中位数为 {risk_lookup["第180天价格P50"]:.2f} 美元/桶，5%-95% 区间为 {risk_lookup["第180天价格P05"]:.2f}--{risk_lookup["第180天价格P95"]:.2f} 美元/桶；外推期最高价突破120美元/桶的概率为 {risk_lookup["外推期最高价突破120"]:.1%}，突破130美元/桶的概率为 {risk_lookup["外推期最高价突破130"]:.1%}。这说明长期风险的重点不是中性路径的点预测，而是供应中断偏大、SPR不足、绕道恢复慢和制度风险溢价同时出现时的尾部组合。

## 输出产物

- `{SAMPLE_METRICS_CSV.relative_to(PROJECT_ROOT)}`
- `{QUANTILE_CSV.relative_to(PROJECT_ROOT)}`
- `{TAIL_RISK_CSV.relative_to(PROJECT_ROOT)}`
- `{PATH_SAMPLE_CSV.relative_to(PROJECT_ROOT)}`
- `{FAN_FIGURE.relative_to(PROJECT_ROOT)}`
- `{TAIL_FIGURE.relative_to(PROJECT_ROOT)}`
- `{ACADEMIC_PANEL_FIGURE.relative_to(PROJECT_ROOT)}`
- `{PATH_CLOUD_FIGURE.relative_to(PROJECT_ROOT)}`
"""


def write_outputs(metrics: pd.DataFrame, quantiles: pd.DataFrame, tail_risk: pd.DataFrame, path_frame: pd.DataFrame) -> None:
    ensure_parent(SAMPLE_METRICS_CSV)
    ensure_parent(QUANTILE_CSV)
    ensure_parent(TAIL_RISK_CSV)
    ensure_parent(PATH_SAMPLE_CSV)
    ensure_parent(REPORT_PATH)
    metrics.to_csv(SAMPLE_METRICS_CSV, index=False)
    quantiles.to_csv(QUANTILE_CSV, index=False)
    tail_risk.to_csv(TAIL_RISK_CSV, index=False)
    sampled_ids = _select_path_samples(metrics, n=320, seed=RANDOM_SEED + 13)
    path_frame[path_frame["sample_id"].isin(sampled_ids)].to_csv(PATH_SAMPLE_CSV, index=False)
    save_figures(metrics, quantiles, path_frame)
    REPORT_PATH.write_text(build_report(metrics, tail_risk), encoding="utf-8")


def main() -> None:
    metrics, quantiles, tail_risk, path_frame = run_monte_carlo()
    write_outputs(metrics, quantiles, tail_risk, path_frame)
    print("Stage 6.5 Monte Carlo scenario tree complete")
    print(f"Sample metrics: {SAMPLE_METRICS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Quantiles: {QUANTILE_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Tail risk: {TAIL_RISK_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Fan figure: {FAN_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Tail figure: {TAIL_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Academic panel: {ACADEMIC_PANEL_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Path cloud: {PATH_CLOUD_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
