"""State-transition enhancement for long-horizon scenario forecasts.

The deterministic three-scenario paths are useful center paths, but a real oil
market will not move along a smooth line. This module overlays a small Markov
state process on top of the existing physical scenario paths so the paper can
show conditional path uncertainty without inventing future observed prices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parent
from src.analysis.historical_volatility_calibration import load_historical_model_factors
from src.models import dynamic_short_term as dynamic
from src.scenarios import scenario_forecast as scenario


RANDOM_SEED = 20260512
N_SAMPLES = 3000
OUTPUT_DIR = PROJECT_ROOT / "output" / "scenarios"
PATH_QUANTILE_CSV = OUTPUT_DIR / "长期状态转移路径分位数.csv"
SAMPLE_METRICS_CSV = OUTPUT_DIR / "长期状态转移样本指标.csv"
STATE_SHARE_CSV = OUTPUT_DIR / "长期状态转移状态占比.csv"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "长期状态转移预测增强报告.md"
FAN_FIGURE = PROJECT_ROOT / "figures" / "long_term_state_transition_fan.png"


@dataclass(frozen=True)
class StateSpec:
    state: str
    label: str
    center_scenario: str
    volatility_multiplier: float
    transition_jump_mean: float
    transition_jump_sigma: float


STATES: tuple[StateSpec, ...] = (
    StateSpec("easing", "缓和", "optimistic", 0.55, -0.50, 0.30),
    StateSpec("neutral", "维持", "neutral", 0.75, 0.00, 0.25),
    StateSpec("escalation", "升级", "pessimistic", 1.05, 0.75, 0.45),
)
STATE_INDEX = {spec.state: i for i, spec in enumerate(STATES)}

TRANSITION_MATRIX = np.array(
    [
        [0.90, 0.09, 0.01],
        [0.16, 0.78, 0.06],
        [0.07, 0.24, 0.69],
    ],
    dtype=float,
)

MARKER_DAYS = [90, 120, 150, 180]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_result = pd.read_csv(scenario.SCENARIO_RESULT_CSV, parse_dates=["trade_date"])
    short_path = pd.read_csv(
        PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv",
        parse_dates=["trade_date"],
    )
    history = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "布伦特原油期货主力合约价格数据_清洗后.csv",
        parse_dates=["trade_date"],
    )
    return scenario_result, short_path, history


def estimate_event_volatility(short_path: pd.DataFrame, history: pd.DataFrame) -> float:
    factors = load_historical_model_factors()
    if factors.state_daily_sigma > 0:
        return factors.state_daily_sigma

    event_return = short_path["actual_price"].astype(float).pct_change().dropna()
    history_return = history["close_price"].astype(float).pct_change().dropna()
    event_sigma = float(event_return.std(ddof=0))
    historical_sigma = float(history_return.std(ddof=0))
    # Blend event volatility with the long history so the disturbance is not
    # completely dominated by one short crisis window.
    blended_sigma = 0.70 * event_sigma + 0.30 * historical_sigma
    return float(np.clip(blended_sigma * 0.35, 0.006, 0.014))


def center_lookup(scenario_result: pd.DataFrame) -> dict[str, pd.DataFrame]:
    future = scenario_result[~scenario_result["is_observed_price"].astype(bool)].copy()
    return {
        key: group.sort_values("day_index").reset_index(drop=True)
        for key, group in future.groupby("scenario", sort=False)
    }


def sample_next_state(
    rng: np.random.Generator,
    current_state: str,
    day_index: int,
) -> str:
    probs = TRANSITION_MATRIX[STATE_INDEX[current_state]].copy()
    if day_index >= 120:
        probs[STATE_INDEX["easing"]] += 0.04
        probs[STATE_INDEX["escalation"]] -= 0.04
    if day_index >= 150:
        probs[STATE_INDEX["easing"]] += 0.03
        probs[STATE_INDEX["neutral"]] -= 0.02
        probs[STATE_INDEX["escalation"]] -= 0.01
    probs = np.clip(probs, 0.01, 0.98)
    probs = probs / probs.sum()
    return str(rng.choice([spec.state for spec in STATES], p=probs))


def simulate_path(
    rng: np.random.Generator,
    sample_id: int,
    centers: dict[str, pd.DataFrame],
    start_price: float,
    daily_sigma: float,
    transition_jump_sigma_scale: float,
) -> pd.DataFrame:
    state = "neutral"
    previous_price = start_price
    previous_noise = 0.0
    rows: list[dict[str, Any]] = []
    dates = centers["neutral"][["day_index", "trade_date"]].copy()

    for _, base_row in dates.iterrows():
        day_index = int(base_row["day_index"])
        next_state = sample_next_state(rng, state, day_index)
        spec = STATES[STATE_INDEX[next_state]]
        center_row = centers[spec.center_scenario]
        target_price = float(center_row.loc[center_row["day_index"] == day_index, "forecast_price"].iloc[0])

        noise_scale = max(previous_price * daily_sigma * spec.volatility_multiplier, 0.15)
        innovation = float(rng.normal(0.0, noise_scale))
        previous_noise = 0.30 * previous_noise + innovation
        transition_jump = 0.0
        if next_state != state:
            transition_jump = float(
                rng.normal(spec.transition_jump_mean, spec.transition_jump_sigma * transition_jump_sigma_scale)
            )

        price = 0.82 * previous_price + 0.18 * target_price + previous_noise + transition_jump
        price = float(np.clip(price, 70.0, 145.0))

        rows.append(
            {
                "sample_id": sample_id,
                "day_index": day_index,
                "trade_date": base_row["trade_date"],
                "状态": spec.label,
                "state": next_state,
                "center_scenario": spec.center_scenario,
                "center_price": target_price,
                "forecast_price": price,
                "state_noise": previous_noise,
                "transition_jump": transition_jump,
            }
        )
        state = next_state
        previous_price = price

    return pd.DataFrame(rows)


def build_quantiles(paths: pd.DataFrame) -> pd.DataFrame:
    return (
        paths.groupby(["day_index", "trade_date"])["forecast_price"]
        .quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        .unstack()
        .reset_index()
        .rename(columns={0.05: "p05", 0.10: "p10", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.90: "p90", 0.95: "p95"})
    )


def summarize_samples(paths: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_id, group in paths.groupby("sample_id", sort=True):
        group = group.sort_values("day_index")
        final = group.loc[group["day_index"] == 180].iloc[0]
        row: dict[str, Any] = {
            "sample_id": int(sample_id),
            "第180天价格": float(final["forecast_price"]),
            "外推期最高价": float(group["forecast_price"].max()),
            "外推期最低价": float(group["forecast_price"].min()),
            "外推期均价": float(group["forecast_price"].mean()),
            "突破120": bool(group["forecast_price"].max() >= 120),
            "突破130": bool(group["forecast_price"].max() >= 130),
            "最终状态": str(final["状态"]),
            "缓和占比": float((group["state"] == "easing").mean()),
            "维持占比": float((group["state"] == "neutral").mean()),
            "升级占比": float((group["state"] == "escalation").mean()),
        }
        for day in MARKER_DAYS:
            row[f"第{day}天价格"] = float(group.loc[group["day_index"] == day, "forecast_price"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def build_state_share(paths: pd.DataFrame) -> pd.DataFrame:
    counts = (
        paths.groupby(["day_index", "trade_date", "状态"])
        .size()
        .reset_index(name="count")
    )
    counts["状态占比"] = counts["count"] / counts.groupby(["day_index", "trade_date"])["count"].transform("sum")
    return counts


def save_figure(quantiles: pd.DataFrame, state_share: pd.DataFrame, scenario_result: pd.DataFrame) -> None:
    dynamic.configure_plot_style()
    ensure_parent(FAN_FIGURE)

    fig, axes = plt.subplots(2, 1, figsize=(11.8, 8.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    ax = axes[0]
    dates = pd.to_datetime(quantiles["trade_date"])
    ax.fill_between(dates, quantiles["p05"], quantiles["p95"], color="#dbeafe", alpha=0.70, label="5%-95% 状态区间")
    ax.fill_between(dates, quantiles["p25"], quantiles["p75"], color="#60a5fa", alpha=0.45, label="25%-75% 状态区间")
    ax.plot(dates, quantiles["p50"], color="#1d4ed8", linewidth=2.3, label="状态转移中位数")
    neutral = scenario_result[
        (scenario_result["scenario"] == "neutral") & (~scenario_result["is_observed_price"].astype(bool))
    ].copy()
    ax.plot(
        neutral["trade_date"],
        neutral["forecast_price"],
        color="#111827",
        linewidth=1.6,
        linestyle="--",
        label="原中性中心线",
    )
    ax.axhspan(110, 120, color="#f59e0b", alpha=0.11, label="110-120美元参考区间")
    ax.axhline(120, color="#dc2626", linestyle=":", linewidth=1.2)
    ax.set_title("长期状态转移情景树：中心路径不等于每日精确预测")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper right", ncol=2)

    pivot = state_share.pivot_table(index="trade_date", columns="状态", values="状态占比", fill_value=0.0).sort_index()
    state_colors = {"缓和": "#10b981", "维持": "#2563eb", "升级": "#dc2626"}
    axes[1].stackplot(
        pd.to_datetime(pivot.index),
        [pivot.get(label, pd.Series(0.0, index=pivot.index)) for label in ["缓和", "维持", "升级"]],
        labels=["缓和", "维持", "升级"],
        colors=[state_colors[label] for label in ["缓和", "维持", "升级"]],
        alpha=0.72,
    )
    axes[1].set_ylabel("状态占比")
    axes[1].set_xlabel("日期")
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="upper right", ncol=3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FAN_FIGURE, dpi=220)
    plt.close(fig)


def build_report(
    metrics: pd.DataFrame,
    quantiles: pd.DataFrame,
    daily_sigma: float,
    transition_jump_sigma_scale: float,
) -> str:
    final_q = quantiles[quantiles["day_index"] == 180].iloc[0]
    peak_p95 = float(metrics["外推期最高价"].quantile(0.95))
    return f"""# 长期状态转移预测增强报告

## 核心结论

本增强模型不替代三情景物理中心路径，而是在其上加入缓和、维持、升级三类状态的马尔可夫切换，并用附件历史价格估计日度波动扰动。它回答的是：如果未来事件状态会切换，长期价格可能围绕中心路径怎样波动。

本轮使用 {len(metrics)} 条状态转移样本，日度扰动波动率由附件历史数据校准为 {daily_sigma:.4f}，状态切换跳变标准差乘数为 {transition_jump_sigma_scale:.2f}。第 180 天价格中位数为 {final_q["p50"]:.2f} 美元/桶，5%-95% 区间为 {final_q["p05"]:.2f}--{final_q["p95"]:.2f} 美元/桶；外推期最高价 P95 为 {peak_p95:.2f} 美元/桶。突破 120 美元/桶的条件概率为 {metrics["突破120"].mean():.1%}，突破 130 美元/桶的条件概率为 {metrics["突破130"].mean():.1%}。

## 为什么它比单条线更可信

- 原三情景线保留为物理中心路径，用来表达供应恢复、SPR 收缩、需求弹性和风险溢价衰减的慢变量。
- 状态转移层表达未来事件的不确定性：冲突缓和时向乐观中心靠近，维持时围绕中性中心，升级时向悲观中心和跳涨方向移动。
- 扰动强度和状态切换跳变尺度来自 2017--2025 附件历史价格波动，不使用爬虫数据，也不生成未来真实价格。
- 输出图应作为长期主图的辅助证据：长期预测不能承诺逐日精确命中，只能给出条件中心、概率区间和尾部风险。

## 输出产物

- `{PATH_QUANTILE_CSV.relative_to(PROJECT_ROOT)}`
- `{SAMPLE_METRICS_CSV.relative_to(PROJECT_ROOT)}`
- `{STATE_SHARE_CSV.relative_to(PROJECT_ROOT)}`
- `{FAN_FIGURE.relative_to(PROJECT_ROOT)}`
"""


def run_state_transition(n_samples: int = N_SAMPLES) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    scenario_result, short_path, history = load_inputs()
    centers = center_lookup(scenario_result)
    cutoff_price = float(short_path.sort_values("trade_date").iloc[-1]["simulated_price"])
    factors = load_historical_model_factors()
    daily_sigma = estimate_event_volatility(short_path, history)
    rng = np.random.default_rng(RANDOM_SEED)
    paths = pd.concat(
        [
            simulate_path(
                rng,
                sample_id,
                centers,
                cutoff_price,
                daily_sigma,
                factors.transition_jump_sigma_scale,
            )
            for sample_id in range(1, n_samples + 1)
        ],
        ignore_index=True,
    )
    quantiles = build_quantiles(paths)
    metrics = summarize_samples(paths)
    state_share = build_state_share(paths)
    return metrics, quantiles, state_share, daily_sigma


def write_outputs(metrics: pd.DataFrame, quantiles: pd.DataFrame, state_share: pd.DataFrame, daily_sigma: float) -> None:
    ensure_parent(PATH_QUANTILE_CSV)
    ensure_parent(SAMPLE_METRICS_CSV)
    ensure_parent(STATE_SHARE_CSV)
    ensure_parent(REPORT_PATH)
    metrics.to_csv(SAMPLE_METRICS_CSV, index=False)
    quantiles.to_csv(PATH_QUANTILE_CSV, index=False)
    state_share.to_csv(STATE_SHARE_CSV, index=False)
    scenario_result, _, _ = load_inputs()
    save_figure(quantiles, state_share, scenario_result)
    factors = load_historical_model_factors()
    REPORT_PATH.write_text(
        build_report(metrics, quantiles, daily_sigma, factors.transition_jump_sigma_scale),
        encoding="utf-8",
    )


def main() -> None:
    metrics, quantiles, state_share, daily_sigma = run_state_transition()
    write_outputs(metrics, quantiles, state_share, daily_sigma)
    print("Long-term state-transition enhancement complete")
    print(f"Metrics: {SAMPLE_METRICS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Quantiles: {PATH_QUANTILE_CSV.relative_to(PROJECT_ROOT)}")
    print(f"State share: {STATE_SHARE_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Figure: {FAN_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
