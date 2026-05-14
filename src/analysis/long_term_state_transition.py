"""State-transition enhancement for long-horizon scenario forecasts.

The deterministic three-scenario paths are useful center paths, but a real oil
market will not move along a smooth line. This module overlays a small Markov
state process on top of the existing physical scenario paths so the paper can
show conditional path uncertainty without inventing future observed prices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parent
from src.common.plotting import SCENARIO_COLORS, direct_label
from src.analysis.historical_volatility_calibration import load_historical_model_factors
from src.models import dynamic_short_term as dynamic
from src.scenarios import scenario_forecast as scenario


RANDOM_SEED = 20260512
N_SAMPLES = 3000
OUTPUT_DIR = PROJECT_ROOT / "output" / "scenarios"
OVX_METRICS_CSV = PROJECT_ROOT / "output" / "risk" / "OVX滞后风险检验指标.csv"
GPR_METRICS_CSV = PROJECT_ROOT / "output" / "risk" / "滞后风险指数审计指标.csv"
PATH_QUANTILE_CSV = OUTPUT_DIR / "长期状态转移路径分位数.csv"
SAMPLE_METRICS_CSV = OUTPUT_DIR / "长期状态转移样本指标.csv"
STATE_SHARE_CSV = OUTPUT_DIR / "长期状态转移状态占比.csv"
STATE_RISK_CONSTRAINT_CSV = OUTPUT_DIR / "长期状态转移风险约束.csv"
STATE_TRANSITION_MATRIX_CSV = OUTPUT_DIR / "长期状态转移基础矩阵.csv"
STATE_TRANSITION_SNAPSHOT_CSV = OUTPUT_DIR / "长期状态转移时变矩阵快照.csv"
STATE_RISK_ABLATION_CSV = OUTPUT_DIR / "长期状态转移风险约束消融.csv"
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


@dataclass(frozen=True)
class StateRiskConstraints:
    ovx_event_percentile: float = 0.0
    ovx_lag1_vol_corr: float = 0.0
    gpr_event_percentile: float = 0.0
    ovx_pressure: float = 0.0
    gpr_pressure: float = 0.0
    combined_pressure: float = 0.0
    evidence_note: str = "未读取到 GPR/OVX 风险审计指标，状态转移使用基础矩阵。"


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
MATRIX_SNAPSHOT_DAYS = [60, 90, 120, 150, 180]


def _load_metric_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_metric(frame: pd.DataFrame, metric: str, default: float = 0.0) -> float:
    if frame.empty or "指标" not in frame.columns or "数值" not in frame.columns:
        return default
    row = frame[frame["指标"].astype(str) == metric]
    if row.empty:
        return default
    value = pd.to_numeric(row.iloc[0]["数值"], errors="coerce")
    return default if pd.isna(value) else float(value)


def _percent_value(value: float) -> float:
    return value * 100 if 0 <= value <= 1 else value


def load_state_risk_constraints(write_output: bool = True) -> StateRiskConstraints:
    """Use lagged GPR and OVX audits to tilt state transitions conservatively."""
    ovx = _load_metric_frame(OVX_METRICS_CSV)
    gpr = _load_metric_frame(GPR_METRICS_CSV)

    ovx_percentile = _safe_metric(ovx, "冲突窗口OVX均值历史分位")
    ovx_lag1_vol_corr = _safe_metric(ovx, "OVX_lag1_与7日实现波动相关")
    gpr_march = _percent_value(_safe_metric(gpr, "2026年3月GPR历史分位数"))
    gpr_april = _percent_value(_safe_metric(gpr, "2026年4月GPR历史分位数"))
    gpr_event_percentile = max(gpr_march, gpr_april)

    ovx_percentile_pressure = np.clip((ovx_percentile - 85.0) / 15.0, 0.0, 1.0)
    ovx_corr_pressure = np.clip((ovx_lag1_vol_corr - 0.50) / 0.40, 0.0, 1.0)
    ovx_pressure = float(np.clip(0.65 * ovx_percentile_pressure + 0.35 * ovx_corr_pressure, 0.0, 1.0))
    gpr_pressure = float(np.clip((gpr_event_percentile - 90.0) / 10.0, 0.0, 1.0))
    combined_pressure = float(np.clip(0.60 * ovx_pressure + 0.40 * gpr_pressure, 0.0, 1.0))

    constraints = StateRiskConstraints(
        ovx_event_percentile=ovx_percentile,
        ovx_lag1_vol_corr=ovx_lag1_vol_corr,
        gpr_event_percentile=gpr_event_percentile,
        ovx_pressure=ovx_pressure,
        gpr_pressure=gpr_pressure,
        combined_pressure=combined_pressure,
        evidence_note=(
            "使用 OVX 冲突窗口历史分位数、滞后1日 OVX 与7日实现波动率相关性，"
            "以及 GPR 事件月历史分位数，温和调整长期状态转移概率；"
            "不使用同日价格方向信息，不生成未来真实价格。"
        ),
    )
    if write_output:
        write_state_risk_constraints(constraints)
    return constraints


def write_state_risk_constraints(constraints: StateRiskConstraints) -> None:
    ensure_parent(STATE_RISK_CONSTRAINT_CSV)
    pd.DataFrame(
        [
            {"约束项": "OVX冲突窗口均值历史分位", "数值": constraints.ovx_event_percentile},
            {"约束项": "OVX滞后1日与7日实现波动相关", "数值": constraints.ovx_lag1_vol_corr},
            {"约束项": "GPR事件月最高历史分位", "数值": constraints.gpr_event_percentile},
            {"约束项": "OVX市场波动压力", "数值": constraints.ovx_pressure},
            {"约束项": "GPR地缘风险压力", "数值": constraints.gpr_pressure},
            {"约束项": "综合状态转移压力", "数值": constraints.combined_pressure},
        ]
    ).assign(证据说明=constraints.evidence_note).to_csv(STATE_RISK_CONSTRAINT_CSV, index=False)


def write_transition_matrix() -> None:
    ensure_parent(STATE_TRANSITION_MATRIX_CSV)
    labels = [spec.label for spec in STATES]
    frame = pd.DataFrame(TRANSITION_MATRIX, columns=[f"转为{label}" for label in labels])
    frame.insert(0, "当前状态", labels)
    frame.to_csv(STATE_TRANSITION_MATRIX_CSV, index=False)


def transition_probabilities(
    current_state: str,
    day_index: int,
    risk_constraints: StateRiskConstraints,
) -> np.ndarray:
    """Return the normalized transition probabilities for a day/state pair."""
    probs = TRANSITION_MATRIX[STATE_INDEX[current_state]].copy()

    # Market-priced and geopolitical risk should mostly affect the early and
    # middle extrapolation window. The effect decays as the blockade persists
    # and physical recovery evidence accumulates.
    early_risk_decay = float(np.clip(1.0 - max(day_index - 60, 0) / 140.0, 0.18, 1.0))
    risk_shift = risk_constraints.combined_pressure * early_risk_decay
    escalation_boost = 0.020 * risk_shift
    easing_discount = 0.014 * risk_shift
    probs[STATE_INDEX["escalation"]] += escalation_boost
    probs[STATE_INDEX["easing"]] -= easing_discount
    if current_state == "escalation":
        probs[STATE_INDEX["escalation"]] += 0.010 * risk_shift
        probs[STATE_INDEX["neutral"]] -= 0.004 * risk_shift
    elif current_state == "easing":
        probs[STATE_INDEX["neutral"]] += 0.006 * risk_shift

    if day_index >= 120:
        probs[STATE_INDEX["easing"]] += 0.04
        probs[STATE_INDEX["escalation"]] -= 0.04
    if day_index >= 150:
        probs[STATE_INDEX["easing"]] += 0.03
        probs[STATE_INDEX["neutral"]] -= 0.02
        probs[STATE_INDEX["escalation"]] -= 0.01
    probs = np.clip(probs, 0.01, 0.98)
    return probs / probs.sum()


def build_transition_snapshots(risk_constraints: StateRiskConstraints) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = [spec.label for spec in STATES]
    for day in MATRIX_SNAPSHOT_DAYS:
        for spec in STATES:
            probs = transition_probabilities(spec.state, day, risk_constraints)
            rows.append(
                {
                    "day_index": day,
                    "当前状态": spec.label,
                    "转为缓和": float(probs[0]),
                    "转为维持": float(probs[1]),
                    "转为升级": float(probs[2]),
                    "综合状态转移压力": risk_constraints.combined_pressure,
                    "说明": "已加入GPR/OVX压力、时间衰减和后期缓和修正",
                }
            )
    return pd.DataFrame(rows, columns=["day_index", "当前状态", *[f"转为{label}" for label in labels], "综合状态转移压力", "说明"])


def write_transition_snapshots(risk_constraints: StateRiskConstraints) -> pd.DataFrame:
    frame = build_transition_snapshots(risk_constraints)
    ensure_parent(STATE_TRANSITION_SNAPSHOT_CSV)
    frame.to_csv(STATE_TRANSITION_SNAPSHOT_CSV, index=False)
    return frame


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
    risk_constraints: StateRiskConstraints,
) -> str:
    probs = transition_probabilities(current_state, day_index, risk_constraints)
    return str(rng.choice([spec.state for spec in STATES], p=probs))


def simulate_path(
    rng: np.random.Generator,
    sample_id: int,
    centers: dict[str, pd.DataFrame],
    start_price: float,
    daily_sigma: float,
    transition_jump_sigma_scale: float,
    risk_constraints: StateRiskConstraints,
) -> pd.DataFrame:
    state = "neutral"
    previous_price = start_price
    previous_noise = 0.0
    rows: list[dict[str, Any]] = []
    dates = centers["neutral"][["day_index", "trade_date"]].copy()

    for _, base_row in dates.iterrows():
        day_index = int(base_row["day_index"])
        next_state = sample_next_state(rng, state, day_index, risk_constraints)
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
                "combined_risk_pressure": risk_constraints.combined_pressure,
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
    ax.fill_between(dates, quantiles["p05"], quantiles["p95"], color=SCENARIO_COLORS["band_outer"], alpha=0.15, label="5%-95% 状态区间")
    ax.fill_between(dates, quantiles["p25"], quantiles["p75"], color=SCENARIO_COLORS["band_inner"], alpha=0.30, label="25%-75% 状态区间")
    ax.plot(dates, quantiles["p50"], color=SCENARIO_COLORS["neutral"], linewidth=2.6, label="状态转移中位数")
    neutral = scenario_result[
        (scenario_result["scenario"] == "neutral") & (~scenario_result["is_observed_price"].astype(bool))
    ].copy()
    ax.plot(
        neutral["trade_date"],
        neutral["forecast_price"],
        color=SCENARIO_COLORS["actual"],
        linewidth=1.6,
        linestyle="--",
        label="原中性中心线",
    )
    ax.axhspan(110, 120, color=SCENARIO_COLORS["optimistic"], alpha=0.07, label="110-120美元参考区间")
    ax.axhline(120, color=SCENARIO_COLORS["risk"], linestyle=":", linewidth=1.2)
    ax.set_xlim(dates.iloc[0], dates.iloc[-1] + pd.Timedelta(days=14))
    direct_label(ax, dates.iloc[-1], quantiles["p50"].iloc[-1], "状态转移中位数", SCENARIO_COLORS["neutral"], dx=8, dy=0)
    if not neutral.empty:
        direct_label(
            ax,
            neutral["trade_date"].iloc[-1],
            neutral["forecast_price"].iloc[-1],
            "原中性中心线",
            SCENARIO_COLORS["actual"],
            dx=8,
            dy=-10,
        )
    ax.set_title("长期状态转移情景树：中心路径不等于每日精确预测")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper right", ncol=2)

    pivot = state_share.pivot_table(index="trade_date", columns="状态", values="状态占比", fill_value=0.0).sort_index()
    state_colors = {"缓和": SCENARIO_COLORS["optimistic"], "维持": SCENARIO_COLORS["neutral"], "升级": SCENARIO_COLORS["pessimistic"]}
    axes[1].stackplot(
        pd.to_datetime(pivot.index),
        [pivot.get(label, pd.Series(0.0, index=pivot.index)) for label in ["缓和", "维持", "升级"]],
        labels=["缓和", "维持", "升级"],
        colors=[state_colors[label] for label in ["缓和", "维持", "升级"]],
        alpha=0.62,
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
    risk_constraints: StateRiskConstraints,
    transition_snapshots: pd.DataFrame,
    risk_ablation: pd.DataFrame,
) -> str:
    final_q = quantiles[quantiles["day_index"] == 180].iloc[0]
    peak_p95 = float(metrics["外推期最高价"].quantile(0.95))
    neutral_rows = transition_snapshots[transition_snapshots["当前状态"] == "维持"]
    neutral_snapshot = "\n".join(
        "| {day} | {easing:.3f} | {neutral:.3f} | {escalation:.3f} |".format(
            day=int(row["day_index"]),
            easing=float(row["转为缓和"]),
            neutral=float(row["转为维持"]),
            escalation=float(row["转为升级"]),
        )
        for _, row in neutral_rows.iterrows()
    )
    risk_ablation_rows = "\n".join(
        "| {name} | {p120:.1%} | {p130:.1%} | {peak:.2f} | {final_p95:.2f} |".format(
            name=str(row["模型设定"]),
            p120=float(row["突破120概率"]),
            p130=float(row["突破130概率"]),
            peak=float(row["外推期最高价P95"]),
            final_p95=float(row["第180天价格P95"]),
        )
        for _, row in risk_ablation.iterrows()
    )
    return f"""# 长期状态转移预测增强报告

## 核心结论

本增强模型不替代三情景物理中心路径，而是在其上加入缓和、维持、升级三类状态的马尔可夫切换，并用附件历史价格估计日度波动扰动。它回答的是：如果未来事件状态会切换，长期价格可能围绕中心路径怎样波动。

本轮使用 {len(metrics)} 条状态转移样本，日度扰动波动率由附件历史数据校准为 {daily_sigma:.4f}，状态切换跳变标准差乘数为 {transition_jump_sigma_scale:.2f}。状态转移概率进一步受到 GPR/OVX 风险约束，综合状态转移压力为 {risk_constraints.combined_pressure:.3f}。第 180 天价格中位数为 {final_q["p50"]:.2f} 美元/桶，5%-95% 区间为 {final_q["p05"]:.2f}--{final_q["p95"]:.2f} 美元/桶；外推期最高价 P95 为 {peak_p95:.2f} 美元/桶。突破 120 美元/桶的条件概率为 {metrics["突破120"].mean():.1%}，突破 130 美元/桶的条件概率为 {metrics["突破130"].mean():.1%}。

## 风险约束来源

| 约束项 | 数值 | 含义 |
|---|---:|---|
| OVX 冲突窗口均值历史分位 | {risk_constraints.ovx_event_percentile:.1f}% | 期权市场隐含波动率处于历史高位 |
| OVX 滞后1日与7日实现波动相关 | {risk_constraints.ovx_lag1_vol_corr:.3f} | OVX 更适合约束波动和尾部风险 |
| GPR 事件月最高历史分位 | {risk_constraints.gpr_event_percentile:.1f}% | 地缘风险新闻指数处于历史高位 |
| 综合状态转移压力 | {risk_constraints.combined_pressure:.3f} | 温和提高早中期升级概率，并随时间衰减 |

## 基础转移矩阵

行、列顺序均为“缓和、维持、升级”。基础矩阵为：

| 当前状态 | 转为缓和 | 转为维持 | 转为升级 |
|---|---:|---:|---:|
| 缓和 | 0.90 | 0.09 | 0.01 |
| 维持 | 0.16 | 0.78 | 0.06 |
| 升级 | 0.07 | 0.24 | 0.69 |

GPR/OVX 综合状态转移压力通过时变扰动项提高早中期升级概率，并在第 120 天和第 150 天后逐步提高缓和概率。所有行在扰动后重新归一化，因此每一日转移概率仍为合法概率分布。

## 时变转移矩阵快照

以下表格展示从“维持”状态出发时，在关键外推日的修正后转移概率。完整三状态快照见 `{STATE_TRANSITION_SNAPSHOT_CSV.relative_to(PROJECT_ROOT)}`。

| 外推日 | 转为缓和 | 转为维持 | 转为升级 |
|---:|---:|---:|---:|
{neutral_snapshot}

## GPR/OVX 风险约束消融

为检查外部风险变量是否只是装饰项，本文比较“仅基础矩阵”和“基础矩阵+GPR/OVX风险约束”两种状态转移设定。完整结果见 `{STATE_RISK_ABLATION_CSV.relative_to(PROJECT_ROOT)}`。

| 模型设定 | 突破120概率 | 突破130概率 | 外推期最高价P95 | 第180天价格P95 |
|---|---:|---:|---:|---:|
{risk_ablation_rows}

## 为什么它比单条线更可信

- 原三情景线保留为物理中心路径，用来表达供应恢复、SPR 收缩、需求弹性和风险溢价衰减的慢变量。
- 状态转移层表达未来事件的不确定性：冲突缓和时向乐观中心靠近，维持时围绕中性中心，升级时向悲观中心和跳涨方向移动；GPR/OVX 风险压力会温和提高早中期升级概率。
- 扰动强度和状态切换跳变尺度来自 2017--2025 附件历史价格波动，不使用爬虫数据，也不生成未来真实价格。
- 输出图应作为长期主图的辅助证据：长期预测不能承诺逐日精确命中，只能给出条件中心、概率区间和尾部风险。

## 输出产物

- `{PATH_QUANTILE_CSV.relative_to(PROJECT_ROOT)}`
- `{SAMPLE_METRICS_CSV.relative_to(PROJECT_ROOT)}`
- `{STATE_SHARE_CSV.relative_to(PROJECT_ROOT)}`
- `{STATE_RISK_CONSTRAINT_CSV.relative_to(PROJECT_ROOT)}`
- `{STATE_TRANSITION_MATRIX_CSV.relative_to(PROJECT_ROOT)}`
- `{STATE_TRANSITION_SNAPSHOT_CSV.relative_to(PROJECT_ROOT)}`
- `{STATE_RISK_ABLATION_CSV.relative_to(PROJECT_ROOT)}`
- `{FAN_FIGURE.relative_to(PROJECT_ROOT)}`
"""


def _run_state_transition_with_constraints(
    risk_constraints: StateRiskConstraints,
    n_samples: int = N_SAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
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
                risk_constraints,
            )
            for sample_id in range(1, n_samples + 1)
        ],
        ignore_index=True,
    )
    quantiles = build_quantiles(paths)
    metrics = summarize_samples(paths)
    state_share = build_state_share(paths)
    return metrics, quantiles, state_share, daily_sigma


def run_state_transition(n_samples: int = N_SAMPLES) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    return _run_state_transition_with_constraints(load_state_risk_constraints(), n_samples=n_samples)


def build_risk_constraint_ablation(current_constraints: StateRiskConstraints) -> pd.DataFrame:
    no_risk_constraints = StateRiskConstraints(
        evidence_note="消融设定：保留基础转移矩阵和历史波动扰动，但不加入GPR/OVX风险压力。"
    )
    settings = [
        ("仅基础矩阵", no_risk_constraints),
        ("基础矩阵+GPR/OVX风险约束", current_constraints),
    ]
    rows: list[dict[str, Any]] = []
    for label, constraints in settings:
        metrics, quantiles, _, _ = _run_state_transition_with_constraints(constraints, n_samples=N_SAMPLES)
        final_q = quantiles[quantiles["day_index"] == 180].iloc[0]
        rows.append(
            {
                "模型设定": label,
                "综合状态转移压力": constraints.combined_pressure,
                "突破120概率": float(metrics["突破120"].mean()),
                "突破130概率": float(metrics["突破130"].mean()),
                "外推期最高价P95": float(metrics["外推期最高价"].quantile(0.95)),
                "第180天价格P50": float(final_q["p50"]),
                "第180天价格P95": float(final_q["p95"]),
                "升级状态平均占比": float(metrics["升级占比"].mean()),
            }
        )
    frame = pd.DataFrame(rows)
    ensure_parent(STATE_RISK_ABLATION_CSV)
    frame.to_csv(STATE_RISK_ABLATION_CSV, index=False)
    return frame


def write_outputs(metrics: pd.DataFrame, quantiles: pd.DataFrame, state_share: pd.DataFrame, daily_sigma: float) -> None:
    ensure_parent(PATH_QUANTILE_CSV)
    ensure_parent(SAMPLE_METRICS_CSV)
    ensure_parent(STATE_SHARE_CSV)
    ensure_parent(REPORT_PATH)
    metrics.to_csv(SAMPLE_METRICS_CSV, index=False)
    quantiles.to_csv(PATH_QUANTILE_CSV, index=False)
    state_share.to_csv(STATE_SHARE_CSV, index=False)
    write_transition_matrix()
    risk_constraints = load_state_risk_constraints()
    transition_snapshots = write_transition_snapshots(risk_constraints)
    risk_ablation = build_risk_constraint_ablation(risk_constraints)
    scenario_result, _, _ = load_inputs()
    save_figure(quantiles, state_share, scenario_result)
    factors = load_historical_model_factors()
    REPORT_PATH.write_text(
        build_report(
            metrics,
            quantiles,
            daily_sigma,
            factors.transition_jump_sigma_scale,
            risk_constraints,
            transition_snapshots,
            risk_ablation,
        ),
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
