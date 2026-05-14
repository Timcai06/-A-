"""Search a tail-preserving challenger for the long-horizon state model.

The current long-term state-transition model relaxes quickly after day 120.
This script tests a small family of transition rules that preserve late tail
risk when GPR/OVX pressure remains high. It writes candidate outputs only; it
does not overwrite the official long-term forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import long_term_state_transition as baseline
from src.common.paths import PROJECT_ROOT, ensure_parent
from src.common.plotting import SCENARIO_COLORS, configure_plot_style
from src.analysis.historical_volatility_calibration import load_historical_model_factors
from src.scenarios import scenario_forecast as scenario


OUTPUT_DIR = PROJECT_ROOT / "output" / "scenarios"
SEARCH_CSV = OUTPUT_DIR / "长期尾部保持Challenger搜索表.csv"
BEST_QUANTILE_CSV = OUTPUT_DIR / "长期尾部保持Challenger分位数.csv"
BEST_METRICS_CSV = OUTPUT_DIR / "长期尾部保持Challenger样本指标.csv"
BEST_STATE_SHARE_CSV = OUTPUT_DIR / "长期尾部保持Challenger状态占比.csv"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "长期尾部保持Challenger报告.md"
FIGURE_PATH = PROJECT_ROOT / "output" / "candidate_figures" / "长期尾部保持Challenger对比.png"

RANDOM_SEED = 20260514
SEARCH_SAMPLES = 1200
FINAL_SAMPLES = 3000
MARKER_DAYS = [90, 120, 150, 180]


@dataclass(frozen=True)
class TailSetting:
    name: str
    late_easing_scale: float
    risk_decay_floor: float
    risk_boost_scale: float
    late_escalation_floor: float


SETTINGS: tuple[TailSetting, ...] = (
    TailSetting("轻度尾部保持", 0.65, 0.28, 1.10, 0.025),
    TailSetting("中度尾部保持", 0.45, 0.38, 1.25, 0.040),
    TailSetting("高尾部保持", 0.30, 0.45, 1.35, 0.055),
    TailSetting("强风险惯性", 0.35, 0.55, 1.45, 0.045),
    TailSetting("低缓和高惯性", 0.20, 0.50, 1.30, 0.060),
    TailSetting("保守折中", 0.55, 0.35, 1.15, 0.035),
)


def _normalize(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 0.01, 0.98)
    return probs / probs.sum()


def transition_probabilities(
    current_state: str,
    day_index: int,
    risk_constraints: baseline.StateRiskConstraints,
    setting: TailSetting,
) -> np.ndarray:
    probs = baseline.TRANSITION_MATRIX[baseline.STATE_INDEX[current_state]].copy()
    risk_decay = float(np.clip(1.0 - max(day_index - 60, 0) / 150.0, setting.risk_decay_floor, 1.0))
    risk_shift = risk_constraints.combined_pressure * risk_decay * setting.risk_boost_scale

    escalation_boost = 0.020 * risk_shift
    easing_discount = 0.014 * risk_shift
    probs[baseline.STATE_INDEX["escalation"]] += escalation_boost
    probs[baseline.STATE_INDEX["easing"]] -= easing_discount
    if current_state == "escalation":
        probs[baseline.STATE_INDEX["escalation"]] += 0.012 * risk_shift
        probs[baseline.STATE_INDEX["neutral"]] -= 0.004 * risk_shift
    elif current_state == "easing":
        probs[baseline.STATE_INDEX["neutral"]] += 0.006 * risk_shift

    if day_index >= 120:
        probs[baseline.STATE_INDEX["easing"]] += 0.04 * setting.late_easing_scale
        probs[baseline.STATE_INDEX["escalation"]] -= 0.04 * setting.late_easing_scale
    if day_index >= 150:
        probs[baseline.STATE_INDEX["easing"]] += 0.03 * setting.late_easing_scale
        probs[baseline.STATE_INDEX["neutral"]] -= 0.02 * setting.late_easing_scale
        probs[baseline.STATE_INDEX["escalation"]] -= 0.01 * setting.late_easing_scale

    if day_index >= 120 and probs[baseline.STATE_INDEX["escalation"]] < setting.late_escalation_floor:
        deficit = setting.late_escalation_floor - probs[baseline.STATE_INDEX["escalation"]]
        probs[baseline.STATE_INDEX["escalation"]] += deficit
        probs[baseline.STATE_INDEX["easing"]] -= deficit * 0.70
        probs[baseline.STATE_INDEX["neutral"]] -= deficit * 0.30

    return _normalize(probs)


def sample_next_state(
    rng: np.random.Generator,
    current_state: str,
    day_index: int,
    risk_constraints: baseline.StateRiskConstraints,
    setting: TailSetting,
) -> str:
    probs = transition_probabilities(current_state, day_index, risk_constraints, setting)
    return str(rng.choice([spec.state for spec in baseline.STATES], p=probs))


def simulate_path(
    rng: np.random.Generator,
    sample_id: int,
    centers: dict[str, pd.DataFrame],
    start_price: float,
    daily_sigma: float,
    transition_jump_sigma_scale: float,
    risk_constraints: baseline.StateRiskConstraints,
    setting: TailSetting,
) -> pd.DataFrame:
    state = "neutral"
    previous_price = start_price
    previous_noise = 0.0
    rows: list[dict[str, Any]] = []
    dates = centers["neutral"][["day_index", "trade_date"]].copy()

    for _, base_row in dates.iterrows():
        day_index = int(base_row["day_index"])
        next_state = sample_next_state(rng, state, day_index, risk_constraints, setting)
        spec = baseline.STATES[baseline.STATE_INDEX[next_state]]
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
                "candidate": setting.name,
            }
        )
        state = next_state
        previous_price = price
    return pd.DataFrame(rows)


def run_setting(setting: TailSetting, n_samples: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_result, short_path, history = baseline.load_inputs()
    centers = baseline.center_lookup(scenario_result)
    cutoff_price = float(short_path.sort_values("trade_date").iloc[-1]["simulated_price"])
    daily_sigma = baseline.estimate_event_volatility(short_path, history)
    jump_scale = load_historical_model_factors().transition_jump_sigma_scale
    constraints = baseline.load_state_risk_constraints(write_output=False)
    rng = np.random.default_rng(RANDOM_SEED + int(setting.late_escalation_floor * 10000) + n_samples)
    paths = pd.concat(
        [
            simulate_path(rng, sample_id, centers, cutoff_price, daily_sigma, jump_scale, constraints, setting)
            for sample_id in range(1, n_samples + 1)
        ],
        ignore_index=True,
    )
    return paths, baseline.build_quantiles(paths), baseline.summarize_samples(paths)


def summarize_candidate(setting: TailSetting, quantiles: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, Any]:
    final_q = quantiles[quantiles["day_index"] == 180].iloc[0]
    day90_q = quantiles[quantiles["day_index"] == 90].iloc[0]
    p120 = float(metrics["突破120"].mean())
    p130 = float(metrics["突破130"].mean())
    final_p50 = float(final_q["p50"])
    final_width = float(final_q["p95"] - final_q["p05"])
    neutral_target = float(pd.read_csv(scenario.SCENARIO_METRICS_CSV).query("scenario == 'neutral'")["第180天价格"].iloc[0])

    score = 0.0
    score -= abs(final_p50 - neutral_target) / 6.0
    score -= abs(final_width - 14.0) / 10.0
    score -= abs(p120 - 0.36) * 1.2
    score -= max(p130 - 0.10, 0.0) * 2.0
    score -= max(84.0 - final_p50, 0.0) * 0.3
    score -= max(final_p50 - 98.0, 0.0) * 0.3

    return {
        "candidate": setting.name,
        "late_easing_scale": setting.late_easing_scale,
        "risk_decay_floor": setting.risk_decay_floor,
        "risk_boost_scale": setting.risk_boost_scale,
        "late_escalation_floor": setting.late_escalation_floor,
        "第90天P95-P05": float(day90_q["p95"] - day90_q["p05"]),
        "第180天P50": final_p50,
        "第180天P95": float(final_q["p95"]),
        "第180天P95-P05": final_width,
        "突破120概率": p120,
        "突破130概率": p130,
        "外推期最高价P95": float(metrics["外推期最高价"].quantile(0.95)),
        "升级状态平均占比": float(metrics["升级占比"].mean()),
        "第180天最终升级占比": float((metrics["最终状态"] == "升级").mean()),
        "综合评分": score,
    }


def search_candidates() -> tuple[TailSetting, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for setting in SETTINGS:
        _, quantiles, metrics = run_setting(setting, SEARCH_SAMPLES)
        rows.append(summarize_candidate(setting, quantiles, metrics))
    search = pd.DataFrame(rows).sort_values("综合评分", ascending=False).reset_index(drop=True)
    ensure_parent(SEARCH_CSV)
    search.to_csv(SEARCH_CSV, index=False)
    best_name = str(search.iloc[0]["candidate"])
    best = next(setting for setting in SETTINGS if setting.name == best_name)
    return best, search


def save_best_outputs(setting: TailSetting) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths, quantiles, metrics = run_setting(setting, FINAL_SAMPLES)
    state_share = baseline.build_state_share(paths)
    ensure_parent(BEST_QUANTILE_CSV)
    quantiles.to_csv(BEST_QUANTILE_CSV, index=False)
    metrics.to_csv(BEST_METRICS_CSV, index=False)
    state_share.to_csv(BEST_STATE_SHARE_CSV, index=False)
    return quantiles, metrics, state_share


def save_figure(challenger_quantiles: pd.DataFrame, challenger_state_share: pd.DataFrame) -> None:
    current_quantiles = pd.read_csv(baseline.PATH_QUANTILE_CSV, parse_dates=["trade_date"])
    configure_plot_style(savefig_dpi=220, figure_dpi=150, title_size=12)
    ensure_parent(FIGURE_PATH)
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.8), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})

    axes[0].fill_between(
        current_quantiles["day_index"],
        current_quantiles["p05"],
        current_quantiles["p95"],
        color=SCENARIO_COLORS["muted"],
        alpha=0.12,
        label="当前状态区间",
    )
    axes[0].plot(current_quantiles["day_index"], current_quantiles["p50"], color=SCENARIO_COLORS["muted"], linewidth=1.7, linestyle="--", label="当前P50")
    axes[0].fill_between(
        challenger_quantiles["day_index"],
        challenger_quantiles["p05"],
        challenger_quantiles["p95"],
        color=SCENARIO_COLORS["risk"],
        alpha=0.13,
        label="尾部保持区间",
    )
    axes[0].plot(challenger_quantiles["day_index"], challenger_quantiles["p50"], color=SCENARIO_COLORS["neutral"], linewidth=2.4, label="尾部保持P50")
    axes[0].plot(challenger_quantiles["day_index"], challenger_quantiles["p95"], color=SCENARIO_COLORS["pessimistic"], linewidth=1.7, label="尾部保持P95")
    axes[0].set_title("长期状态转移：当前模型与尾部保持Challenger")
    axes[0].set_ylabel("美元/桶")
    axes[0].legend(loc="upper right", ncol=3)

    pivot = challenger_state_share.pivot_table(index="day_index", columns="状态", values="状态占比", fill_value=0.0).sort_index()
    axes[1].stackplot(
        pivot.index,
        [pivot.get(label, pd.Series(0.0, index=pivot.index)) for label in ["缓和", "维持", "升级"]],
        labels=["缓和", "维持", "升级"],
        colors=[SCENARIO_COLORS["optimistic"], SCENARIO_COLORS["neutral"], SCENARIO_COLORS["pessimistic"]],
        alpha=0.72,
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("状态占比")
    axes[1].set_xlabel("封锁后天数")
    axes[1].legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH)
    plt.close(fig)


def write_report(best: TailSetting, search: pd.DataFrame, quantiles: pd.DataFrame, metrics: pd.DataFrame, state_share: pd.DataFrame) -> None:
    current_quantiles = pd.read_csv(baseline.PATH_QUANTILE_CSV)
    current_metrics = pd.read_csv(baseline.SAMPLE_METRICS_CSV)
    final_current = current_quantiles[current_quantiles["day_index"] == 180].iloc[0]
    final_best = quantiles[quantiles["day_index"] == 180].iloc[0]
    state_pivot = state_share.pivot_table(index="day_index", columns="状态", values="状态占比", fill_value=0.0)
    best_day180_escalation = float(state_pivot.loc[180].get("升级", 0.0))
    current_state_share = pd.read_csv(baseline.STATE_SHARE_CSV)
    current_pivot = current_state_share.pivot_table(index="day_index", columns="状态", values="状态占比", fill_value=0.0)
    current_day180_escalation = float(current_pivot.loc[180].get("升级", 0.0))

    top_rows = "\n".join(
        "| {name} | {p50:.2f} | {width:.2f} | {p120:.1%} | {p130:.1%} | {score:.3f} |".format(
            name=str(row["candidate"]),
            p50=float(row["第180天P50"]),
            width=float(row["第180天P95-P05"]),
            p120=float(row["突破120概率"]),
            p130=float(row["突破130概率"]),
            score=float(row["综合评分"]),
        )
        for _, row in search.head(4).iterrows()
    )
    report = f"""# 长期尾部保持 Challenger 报告

## 这轮在优化什么

当前长期状态转移模型的主要问题不是“没有预测逻辑”，而是第 120 天以后缓和概率上升较快，导致第 180 天升级状态占比和概率扇形宽度偏低。本 Challenger 只调整状态转移层：降低后期机械缓和修正、保留一部分 GPR/OVX 风险惯性，并给“升级”状态设置温和的后期概率底线。

## 候选搜索结果

| 候选设定 | 第180天P50 | 第180天P95-P05 | 突破120概率 | 突破130概率 | 综合评分 |
|---|---:|---:|---:|---:|---:|
{top_rows}

综合评分不是论文结论，只是工程筛选标准：希望 P50 不明显偏离中性中心，同时让第 180 天区间不要过早塌缩，并控制 130 美元以上极端尾部不过度夸张。

## 最优候选

最优候选为 **{best.name}**：

| 参数 | 数值 | 作用 |
|---|---:|---|
| late_easing_scale | {best.late_easing_scale:.2f} | 降低第120/150天后的机械缓和强度 |
| risk_decay_floor | {best.risk_decay_floor:.2f} | 保留长期 GPR/OVX 风险惯性 |
| risk_boost_scale | {best.risk_boost_scale:.2f} | 温和放大外部风险变量对升级概率的影响 |
| late_escalation_floor | {best.late_escalation_floor:.3f} | 给后期升级概率设置下限 |

## 与当前模型对比

| 指标 | 当前状态转移模型 | 尾部保持Challenger |
|---|---:|---:|
| 第180天P50 | {float(final_current["p50"]):.2f} | {float(final_best["p50"]):.2f} |
| 第180天P95 | {float(final_current["p95"]):.2f} | {float(final_best["p95"]):.2f} |
| 第180天P95-P05 | {float(final_current["p95"] - final_current["p05"]):.2f} | {float(final_best["p95"] - final_best["p05"]):.2f} |
| 突破120概率 | {float(current_metrics["突破120"].mean()):.1%} | {float(metrics["突破120"].mean()):.1%} |
| 突破130概率 | {float(current_metrics["突破130"].mean()):.1%} | {float(metrics["突破130"].mean()):.1%} |
| 第180天升级状态占比 | {current_day180_escalation:.1%} | {best_day180_escalation:.1%} |

## 我的判断

这个 Challenger 比当前模型更符合“长期封锁仍有再升级和库存耗尽尾部风险”的直觉，但它还不能直接替换正式模型。原因是它提升了长期尾部风险，却还没有用真实期限结构或更完整的供需数据校准后期概率底线。下一步应当继续做两件事：一是用 2017--2025 高波动历史片段估计状态转移概率范围；二是把 OVX 或期货期限结构作为时变风险输入，而不是只用事件窗口固定压力。

## 输出产物

- `{SEARCH_CSV.relative_to(PROJECT_ROOT)}`
- `{BEST_QUANTILE_CSV.relative_to(PROJECT_ROOT)}`
- `{BEST_METRICS_CSV.relative_to(PROJECT_ROOT)}`
- `{BEST_STATE_SHARE_CSV.relative_to(PROJECT_ROOT)}`
- `{FIGURE_PATH.relative_to(PROJECT_ROOT)}`
"""
    ensure_parent(REPORT_PATH)
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    best, search = search_candidates()
    quantiles, metrics, state_share = save_best_outputs(best)
    save_figure(quantiles, state_share)
    write_report(best, search, quantiles, metrics, state_share)
    print("Long-term tail-risk challenger complete")
    print(f"Best candidate: {best.name}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
