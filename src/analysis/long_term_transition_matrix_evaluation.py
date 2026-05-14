"""Evaluate historical transition matrices for the long-term scenario tree."""

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


MATRIX_CSV = PROJECT_ROOT / "output" / "scenarios" / "长期历史状态转移校准矩阵.csv"
EVALUATION_CSV = PROJECT_ROOT / "output" / "scenarios" / "长期转移矩阵替换评估表.csv"
BEST_QUANTILE_CSV = PROJECT_ROOT / "output" / "scenarios" / "长期历史校准矩阵候选分位数.csv"
BEST_METRICS_CSV = PROJECT_ROOT / "output" / "scenarios" / "长期历史校准矩阵候选样本指标.csv"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "长期转移矩阵替换评估报告.md"
FIGURE_PATH = PROJECT_ROOT / "output" / "candidate_figures" / "长期转移矩阵替换评估.png"

RANDOM_SEED = 20260514
N_SAMPLES = 2400
STATE_ORDER = ["缓和", "维持", "升级"]
STATE_KEY_ORDER = ["easing", "neutral", "escalation"]


@dataclass(frozen=True)
class MatrixVariant:
    name: str
    matrix_name: str
    risk_scale: float
    late_easing_scale: float
    note: str


VARIANTS = (
    MatrixVariant("当前基础矩阵复跑", "当前基础矩阵", 1.00, 1.00, "复跑当前规则，作为随机误差参照。"),
    MatrixVariant("历史推荐矩阵_原始风险", "历史收缩推荐矩阵", 1.00, 1.00, "直接把历史推荐矩阵接入，检验是否过激。"),
    MatrixVariant("历史推荐矩阵_风险折半", "历史收缩推荐矩阵", 0.50, 1.00, "保留历史转移概率，但降低额外GPR/OVX扰动。"),
    MatrixVariant("历史推荐矩阵_尾部折中", "历史收缩推荐矩阵", 0.65, 0.55, "降低后期机械缓和，同时不过度叠加风险扰动。"),
    MatrixVariant("历史全样本矩阵", "历史全样本矩阵", 0.75, 0.75, "用全样本历史状态矩阵作为偏保守经验参照。"),
)


def load_matrix(matrix_name: str) -> np.ndarray:
    table = pd.read_csv(MATRIX_CSV)
    sub = table[table["矩阵"] == matrix_name].copy()
    if len(sub) != 3:
        raise RuntimeError(f"矩阵 {matrix_name} 行数异常: {len(sub)}")
    sub = sub.set_index("当前状态").reindex(STATE_ORDER)
    matrix = sub[["转为缓和", "转为维持", "转为升级"]].astype(float).to_numpy()
    row_sums = matrix.sum(axis=1)
    fixed = matrix.copy()
    for idx, row_sum in enumerate(row_sums):
        if row_sum <= 0:
            fixed[idx] = baseline.TRANSITION_MATRIX[idx]
    fixed = fixed / fixed.sum(axis=1, keepdims=True)
    return fixed


def transition_probabilities(
    current_state: str,
    day_index: int,
    risk_constraints: baseline.StateRiskConstraints,
    variant: MatrixVariant,
    matrix: np.ndarray,
) -> np.ndarray:
    probs = matrix[baseline.STATE_INDEX[current_state]].copy()
    early_risk_decay = float(np.clip(1.0 - max(day_index - 60, 0) / 150.0, 0.25, 1.0))
    risk_shift = risk_constraints.combined_pressure * early_risk_decay * variant.risk_scale
    probs[baseline.STATE_INDEX["escalation"]] += 0.014 * risk_shift
    probs[baseline.STATE_INDEX["easing"]] -= 0.010 * risk_shift
    if current_state == "escalation":
        probs[baseline.STATE_INDEX["escalation"]] += 0.006 * risk_shift
    elif current_state == "easing":
        probs[baseline.STATE_INDEX["neutral"]] += 0.004 * risk_shift

    if day_index >= 120:
        probs[baseline.STATE_INDEX["easing"]] += 0.04 * variant.late_easing_scale
        probs[baseline.STATE_INDEX["escalation"]] -= 0.04 * variant.late_easing_scale
    if day_index >= 150:
        probs[baseline.STATE_INDEX["easing"]] += 0.03 * variant.late_easing_scale
        probs[baseline.STATE_INDEX["neutral"]] -= 0.02 * variant.late_easing_scale
        probs[baseline.STATE_INDEX["escalation"]] -= 0.01 * variant.late_easing_scale

    probs = np.clip(probs, 0.01, 0.98)
    return probs / probs.sum()


def simulate_variant(variant: MatrixVariant) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_result, short_path, history = baseline.load_inputs()
    centers = baseline.center_lookup(scenario_result)
    cutoff_price = float(short_path.sort_values("trade_date").iloc[-1]["simulated_price"])
    daily_sigma = baseline.estimate_event_volatility(short_path, history)
    jump_scale = load_historical_model_factors().transition_jump_sigma_scale
    constraints = baseline.load_state_risk_constraints(write_output=False)
    matrix = load_matrix(variant.matrix_name)
    rng = np.random.default_rng(RANDOM_SEED + len(variant.name))
    dates = centers["neutral"][["day_index", "trade_date"]].copy()
    rows: list[dict[str, Any]] = []

    for sample_id in range(1, N_SAMPLES + 1):
        state = "neutral"
        previous_price = cutoff_price
        previous_noise = 0.0
        for _, base_row in dates.iterrows():
            day_index = int(base_row["day_index"])
            probs = transition_probabilities(state, day_index, constraints, variant, matrix)
            next_state = str(rng.choice(STATE_KEY_ORDER, p=probs))
            spec = baseline.STATES[baseline.STATE_INDEX[next_state]]
            center_row = centers[spec.center_scenario]
            target_price = float(center_row.loc[center_row["day_index"] == day_index, "forecast_price"].iloc[0])
            noise_scale = max(previous_price * daily_sigma * spec.volatility_multiplier, 0.15)
            previous_noise = 0.30 * previous_noise + float(rng.normal(0.0, noise_scale))
            transition_jump = 0.0
            if next_state != state:
                transition_jump = float(rng.normal(spec.transition_jump_mean, spec.transition_jump_sigma * jump_scale))
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
                    "variant": variant.name,
                }
            )
            state = next_state
            previous_price = price

    paths = pd.DataFrame(rows)
    return paths, baseline.build_quantiles(paths), baseline.summarize_samples(paths)


def summarize_variant(variant: MatrixVariant, quantiles: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, Any]:
    final_q = quantiles[quantiles["day_index"] == 180].iloc[0]
    day90_q = quantiles[quantiles["day_index"] == 90].iloc[0]
    day180_width = float(final_q["p95"] - final_q["p05"])
    p120 = float(metrics["突破120"].mean())
    p130 = float(metrics["突破130"].mean())
    p50 = float(final_q["p50"])
    p95 = float(final_q["p95"])
    score = 0.0
    score -= abs(p50 - 91.37) / 6.0
    score -= abs(day180_width - 14.0) / 10.0
    score -= abs(p120 - 0.35) * 1.0
    score -= max(p130 - 0.08, 0.0) * 2.0
    score -= max(p95 - 108.0, 0.0) * 0.12
    return {
        "候选": variant.name,
        "使用矩阵": variant.matrix_name,
        "风险扰动倍率": variant.risk_scale,
        "后期缓和倍率": variant.late_easing_scale,
        "第90天P95-P05": float(day90_q["p95"] - day90_q["p05"]),
        "第180天P50": p50,
        "第180天P95": p95,
        "第180天P95-P05": day180_width,
        "突破120概率": p120,
        "突破130概率": p130,
        "外推期最高价P95": float(metrics["外推期最高价"].quantile(0.95)),
        "升级状态平均占比": float(metrics["升级占比"].mean()),
        "第180天最终升级占比": float((metrics["最终状态"] == "升级").mean()),
        "综合评分": score,
        "说明": variant.note,
    }


def run_evaluation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for variant in VARIANTS:
        _, quantiles, metrics = simulate_variant(variant)
        rows.append(summarize_variant(variant, quantiles, metrics))
        outputs[variant.name] = (quantiles, metrics)
    evaluation = pd.DataFrame(rows).sort_values("综合评分", ascending=False).reset_index(drop=True)
    ensure_parent(EVALUATION_CSV)
    evaluation.to_csv(EVALUATION_CSV, index=False)
    best_name = str(evaluation.iloc[0]["候选"])
    best_quantiles, best_metrics = outputs[best_name]
    best_quantiles.to_csv(BEST_QUANTILE_CSV, index=False)
    best_metrics.to_csv(BEST_METRICS_CSV, index=False)
    return evaluation, best_quantiles, best_metrics


def save_figure(evaluation: pd.DataFrame) -> None:
    configure_plot_style(savefig_dpi=220, figure_dpi=150, title_size=12)
    ensure_parent(FIGURE_PATH)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    labels = evaluation["候选"].tolist()
    x = np.arange(len(labels))
    axes[0].bar(x, evaluation["第180天P95-P05"], color=SCENARIO_COLORS["neutral"], alpha=0.82)
    axes[0].axhline(8.79, color=SCENARIO_COLORS["muted"], linestyle="--", linewidth=1.4, label="当前正式输出")
    axes[0].set_title("第180天概率区间宽度")
    axes[0].set_ylabel("P95-P05")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=24, ha="right")
    axes[0].legend(loc="upper right")

    axes[1].scatter(
        evaluation["突破120概率"],
        evaluation["突破130概率"],
        s=90,
        color=SCENARIO_COLORS["risk"],
        alpha=0.78,
    )
    for _, row in evaluation.iterrows():
        axes[1].annotate(str(row["候选"]).replace("_", "\n"), (row["突破120概率"], row["突破130概率"]), xytext=(5, 2), textcoords="offset points", fontsize=8)
    axes[1].set_title("尾部风险概率")
    axes[1].set_xlabel("突破120概率")
    axes[1].set_ylabel("突破130概率")
    axes[1].set_xlim(0, max(0.55, float(evaluation["突破120概率"].max()) + 0.04))
    axes[1].set_ylim(0, max(0.12, float(evaluation["突破130概率"].max()) + 0.02))
    fig.tight_layout()
    fig.savefig(FIGURE_PATH)
    plt.close(fig)


def write_report(evaluation: pd.DataFrame) -> None:
    ensure_parent(REPORT_PATH)
    rows = "\n".join(
        "| {name} | {matrix} | {p50:.2f} | {width:.2f} | {p120:.1%} | {p130:.1%} | {score:.3f} |".format(
            name=str(row["候选"]),
            matrix=str(row["使用矩阵"]),
            p50=float(row["第180天P50"]),
            width=float(row["第180天P95-P05"]),
            p120=float(row["突破120概率"]),
            p130=float(row["突破130概率"]),
            score=float(row["综合评分"]),
        )
        for _, row in evaluation.iterrows()
    )
    best = evaluation.iloc[0]
    report = f"""# 长期转移矩阵替换评估报告

## 评估目的

上一轮历史校准发现，高波动条件下“维持→升级”和“升级→升级”的经验概率高于当前基础矩阵。本报告把不同历史矩阵候选实际接入长期情景树，检查它们是否能提高尾部风险表达，同时避免把长期预测推得过激。

## 候选结果

| 候选 | 使用矩阵 | 第180天P50 | 第180天P95-P05 | 突破120概率 | 突破130概率 | 综合评分 |
|---|---|---:|---:|---:|---:|---:|
{rows}

## 当前判断

综合评分最高的是 **{best["候选"]}**。它使用 `{best["使用矩阵"]}`，第180天 P50 为 {float(best["第180天P50"]):.2f}，P95-P05 为 {float(best["第180天P95-P05"]):.2f}，突破120概率为 {float(best["突破120概率"]):.1%}，突破130概率为 {float(best["突破130概率"]):.1%}。

这说明历史高波动矩阵不能无脑全量接入，因为它会显著抬高升级状态；更合理的做法是把历史矩阵作为“概率上界”和“尾部保持依据”，再通过风险扰动倍率和后期缓和倍率做折中。若后续替换正式长期模型，优先考虑本报告排名第一的候选，而不是直接使用纯历史高波动矩阵。

## 输出产物

- `{EVALUATION_CSV.relative_to(PROJECT_ROOT)}`
- `{BEST_QUANTILE_CSV.relative_to(PROJECT_ROOT)}`
- `{BEST_METRICS_CSV.relative_to(PROJECT_ROOT)}`
- `{FIGURE_PATH.relative_to(PROJECT_ROOT)}`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    evaluation, _, _ = run_evaluation()
    save_figure(evaluation)
    write_report(evaluation)
    print("Long-term transition matrix evaluation complete")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
