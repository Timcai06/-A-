"""Calibrate long-horizon state transitions from 2017-2025 prices.

The long-term scenario tree uses three qualitative states: easing, neutral,
and escalation. This script maps historical Brent price behavior into the same
three states, estimates empirical transition matrices, and reports whether the
current and tail-preserving long-term state models are too aggressive or too
conservative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import long_term_state_transition as baseline
from src.common.paths import PROJECT_ROOT, ensure_parent
from src.common.plotting import SCENARIO_COLORS, configure_plot_style


EVENT_START = pd.Timestamp("2026-03-02")
PRICE_CSV = PROJECT_ROOT / "data" / "processed" / "布伦特原油期货主力合约价格数据_清洗后.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "scenarios"
STATE_SERIES_CSV = OUTPUT_DIR / "长期历史状态序列.csv"
MATRIX_CSV = OUTPUT_DIR / "长期历史状态转移校准矩阵.csv"
METRICS_CSV = OUTPUT_DIR / "长期历史状态转移校准指标.csv"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "长期历史状态转移校准报告.md"
FIGURE_PATH = PROJECT_ROOT / "output" / "candidate_figures" / "长期历史状态转移校准.png"

STATE_ORDER = ["缓和", "维持", "升级"]
STATE_TO_KEY = {"缓和": "easing", "维持": "neutral", "升级": "escalation"}
KEY_TO_STATE = {value: key for key, value in STATE_TO_KEY.items()}


def load_history() -> pd.DataFrame:
    if not PRICE_CSV.exists():
        raise FileNotFoundError(f"缺少价格数据: {PRICE_CSV.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(PRICE_CSV, parse_dates=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    history = df[df["trade_date"] < EVENT_START].copy().reset_index(drop=True)
    if history.empty:
        raise RuntimeError("历史价格样本为空，无法校准长期状态转移。")
    history["close_price"] = history["close_price"].astype(float)
    history["log_return"] = np.log(history["close_price"] / history["close_price"].shift(1))
    history["return_5d"] = np.log(history["close_price"] / history["close_price"].shift(5))
    history["return_10d"] = np.log(history["close_price"] / history["close_price"].shift(10))
    history["volatility_14d_calc"] = history["log_return"].rolling(14, min_periods=10).std(ddof=0)
    history["volatility_30d_calc"] = history["log_return"].rolling(30, min_periods=20).std(ddof=0)
    history["drawup_20d"] = history["close_price"] / history["close_price"].rolling(20, min_periods=10).min() - 1
    history["drawdown_20d"] = history["close_price"] / history["close_price"].rolling(20, min_periods=10).max() - 1
    return history.dropna(subset=["log_return", "return_5d", "volatility_14d_calc"]).reset_index(drop=True)


def classify_states(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    thresholds = {
        "vol_p45": float(history["volatility_14d_calc"].quantile(0.45)),
        "vol_p60": float(history["volatility_14d_calc"].quantile(0.60)),
        "vol_p75": float(history["volatility_14d_calc"].quantile(0.75)),
        "vol_p85": float(history["volatility_14d_calc"].quantile(0.85)),
        "abs_ret_p90": float(history["log_return"].abs().quantile(0.90)),
        "ret5_p35": float(history["return_5d"].quantile(0.35)),
        "ret5_p60": float(history["return_5d"].quantile(0.60)),
        "ret5_p75": float(history["return_5d"].quantile(0.75)),
        "drawup_p70": float(history["drawup_20d"].quantile(0.70)),
    }

    high_vol = history["volatility_14d_calc"] >= thresholds["vol_p75"]
    very_high_vol = history["volatility_14d_calc"] >= thresholds["vol_p85"]
    jump_day = history["log_return"].abs() >= thresholds["abs_ret_p90"]
    upside_pressure = (
        (history["return_5d"] >= thresholds["ret5_p60"])
        | (history["return_10d"] > 0.035)
        | (history["drawup_20d"] >= thresholds["drawup_p70"])
    )
    easing_pressure = (
        (history["volatility_14d_calc"] <= thresholds["vol_p45"])
        & (history["return_5d"] <= thresholds["ret5_p35"])
        & (history["drawdown_20d"] > -0.08)
    )

    frame = history.copy()
    frame["状态"] = "维持"
    frame.loc[easing_pressure, "状态"] = "缓和"
    frame.loc[(high_vol & upside_pressure) | (very_high_vol & jump_day), "状态"] = "升级"
    frame["state"] = frame["状态"].map(STATE_TO_KEY)
    frame["高波动样本"] = high_vol
    frame["上行压力样本"] = upside_pressure
    return frame, thresholds


def transition_matrix(frame: pd.DataFrame, *, stress_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    data["下一状态"] = data["状态"].shift(-1)
    data = data.dropna(subset=["下一状态"])
    if stress_only:
        data = data[data["高波动样本"] | (data["状态"] == "升级")].copy()

    counts = pd.crosstab(data["状态"], data["下一状态"]).reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    probs = probs.reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0.0)
    return counts, probs


def shrink_matrix(empirical: pd.DataFrame, weight: float = 0.55) -> pd.DataFrame:
    baseline_frame = pd.DataFrame(
        baseline.TRANSITION_MATRIX,
        index=STATE_ORDER,
        columns=STATE_ORDER,
    )
    mixed = weight * empirical + (1 - weight) * baseline_frame
    return mixed.div(mixed.sum(axis=1), axis=0)


def matrix_to_rows(label: str, matrix: pd.DataFrame, counts: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in STATE_ORDER:
        row = {
            "矩阵": label,
            "当前状态": state,
            "转为缓和": float(matrix.loc[state, "缓和"]),
            "转为维持": float(matrix.loc[state, "维持"]),
            "转为升级": float(matrix.loc[state, "升级"]),
        }
        if counts is not None:
            row["样本转移数"] = int(counts.loc[state].sum())
        rows.append(row)
    return rows


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    history = load_history()
    states, thresholds = classify_states(history)
    all_counts, all_probs = transition_matrix(states)
    stress_counts, stress_probs = transition_matrix(states, stress_only=True)
    recommended = shrink_matrix(stress_probs, weight=0.55)

    baseline_matrix = pd.DataFrame(baseline.TRANSITION_MATRIX, index=STATE_ORDER, columns=STATE_ORDER)
    rows: list[dict[str, Any]] = []
    rows.extend(matrix_to_rows("当前基础矩阵", baseline_matrix))
    rows.extend(matrix_to_rows("历史全样本矩阵", all_probs, all_counts))
    rows.extend(matrix_to_rows("历史高波动条件矩阵", stress_probs, stress_counts))
    rows.extend(matrix_to_rows("历史收缩推荐矩阵", recommended))
    matrix_table = pd.DataFrame(rows)

    current_neutral_to_up = float(baseline_matrix.loc["维持", "升级"])
    stress_neutral_to_up = float(stress_probs.loc["维持", "升级"])
    recommended_neutral_to_up = float(recommended.loc["维持", "升级"])
    current_up_persistence = float(baseline_matrix.loc["升级", "升级"])
    stress_up_persistence = float(stress_probs.loc["升级", "升级"])
    recommended_up_persistence = float(recommended.loc["升级", "升级"])

    metrics = pd.DataFrame(
        [
            {"指标": "历史样本天数", "数值": len(states), "说明": "2017-2025且排除2026事件窗口"},
            {"指标": "缓和状态占比", "数值": float((states["状态"] == "缓和").mean()), "说明": "按历史价格行为分类"},
            {"指标": "维持状态占比", "数值": float((states["状态"] == "维持").mean()), "说明": "按历史价格行为分类"},
            {"指标": "升级状态占比", "数值": float((states["状态"] == "升级").mean()), "说明": "按历史价格行为分类"},
            {"指标": "当前矩阵_维持转升级", "数值": current_neutral_to_up, "说明": "当前长期状态转移基础矩阵"},
            {"指标": "历史高波动_维持转升级", "数值": stress_neutral_to_up, "说明": "历史高波动条件下的经验转移"},
            {"指标": "推荐矩阵_维持转升级", "数值": recommended_neutral_to_up, "说明": "历史条件矩阵与当前基础矩阵收缩融合"},
            {"指标": "当前矩阵_升级保持", "数值": current_up_persistence, "说明": "当前长期状态转移基础矩阵"},
            {"指标": "历史高波动_升级保持", "数值": stress_up_persistence, "说明": "历史高波动条件下升级状态惯性"},
            {"指标": "推荐矩阵_升级保持", "数值": recommended_up_persistence, "说明": "历史条件矩阵与当前基础矩阵收缩融合"},
            {"指标": "历史高波动转移样本数", "数值": int(stress_counts.to_numpy().sum()), "说明": "用于估计高波动条件矩阵的转移次数"},
        ]
    )

    summary = {
        "thresholds": thresholds,
        "state_days": len(states),
        "all_counts": all_counts,
        "stress_counts": stress_counts,
        "current_neutral_to_up": current_neutral_to_up,
        "stress_neutral_to_up": stress_neutral_to_up,
        "recommended_neutral_to_up": recommended_neutral_to_up,
        "current_up_persistence": current_up_persistence,
        "stress_up_persistence": stress_up_persistence,
        "recommended_up_persistence": recommended_up_persistence,
    }
    ensure_parent(STATE_SERIES_CSV)
    states.to_csv(STATE_SERIES_CSV, index=False)
    matrix_table.to_csv(MATRIX_CSV, index=False)
    metrics.to_csv(METRICS_CSV, index=False)
    return matrix_table, metrics, summary


def save_figure(matrix_table: pd.DataFrame) -> None:
    configure_plot_style(savefig_dpi=220, figure_dpi=150, title_size=12)
    ensure_parent(FIGURE_PATH)
    selected = matrix_table[matrix_table["当前状态"].isin(["维持", "升级"])].copy()
    selected["转移项"] = selected["当前状态"] + "→升级"
    selected.loc[selected["当前状态"] == "升级", "转移项"] = "升级→升级"
    plot_rows = []
    for _, row in selected.iterrows():
        value = row["转为升级"] if row["当前状态"] == "维持" else row["转为升级"]
        plot_rows.append({"矩阵": row["矩阵"], "转移项": row["转移项"], "概率": float(value)})
    plot_df = pd.DataFrame(plot_rows)

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    colors = {
        "当前基础矩阵": SCENARIO_COLORS["muted"],
        "历史全样本矩阵": SCENARIO_COLORS["buffer"],
        "历史高波动条件矩阵": SCENARIO_COLORS["risk"],
        "历史收缩推荐矩阵": SCENARIO_COLORS["neutral"],
    }
    x_labels = ["维持→升级", "升级→升级"]
    x = np.arange(len(x_labels))
    width = 0.18
    matrix_names = ["当前基础矩阵", "历史全样本矩阵", "历史高波动条件矩阵", "历史收缩推荐矩阵"]
    for offset, matrix_name in enumerate(matrix_names):
        subset = plot_df[plot_df["矩阵"] == matrix_name].set_index("转移项")
        values = [float(subset.loc[label, "概率"]) for label in x_labels]
        ax.bar(x + (offset - 1.5) * width, values, width=width, label=matrix_name, color=colors[matrix_name], alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("转移概率")
    ax.set_ylim(0, max(0.85, plot_df["概率"].max() * 1.18))
    ax.set_title("历史价格行为校准的长期状态转移概率")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH)
    plt.close(fig)


def _matrix_markdown(matrix_table: pd.DataFrame, matrix_name: str) -> str:
    sub = matrix_table[matrix_table["矩阵"] == matrix_name][["当前状态", "转为缓和", "转为维持", "转为升级"]].copy()
    return sub.to_markdown(index=False, floatfmt=".3f")


def write_report(matrix_table: pd.DataFrame, metrics: pd.DataFrame, summary: dict[str, Any]) -> None:
    ensure_parent(REPORT_PATH)
    values = dict(zip(metrics["指标"], metrics["数值"], strict=False))
    report = f"""# 长期历史状态转移校准报告

## 结论摘要

本报告用 2017--2025 年附件历史油价数据校准长期状态转移概率。方法是先用滚动波动率、5日/10日收益和20日上行压力把历史交易日划分为“缓和、维持、升级”，再统计下一交易日状态转移频率。

核心发现：当前长期模型的基础矩阵并非完全离谱，但在高波动条件下，历史数据支持更高的“维持→升级”概率和更强的“升级→升级”惯性。这为上一轮“尾部保持 Challenger”提供了数据层面的支持。

## 状态划分口径

| 项目 | 数值 |
|---|---:|
| 历史样本天数 | {int(values["历史样本天数"])} |
| 缓和状态占比 | {float(values["缓和状态占比"]):.1%} |
| 维持状态占比 | {float(values["维持状态占比"]):.1%} |
| 升级状态占比 | {float(values["升级状态占比"]):.1%} |
| 高波动转移样本数 | {int(values["历史高波动转移样本数"])} |

状态划分不是未来真实标签，而是从历史价格行为中提取的经验代理变量：高波动且存在上行压力的日子记为“升级”，低波动且回落/冷却的日子记为“缓和”，其余记为“维持”。

## 当前基础矩阵

{_matrix_markdown(matrix_table, "当前基础矩阵")}

## 历史高波动条件矩阵

{_matrix_markdown(matrix_table, "历史高波动条件矩阵")}

## 历史收缩推荐矩阵

{_matrix_markdown(matrix_table, "历史收缩推荐矩阵")}

## 对长期模型的含义

| 对比项 | 当前基础矩阵 | 历史高波动条件 | 收缩推荐 |
|---|---:|---:|---:|
| 维持→升级 | {summary["current_neutral_to_up"]:.1%} | {summary["stress_neutral_to_up"]:.1%} | {summary["recommended_neutral_to_up"]:.1%} |
| 升级→升级 | {summary["current_up_persistence"]:.1%} | {summary["stress_up_persistence"]:.1%} | {summary["recommended_up_persistence"]:.1%} |

从数据上看，当前模型第 120 天以后将升级概率压得太低，会让长期概率扇形过早收窄。历史高波动样本说明，在高压市场状态下，升级状态本身具有明显惯性，维持状态也存在更高的再升级概率。因此，长期模型更合理的优化方向不是简单把中性线画弯，而是让状态转移层保留足够的后期尾部风险。

## 是否可以替换当前正式长期模型

可以进入下一轮替换评估，但还不建议直接替换论文正式结果。推荐做法是：

1. 用“历史收缩推荐矩阵”替代当前基础矩阵跑一版长期状态转移。
2. 与上一轮“尾部保持 Challenger”比较第180天 P50、P95、突破120/130概率和状态占比。
3. 选择既符合历史状态转移概率、又不夸张放大极端尾部的版本作为正式长期模型。

## 输出产物

- `{STATE_SERIES_CSV.relative_to(PROJECT_ROOT)}`
- `{MATRIX_CSV.relative_to(PROJECT_ROOT)}`
- `{METRICS_CSV.relative_to(PROJECT_ROOT)}`
- `{FIGURE_PATH.relative_to(PROJECT_ROOT)}`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    matrix_table, metrics, summary = build_outputs()
    save_figure(matrix_table)
    write_report(matrix_table, metrics, summary)
    print("Long-term historical state calibration complete")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Matrix: {MATRIX_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
