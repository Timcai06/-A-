"""Diagnose the long-horizon scenario model shape and weak spots.

This module does not replace the official long-term model. It reads the
current scenario and state-transition outputs, then writes an engineering
diagnostic report about smoothness, tail persistence, scenario spread, and
where the next model upgrade should focus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parent
from src.common.plotting import SCENARIO_COLORS, configure_plot_style


SCENARIO_RESULT_CSV = PROJECT_ROOT / "output" / "scenarios" / "三情景预测结果.csv"
SCENARIO_METRICS_CSV = PROJECT_ROOT / "output" / "scenarios" / "三情景关键指标.csv"
STATE_QUANTILES_CSV = PROJECT_ROOT / "output" / "scenarios" / "长期状态转移路径分位数.csv"
STATE_SAMPLE_METRICS_CSV = PROJECT_ROOT / "output" / "scenarios" / "长期状态转移样本指标.csv"
STATE_SHARE_CSV = PROJECT_ROOT / "output" / "scenarios" / "长期状态转移状态占比.csv"
TRANSITION_SNAPSHOT_CSV = PROJECT_ROOT / "output" / "scenarios" / "长期状态转移时变矩阵快照.csv"
RISK_ABLATION_CSV = PROJECT_ROOT / "output" / "scenarios" / "长期状态转移风险约束消融.csv"

DIAGNOSTIC_CSV = PROJECT_ROOT / "output" / "scenarios" / "长期模型形态诊断指标.csv"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "长期模型形态诊断报告.md"
FIGURE_PATH = PROJECT_ROOT / "output" / "candidate_figures" / "长期模型形态诊断.png"

MARKER_DAYS = [90, 120, 150, 180]


def _read_csv(path: Path, *, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"缺少长期模型诊断输入: {path.relative_to(PROJECT_ROOT)}")
    return pd.read_csv(path, parse_dates=parse_dates)


def scenario_shape_metrics(scenario_result: pd.DataFrame) -> pd.DataFrame:
    future = scenario_result[~scenario_result["is_observed_price"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for scenario_key, group in future.groupby("scenario", sort=False):
        group = group.sort_values("day_index").copy()
        diff = group["forecast_price"].diff().dropna()
        start_price = float(group.iloc[0]["forecast_price"])
        final_price = float(group[group["day_index"] == 180]["forecast_price"].iloc[0])
        rows.append(
            {
                "指标组": "三情景中心路径",
                "对象": scenario_key,
                "起点价格": start_price,
                "第180天价格": final_price,
                "外推期净变化": final_price - start_price,
                "日均变化": float(diff.mean()),
                "日变化标准差": float(diff.std(ddof=0)),
                "平均绝对日变化": float(diff.abs().mean()),
                "最大单日上行": float(diff.max()),
                "最大单日下行": float(diff.min()),
            }
        )
    return pd.DataFrame(rows)


def quantile_shape_metrics(quantiles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for day in MARKER_DAYS:
        row = quantiles[quantiles["day_index"] == day].iloc[0]
        rows.append(
            {
                "指标组": "状态转移概率扇形",
                "对象": f"第{day}天",
                "p05": float(row["p05"]),
                "p50": float(row["p50"]),
                "p95": float(row["p95"]),
                "p95-p05": float(row["p95"] - row["p05"]),
                "p75-p25": float(row["p75"] - row["p25"]),
                "p95相对p50": float(row["p95"] - row["p50"]),
                "p50相对p05": float(row["p50"] - row["p05"]),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) >= 2:
        first_width = float(frame.iloc[0]["p95-p05"])
        final_width = float(frame.iloc[-1]["p95-p05"])
        frame["相对第90天宽度"] = frame["p95-p05"] / max(first_width, 1e-9)
        frame.loc[frame.index[-1], "第90至180天宽度变化率"] = final_width / max(first_width, 1e-9) - 1
    return frame


def state_share_at_markers(state_share: pd.DataFrame) -> pd.DataFrame:
    pivot = state_share.pivot_table(index="day_index", columns="状态", values="状态占比", fill_value=0.0)
    rows: list[dict[str, Any]] = []
    for day in MARKER_DAYS:
        if day not in pivot.index:
            continue
        rows.append(
            {
                "指标组": "状态占比",
                "对象": f"第{day}天",
                "缓和占比": float(pivot.loc[day].get("缓和", 0.0)),
                "维持占比": float(pivot.loc[day].get("维持", 0.0)),
                "升级占比": float(pivot.loc[day].get("升级", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_diagnostics() -> tuple[pd.DataFrame, dict[str, Any]]:
    scenario_result = _read_csv(SCENARIO_RESULT_CSV, parse_dates=["trade_date"])
    scenario_metrics = _read_csv(SCENARIO_METRICS_CSV)
    quantiles = _read_csv(STATE_QUANTILES_CSV, parse_dates=["trade_date"])
    sample_metrics = _read_csv(STATE_SAMPLE_METRICS_CSV)
    state_share = _read_csv(STATE_SHARE_CSV, parse_dates=["trade_date"])
    transition_snapshots = _read_csv(TRANSITION_SNAPSHOT_CSV)
    risk_ablation = _read_csv(RISK_ABLATION_CSV)

    scenario_shape = scenario_shape_metrics(scenario_result)
    quantile_shape = quantile_shape_metrics(quantiles)
    state_shape = state_share_at_markers(state_share)

    long_table = pd.concat([scenario_shape, quantile_shape, state_shape], ignore_index=True, sort=False)
    ensure_parent(DIAGNOSTIC_CSV)
    long_table.to_csv(DIAGNOSTIC_CSV, index=False)

    neutral_metrics = scenario_metrics[scenario_metrics["scenario"] == "neutral"].iloc[0]
    pessimistic_metrics = scenario_metrics[scenario_metrics["scenario"] == "pessimistic"].iloc[0]
    final_quantile = quantiles[quantiles["day_index"] == 180].iloc[0]
    day90_quantile = quantiles[quantiles["day_index"] == 90].iloc[0]
    final_share = state_shape[state_shape["对象"] == "第180天"].iloc[0]
    day90_share = state_shape[state_shape["对象"] == "第90天"].iloc[0]
    neutral_transition_180 = transition_snapshots[
        (transition_snapshots["当前状态"] == "维持") & (transition_snapshots["day_index"] == 180)
    ].iloc[0]
    base_mask = risk_ablation["模型设定"].astype(str).str.contains("基础矩阵") & ~risk_ablation["模型设定"].astype(str).str.contains("GPR|OVX", regex=True)
    risk_mask = risk_ablation["模型设定"].astype(str).str.contains("GPR|OVX", regex=True)
    ablation_base = risk_ablation[base_mask].iloc[0]
    ablation_risk = risk_ablation[risk_mask].iloc[0]

    summary = {
        "neutral_day180": float(neutral_metrics["第180天价格"]),
        "pessimistic_day180": float(pessimistic_metrics["第180天价格"]),
        "neutral_mean": float(neutral_metrics["外推期均价"]),
        "pessimistic_peak": float(pessimistic_metrics["外推期最高价"]),
        "p50_day180": float(final_quantile["p50"]),
        "p95_day180": float(final_quantile["p95"]),
        "width_day90": float(day90_quantile["p95"] - day90_quantile["p05"]),
        "width_day180": float(final_quantile["p95"] - final_quantile["p05"]),
        "easing_day90": float(day90_share["缓和占比"]),
        "escalation_day90": float(day90_share["升级占比"]),
        "easing_day180": float(final_share["缓和占比"]),
        "escalation_day180": float(final_share["升级占比"]),
        "neutral_to_escalation_day180": float(neutral_transition_180["转为升级"]),
        "risk_p120_lift": float(ablation_risk["突破120概率"] - ablation_base["突破120概率"]),
        "risk_p130_lift": float(ablation_risk["突破130概率"] - ablation_base["突破130概率"]),
        "p120_prob": float(sample_metrics["突破120"].mean()),
        "p130_prob": float(sample_metrics["突破130"].mean()),
        "peak_p95": float(sample_metrics["外推期最高价"].quantile(0.95)),
    }
    return long_table, summary


def save_diagnostic_figure() -> None:
    scenario_result = _read_csv(SCENARIO_RESULT_CSV, parse_dates=["trade_date"])
    quantiles = _read_csv(STATE_QUANTILES_CSV, parse_dates=["trade_date"])
    state_share = _read_csv(STATE_SHARE_CSV, parse_dates=["trade_date"])

    configure_plot_style(savefig_dpi=220, figure_dpi=150, title_size=12)
    ensure_parent(FIGURE_PATH)
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 9.6), sharex=False)

    future = scenario_result[~scenario_result["is_observed_price"].astype(bool)].copy()
    labels = {"optimistic": "乐观中心", "neutral": "中性中心", "pessimistic": "悲观中心"}
    colors = {
        "optimistic": SCENARIO_COLORS["optimistic"],
        "neutral": SCENARIO_COLORS["neutral"],
        "pessimistic": SCENARIO_COLORS["pessimistic"],
    }
    for scenario_key, group in future.groupby("scenario", sort=False):
        group = group.sort_values("day_index")
        axes[0].plot(group["day_index"], group["forecast_price"], label=labels.get(scenario_key, scenario_key), color=colors[scenario_key], linewidth=2.2)
    axes[0].set_title("三情景中心路径形态")
    axes[0].set_ylabel("美元/桶")
    axes[0].legend(loc="upper right", ncol=3)

    width_outer = quantiles["p95"] - quantiles["p05"]
    width_inner = quantiles["p75"] - quantiles["p25"]
    axes[1].plot(quantiles["day_index"], width_outer, color=SCENARIO_COLORS["risk"], linewidth=2.3, label="P95-P05")
    axes[1].plot(quantiles["day_index"], width_inner, color=SCENARIO_COLORS["buffer"], linewidth=2.0, label="P75-P25")
    axes[1].set_title("概率扇形宽度变化")
    axes[1].set_ylabel("区间宽度")
    axes[1].legend(loc="upper right")

    pivot = state_share.pivot_table(index="day_index", columns="状态", values="状态占比", fill_value=0.0).sort_index()
    axes[2].stackplot(
        pivot.index,
        [pivot.get(label, pd.Series(0.0, index=pivot.index)) for label in ["缓和", "维持", "升级"]],
        labels=["缓和", "维持", "升级"],
        colors=[SCENARIO_COLORS["optimistic"], SCENARIO_COLORS["neutral"], SCENARIO_COLORS["pessimistic"]],
        alpha=0.72,
    )
    axes[2].set_title("状态占比演化")
    axes[2].set_ylabel("占比")
    axes[2].set_xlabel("封锁后天数")
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc="upper right", ncol=3)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH)
    plt.close(fig)


def write_report(summary: dict[str, Any]) -> None:
    ensure_parent(REPORT_PATH)
    width_change = summary["width_day180"] / summary["width_day90"] - 1
    report = f"""# 长期模型形态诊断报告

## 诊断结论

当前长期模型不是“逐日精确预测器”，而是由三情景物理中心路径和状态转移概率扇形组成的条件预测框架。它的优点是结构清楚、外部风险变量能够抬升尾部风险；主要短板是后期风险收敛偏快，导致中位线和 180 天区间看起来偏平、偏低。

## 关键证据

| 诊断项 | 当前数值 | 含义 |
|---|---:|---|
| 中性情景第180天价格 | {summary["neutral_day180"]:.2f} | 三情景物理中心线的中性结果 |
| 悲观情景第180天价格 | {summary["pessimistic_day180"]:.2f} | 封锁延续或恢复不顺时的高位中心 |
| 状态转移第180天P50 | {summary["p50_day180"]:.2f} | 概率路径中位数，低于中性中心 |
| 状态转移第180天P95 | {summary["p95_day180"]:.2f} | 后期上尾区间上沿 |
| 第90天P95-P05宽度 | {summary["width_day90"]:.2f} | 早中期不确定性仍较宽 |
| 第180天P95-P05宽度 | {summary["width_day180"]:.2f} | 后期概率扇形明显收窄 |
| 第90至180天宽度变化率 | {width_change:.1%} | 宽度收缩幅度 |
| 第90天升级状态占比 | {summary["escalation_day90"]:.1%} | 早中期仍有尾部冲击 |
| 第180天升级状态占比 | {summary["escalation_day180"]:.1%} | 后期升级状态保留不足 |
| 第180天“维持→升级”概率 | {summary["neutral_to_escalation_day180"]:.1%} | 后期再升级概率已很低 |
| GPR/OVX对突破120概率提升 | {summary["risk_p120_lift"]:.1%} | 外部风险变量确实抬升尾部风险 |
| GPR/OVX对突破130概率提升 | {summary["risk_p130_lift"]:.1%} | 对极端尾部有温和影响 |

## 主要问题

1. **后期状态过度缓和。** 第 180 天缓和状态占比达到 {summary["easing_day180"]:.1%}，升级状态只有 {summary["escalation_day180"]:.1%}。这解释了为什么概率中位线比中性中心线更低。
2. **概率扇形收缩过快。** 第 90 天 P95-P05 宽度为 {summary["width_day90"]:.2f}，第 180 天降到 {summary["width_day180"]:.2f}，长期尾部风险被较快压扁。
3. **中性线不是训练出来的点预测。** 它来自供给恢复、SPR收缩、需求弹性恢复和风险溢价衰减的机制递推，因此平滑是模型结构结果，不代表它可以精确预测未来每一天。
4. **外部风险变量作用偏保守。** GPR/OVX 将突破 120 概率提升 {summary["risk_p120_lift"]:.1%}，说明方向有效，但对第 180 天上尾价格的支撑仍然有限。

## 下一轮真正优化方向

1. **尾部保持型状态转移矩阵。** 把第 120 天和第 150 天后的机械缓和修正改成风险条件触发：只有 OVX/GPR 或供需缺口同步回落时，才快速提高缓和概率。
2. **随机供给恢复路径。** 不再让绕道、SPR、需求收缩完全沿确定曲线走，而是给恢复速度、库存释放效率和政策可信度加随机分布，提升长期路径真实波动。
3. **库存耗尽二次跳涨分支。** 当商业库存或政策释放能力下降到阈值后，允许悲观状态出现二次跳涨，而不是只沿悲观中心线平滑回落。
4. **用历史高波动片段校准转移概率。** 利用 2017--2025 原油价格数据筛选高波动/高跳涨窗口，估计“维持→升级”“升级→缓和”的经验概率范围。
5. **期货期限结构或OVX持续项接入状态层。** 如果未来拿到多期限期货价格，可用近远月价差约束长期中枢；OVX则适合控制概率扇形宽度和升级概率。

## 输出产物

- `{DIAGNOSTIC_CSV.relative_to(PROJECT_ROOT)}`
- `{FIGURE_PATH.relative_to(PROJECT_ROOT)}`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    _, summary = build_diagnostics()
    save_diagnostic_figure()
    write_report(summary)
    print("Long-term diagnostics complete")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figure: {FIGURE_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
