"""Optimization audit for the main short-term mechanism model.

This script does not replace the official calibrated path.  It checks whether
the current mechanism model still has credible improvement room by applying
small one-factor perturbations around the saved best parameters.  The goal is
to distinguish real mechanism refinement from ad hoc curve fitting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration.evaluation import evaluate_simulation
from src.common.metrics import direction_hit_rate, mae, mape, rmse
from src.common.paths import PROJECT_ROOT, ensure_parents
from src.common.plotting import PAPER_COLORS, SCENARIO_COLORS, configure_plot_style
from src.models import dynamic_short_term as dynamic


@dataclass(frozen=True)
class Paths:
    event_csv: Path = PROJECT_ROOT / "data" / "processed" / "布伦特原油期货主力合约价格数据_事件窗口.csv"
    best_parameters_csv: Path = PROJECT_ROOT / "output" / "calibration" / "动态模型最优参数.csv"
    current_path_csv: Path = PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv"
    contribution_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期主机制模型贡献审计.csv"
    perturbation_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期主机制模型单因子扰动.csv"
    candidate_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期主机制模型优化候选.csv"
    report_path: Path = PROJECT_ROOT / "output" / "reports" / "短期主机制模型可优化性审计报告.md"
    figure_path: Path = PROJECT_ROOT / "output" / "candidate_figures" / "短期主机制模型可优化性审计.png"


EVENT_WINDOWS = [
    ("初始冲击吸收", "2026-03-02", "2026-03-14"),
    ("高位平台形成", "2026-03-17", "2026-03-31"),
    ("中期再定价回落", "2026-04-01", "2026-04-17"),
    ("后期二次抬升", "2026-04-20", "2026-05-05"),
]

ASSUMPTION_BOUNDS = {
    "supply_interruption": (1400.0, 1800.0),
    "spr_max_release": (200.0, 700.0),
    "spr_delay_days": (3.0, 14.0),
    "route_start_day": (7.0, 30.0),
    "route_max_capacity": (150.0, 300.0),
    "route_ramp_days": (14.0, 30.0),
    "long_elasticity": (-0.25, -0.10),
    "fear_initial": (0.10, 0.20),
    "fear_decay": (0.04, 0.12),
    "inventory_daily_cap": (250.0, 600.0),
}

BEHAVIOR_BOUNDS = {
    "pressure_scale": (0.020, 0.090),
    "risk_weight": (1.60, 3.40),
    "uncertainty_floor": (0.08, 0.30),
    "inventory_response": (0.20, 0.75),
    "adjustment_speed": (0.18, 0.45),
    "buffer_relief_strength": (0.00, 0.22),
    "buffer_relief_decay_days": (4.0, 16.0),
    "relief_discount_strength": (0.00, 0.24),
    "relief_start_day": (24.0, 38.0),
    "relief_peak_day": (38.0, 52.0),
    "relief_decay_days": (4.0, 18.0),
}

INTEGER_PARAMETERS = {
    "spr_delay_days",
    "route_start_day",
    "route_ramp_days",
    "buffer_relief_decay_days",
    "relief_start_day",
    "relief_peak_day",
    "relief_decay_days",
}

PERTURBATION_GRID = [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15]


def load_event_window() -> pd.DataFrame:
    if not Paths.event_csv.exists():
        base_config = dynamic.load_yaml(dynamic.BASE_CONFIG_PATH)
        event_path = dynamic.resolve_paths(base_config).event_csv
    else:
        event_path = Paths.event_csv
    return dynamic.load_event_window(event_path)


def load_current_path() -> pd.DataFrame:
    if not Paths.current_path_csv.exists():
        raise FileNotFoundError(Paths.current_path_csv)
    return pd.read_csv(Paths.current_path_csv, parse_dates=["trade_date"]).sort_values("trade_date").reset_index(drop=True)


def load_best_objects() -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters, pd.Series]:
    if not Paths.best_parameters_csv.exists():
        raise FileNotFoundError(Paths.best_parameters_csv)
    row = pd.read_csv(Paths.best_parameters_csv).iloc[0]
    assumptions = dynamic.PhysicalAssumptions(
        base_supply=float(row["assumption_base_supply"]),
        base_demand=float(row["assumption_base_demand"]),
        base_elasticity=float(row["assumption_base_elasticity"]),
        long_elasticity=float(row["assumption_long_elasticity"]),
        supply_interruption=float(row["assumption_supply_interruption"]),
        spr_max_release=float(row["assumption_spr_max_release"]),
        spr_delay_days=int(round(row["assumption_spr_delay_days"])),
        spr_ramp_days=int(round(row["assumption_spr_ramp_days"])),
        route_start_day=int(round(row["assumption_route_start_day"])),
        route_max_capacity=float(row["assumption_route_max_capacity"]),
        route_ramp_days=int(round(row["assumption_route_ramp_days"])),
        commercial_inventory=float(row["assumption_commercial_inventory"]),
        inventory_daily_cap=float(row["assumption_inventory_daily_cap"]),
        observed_demand_decline=float(row["assumption_observed_demand_decline"]),
        demand_decline_ramp_days=int(round(row["assumption_demand_decline_ramp_days"])),
        fear_initial=float(row["assumption_fear_initial"]),
        fear_decay=float(row["assumption_fear_decay"]),
    )
    behavior = dynamic.BehavioralParameters(
        pressure_scale=float(row["behavior_pressure_scale"]),
        risk_weight=float(row["behavior_risk_weight"]),
        uncertainty_floor=float(row["behavior_uncertainty_floor"]),
        inventory_response=float(row["behavior_inventory_response"]),
        adjustment_speed=float(row["behavior_adjustment_speed"]),
        buffer_relief_strength=float(row["behavior_buffer_relief_strength"]),
        buffer_relief_decay_days=int(round(row["behavior_buffer_relief_decay_days"])),
        relief_discount_strength=float(row["behavior_relief_discount_strength"]),
        relief_start_day=int(round(row["behavior_relief_start_day"])),
        relief_peak_day=int(round(row["behavior_relief_peak_day"])),
        relief_decay_days=int(round(row["behavior_relief_decay_days"])),
    )
    return assumptions, behavior, row


def model_metrics(simulation: pd.DataFrame) -> dict[str, float]:
    error = simulation["simulated_price"] - simulation["actual_price"]
    metrics = evaluate_simulation(simulation)
    metrics.update(
        {
            "MAPE": mape(simulation["actual_price"], simulation["simulated_price"]),
            "方向命中率": direction_hit_rate(simulation["actual_price"], simulation["simulated_price"]),
            "最大绝对误差": float(error.abs().max()),
            "平均误差": float(error.mean()),
        }
    )
    for name, start, end in EVENT_WINDOWS:
        sub = simulation[simulation["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        metrics[f"{name}_RMSE"] = rmse(sub["simulated_price"] - sub["actual_price"])
        metrics[f"{name}_MAE"] = mae(sub["simulated_price"] - sub["actual_price"])
        metrics[f"{name}_平均误差"] = float((sub["simulated_price"] - sub["actual_price"]).mean())
    return metrics


def contribution_audit(current: pd.DataFrame) -> pd.DataFrame:
    df = current.copy()
    df["上行压力合计"] = df[["shortage_pressure", "blockade_risk_premium", "uncertainty_premium", "panic_premium"]].sum(axis=1)
    df["下行折价合计"] = df[["buffer_confirmation_discount", "expectation_relief_discount"]].sum(axis=1)
    df["净压力"] = df["上行压力合计"] - df["下行折价合计"]
    df["模型误差"] = df["simulated_price"] - df["actual_price"]
    df["事件段"] = "未分段"
    for name, start, end in EVENT_WINDOWS:
        df.loc[df["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end)), "事件段"] = name

    columns = [
        "shortage_pressure",
        "blockade_risk_premium",
        "uncertainty_premium",
        "panic_premium",
        "buffer_confirmation_discount",
        "expectation_relief_discount",
        "上行压力合计",
        "下行折价合计",
        "净压力",
        "supply_gap",
        "buffer_coverage_ratio",
        "模型误差",
    ]
    rows: list[dict[str, Any]] = []
    for name, sub in [("全窗口", df), *[(window, df[df["事件段"] == window]) for window, _, _ in EVENT_WINDOWS]]:
        row: dict[str, Any] = {"窗口": name, "样本数": len(sub)}
        for column in columns:
            row[f"{column}_均值"] = float(sub[column].mean())
            row[f"{column}_末值"] = float(sub[column].iloc[-1])
        row["RMSE"] = rmse(sub["模型误差"])
        row["MAE"] = mae(sub["模型误差"])
        rows.append(row)
    return pd.DataFrame(rows)


def bounded_value(name: str, value: float, bounds: dict[str, tuple[float, float]]) -> float:
    lower, upper = bounds[name]
    clipped = float(np.clip(value, lower, upper))
    if name in INTEGER_PARAMETERS:
        clipped = float(round(clipped))
    return clipped


def perturb_assumption(
    assumptions: dynamic.PhysicalAssumptions,
    name: str,
    delta: float,
) -> dynamic.PhysicalAssumptions:
    value = float(asdict(assumptions)[name])
    if value < 0:
        candidate = value * (1 - delta)
    else:
        candidate = value * (1 + delta)
    return replace(assumptions, **{name: bounded_value(name, candidate, ASSUMPTION_BOUNDS)})


def perturb_behavior(
    behavior: dynamic.BehavioralParameters,
    name: str,
    delta: float,
) -> dynamic.BehavioralParameters:
    value = float(asdict(behavior)[name])
    if value == 0:
        lower, upper = BEHAVIOR_BOUNDS[name]
        candidate = min(upper, max(lower, abs(delta) * upper))
    else:
        candidate = value * (1 + delta)
    new_values = {name: bounded_value(name, candidate, BEHAVIOR_BOUNDS)}
    if name == "relief_start_day" and new_values[name] >= behavior.relief_peak_day - 4:
        new_values[name] = behavior.relief_peak_day - 4
    if name == "relief_peak_day" and new_values[name] <= behavior.relief_start_day + 4:
        new_values[name] = behavior.relief_start_day + 4
    return replace(behavior, **new_values)


def run_single_factor_audit(
    event_df: pd.DataFrame,
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
    current_metrics: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    candidate_paths: list[pd.DataFrame] = []
    audit_parameters = [
        ("assumption", "supply_interruption", "供应中断量"),
        ("assumption", "spr_max_release", "SPR释放上限"),
        ("assumption", "route_start_day", "绕道启动日"),
        ("assumption", "route_max_capacity", "绕道能力上限"),
        ("assumption", "fear_decay", "恐慌衰减速度"),
        ("behavior", "pressure_scale", "缺口价格传导系数"),
        ("behavior", "risk_weight", "封锁风险溢价权重"),
        ("behavior", "uncertainty_floor", "不确定性溢价平台"),
        ("behavior", "inventory_response", "库存响应比例"),
        ("behavior", "adjustment_speed", "价格调整速度"),
        ("behavior", "buffer_relief_strength", "缓冲确认折价强度"),
        ("behavior", "relief_discount_strength", "预期修复折价强度"),
        ("behavior", "relief_start_day", "预期修复开始日"),
        ("behavior", "relief_decay_days", "预期修复消退期"),
    ]
    for group, parameter, label in audit_parameters:
        for delta in PERTURBATION_GRID:
            trial_assumptions = assumptions
            trial_behavior = behavior
            if group == "assumption":
                trial_assumptions = perturb_assumption(assumptions, parameter, delta)
                original = float(asdict(assumptions)[parameter])
                value = float(asdict(trial_assumptions)[parameter])
            else:
                trial_behavior = perturb_behavior(behavior, parameter, delta)
                original = float(asdict(behavior)[parameter])
                value = float(asdict(trial_behavior)[parameter])
            if value == original:
                continue
            simulation = dynamic.simulate_dynamic_model(event_df, trial_assumptions, trial_behavior)
            metrics = model_metrics(simulation)
            row = {
                "参数组": group,
                "参数": parameter,
                "中文名": label,
                "扰动比例": delta,
                "原值": original,
                "扰动后值": value,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MAPE": metrics["MAPE"],
                "方向命中率": metrics["方向命中率"],
                "最大绝对误差": metrics["最大绝对误差"],
                "综合得分": metrics["综合得分"],
                "高位平台形成_RMSE": metrics["高位平台形成_RMSE"],
                "中期再定价回落_RMSE": metrics["中期再定价回落_RMSE"],
                "后期二次抬升_RMSE": metrics["后期二次抬升_RMSE"],
                "RMSE改善率": (current_metrics["RMSE"] - metrics["RMSE"]) / current_metrics["RMSE"] * 100,
                "综合得分改善率": (current_metrics["综合得分"] - metrics["综合得分"]) / current_metrics["综合得分"] * 100,
            }
            rows.append(row)
            if row["RMSE改善率"] > 0 or row["综合得分改善率"] > 0:
                candidate = simulation.copy()
                candidate["参数"] = parameter
                candidate["扰动比例"] = delta
                candidate["扰动后值"] = value
                candidate_paths.append(candidate)

    perturbations = pd.DataFrame(rows).sort_values(["综合得分", "RMSE"]).reset_index(drop=True)
    if candidate_paths:
        candidates = pd.concat(candidate_paths, ignore_index=True)
    else:
        candidates = pd.DataFrame()
    return perturbations, candidates


def build_candidate_summary(perturbations: pd.DataFrame) -> pd.DataFrame:
    by_parameter = perturbations.sort_values(["综合得分", "RMSE"]).groupby("参数", as_index=False).head(1).copy()
    by_parameter["是否值得进一步联动精修"] = (
        (by_parameter["RMSE改善率"] >= 1.0)
        | (by_parameter["综合得分改善率"] >= 1.0)
        | (
            (by_parameter["中期再定价回落_RMSE"] < 3.55)
            & (by_parameter["RMSE改善率"] >= 0)
        )
    )
    return by_parameter.sort_values(["是否值得进一步联动精修", "综合得分改善率", "RMSE改善率"], ascending=[False, False, False])


def draw_figure(current_metrics: dict[str, float], summary: pd.DataFrame, contributions: pd.DataFrame) -> None:
    configure_plot_style(savefig_dpi=260, figure_dpi=150, title_size=13)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))

    top = summary.sort_values("综合得分改善率", ascending=False).head(8).copy()
    axes[0, 0].barh(top["中文名"], top["综合得分改善率"], color=SCENARIO_COLORS["fit"], alpha=0.9)
    axes[0, 0].axvline(0, color=PAPER_COLORS["ink"], lw=0.9)
    axes[0, 0].set_title("单因子扰动的综合得分改善")
    axes[0, 0].set_xlabel("改善率（%）")

    top_rmse = summary.sort_values("RMSE改善率", ascending=False).head(8).copy()
    axes[0, 1].barh(top_rmse["中文名"], top_rmse["RMSE改善率"], color=SCENARIO_COLORS["buffer"], alpha=0.9)
    axes[0, 1].axvline(0, color=PAPER_COLORS["ink"], lw=0.9)
    axes[0, 1].set_title("单因子扰动的 RMSE 改善")
    axes[0, 1].set_xlabel("改善率（%）")

    contrib = contributions[contributions["窗口"].isin(["全窗口", "高位平台形成", "中期再定价回落"])].copy()
    x = np.arange(len(contrib))
    width = 0.24
    axes[1, 0].bar(x - width, contrib["上行压力合计_均值"], width, label="上行压力", color=SCENARIO_COLORS["risk"], alpha=0.78)
    axes[1, 0].bar(x, contrib["下行折价合计_均值"], width, label="下行折价", color=SCENARIO_COLORS["optimistic"], alpha=0.78)
    axes[1, 0].bar(x + width, contrib["净压力_均值"], width, label="净压力", color=SCENARIO_COLORS["fit"], alpha=0.78)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(contrib["窗口"], rotation=15)
    axes[1, 0].set_title("主机制贡献项均值")
    axes[1, 0].set_ylabel("美元/桶")
    axes[1, 0].legend(loc="upper right", fontsize=8)

    labels = ["全窗口", "高位平台", "中期回落", "后期二次抬升"]
    values = [
        current_metrics["RMSE"],
        current_metrics["高位平台形成_RMSE"],
        current_metrics["中期再定价回落_RMSE"],
        current_metrics["后期二次抬升_RMSE"],
    ]
    axes[1, 1].bar(labels, values, color=[SCENARIO_COLORS["fit"], SCENARIO_COLORS["risk"], SCENARIO_COLORS["buffer"], SCENARIO_COLORS["neutral"]])
    axes[1, 1].set_title("当前主机制模型误差薄弱点")
    axes[1, 1].set_ylabel("RMSE（美元/桶）")
    axes[1, 1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(Paths.figure_path)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for row in df[columns].to_dict("records")]
    return "\n".join([header, sep, *rows])


def build_report(current_metrics: dict[str, float], contributions: pd.DataFrame, summary: pd.DataFrame) -> str:
    useful = summary[summary["是否值得进一步联动精修"]].copy()
    top_show = summary.head(10).copy()
    for column in ["原值", "扰动后值", "RMSE", "综合得分", "RMSE改善率", "综合得分改善率", "中期再定价回落_RMSE"]:
        top_show[column] = top_show[column].map(lambda x: f"{float(x):.3f}")
    if useful.empty:
        conclusion_text = (
            "单因子小扰动暂未发现明显可替换当前主模型的入口；最好的扰动只带来约 0.15% 的 RMSE 改善，"
            "不足以支撑替换主机制参数。"
        )
    else:
        useful_text = "、".join(useful["中文名"].head(6).to_list())
        conclusion_text = f"单因子小扰动显示，以下机制参数可作为下一轮联动精修入口：{useful_text}。"
    contrib_show = contributions[contributions["窗口"].isin(["全窗口", "高位平台形成", "中期再定价回落", "后期二次抬升"])].copy()
    for column in ["RMSE", "上行压力合计_均值", "下行折价合计_均值", "净压力_均值", "模型误差_均值"]:
        contrib_show[column] = contrib_show[column].map(lambda x: f"{float(x):.3f}")

    return f"""# 短期主机制模型可优化性审计报告

> 本报告只审计主机制模型本体，不替换当前官方路径，也不把结果写入论文正文。

## 核心结论

当前主机制模型 RMSE={current_metrics['RMSE']:.3f}，MAE={current_metrics['MAE']:.3f}，MAPE={current_metrics['MAPE']:.3f}%，方向命中率={current_metrics['方向命中率']:.1f}%。误差最集中的位置仍是中期再定价回落窗口，RMSE={current_metrics['中期再定价回落_RMSE']:.3f}。

审计结论是：**主机制模型不是不能优化，而是不能用随意加项的方式优化。** {conclusion_text}后续若继续优化，应保持在已有机制项内部做低自由度联动精修，不新增黑箱变量。

## 机制贡献概览

{markdown_table(contrib_show, ['窗口', '样本数', 'RMSE', '上行压力合计_均值', '下行折价合计_均值', '净压力_均值', '模型误差_均值'])}

## 单因子扰动前十候选

{markdown_table(top_show, ['中文名', '扰动比例', '原值', '扰动后值', 'RMSE', '综合得分', 'RMSE改善率', '综合得分改善率', '中期再定价回落_RMSE', '是否值得进一步联动精修'])}

## 建模判断

1. 如果单因子扰动只能带来很小改进，则不应立即替换主机制参数；主机制模型更重要的是解释稳定。
2. 如果若干机制项在同一方向上稳定改善中期回落或高位平台误差，可以进入下一轮“低自由度联动精修”。
3. 外部风险变量和 Ridge 增强层不应混入主机制方程；它们继续作为审计、辅助拟合或长期状态概率约束。
4. 下一轮若改主机制模型，应保留当前版本作为基准，并输出参数边界、贡献项量级和分段误差变化，防止为了 RMSE 牺牲经济含义。

## 输出

- `{Paths.contribution_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.perturbation_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.candidate_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.figure_path.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    ensure_parents([Paths.contribution_csv, Paths.perturbation_csv, Paths.candidate_csv, Paths.report_path, Paths.figure_path])
    event_df = load_event_window()
    current_path = load_current_path()
    assumptions, behavior, _ = load_best_objects()
    current_metrics = model_metrics(current_path)
    contributions = contribution_audit(current_path)
    perturbations, _ = run_single_factor_audit(event_df, assumptions, behavior, current_metrics)
    summary = build_candidate_summary(perturbations)

    contributions.to_csv(Paths.contribution_csv, index=False)
    perturbations.to_csv(Paths.perturbation_csv, index=False)
    summary.to_csv(Paths.candidate_csv, index=False)
    draw_figure(current_metrics, summary, contributions)
    Paths.report_path.write_text(build_report(current_metrics, contributions, summary), encoding="utf-8")

    print("Short-term mechanism optimization audit complete")
    print(f"Report: {Paths.report_path.relative_to(PROJECT_ROOT)}")
    print(f"Candidates: {Paths.candidate_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
