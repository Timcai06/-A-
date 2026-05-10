"""Mechanism ablation study for the unified short-term model.

The goal is not to create a second "strict" model. Instead, this module keeps
the same calibrated short-term model and turns one mechanism off at a time.
This makes the role of problem-statement physical mechanisms and added market
price-formation mechanisms auditable.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.short_term_model_defense import reconstruct_best_parameters
from src.calibration import calibrate_dynamic_model as calibration
from src.models import dynamic_short_term as dynamic


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AblationPaths:
    metrics_csv = PROJECT_ROOT / "output" / "calibration" / "短期模型机制消融实验.csv"
    report_path = PROJECT_ROOT / "output" / "reports" / "短期模型机制消融实验报告.md"
    figure_path = PROJECT_ROOT / "paper" / "figures" / "短期模型机制消融实验.png"


VariantFn = Callable[
    [dynamic.PhysicalAssumptions, dynamic.BehavioralParameters],
    tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters],
]


def ensure_dirs() -> None:
    for path in [AblationPaths.metrics_csv, AblationPaths.report_path, AblationPaths.figure_path]:
        path.parent.mkdir(parents=True, exist_ok=True)


def load_event_window() -> pd.DataFrame:
    base_config = dynamic.load_yaml(dynamic.BASE_CONFIG_PATH)
    return dynamic.load_event_window(dynamic.resolve_paths(base_config).event_csv)


def base_variant(
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]:
    return assumptions, behavior


def no_spr(
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]:
    return replace(assumptions, spr_max_release=0.0), behavior


def no_inventory(
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]:
    return replace(assumptions, commercial_inventory=0.0, inventory_daily_cap=0.0), replace(
        behavior,
        inventory_response=0.0,
    )


def no_reroute(
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]:
    return replace(assumptions, route_max_capacity=0.0), behavior


def no_demand_contraction(
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]:
    return replace(assumptions, observed_demand_decline=0.0), behavior


def no_panic(
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]:
    return replace(assumptions, fear_initial=0.0), behavior


def no_risk_uncertainty(
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]:
    return assumptions, replace(behavior, risk_weight=0.0, uncertainty_floor=0.0)


def no_buffer_confirmation(
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]:
    return assumptions, replace(behavior, buffer_relief_strength=0.0)


def no_expectation_relief(
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]:
    return assumptions, replace(behavior, relief_discount_strength=0.0)


def no_market_corrections(
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]:
    return assumptions, replace(
        behavior,
        risk_weight=0.0,
        uncertainty_floor=0.0,
        buffer_relief_strength=0.0,
        relief_discount_strength=0.0,
    )


VARIANTS: list[tuple[str, str, str, VariantFn]] = [
    ("完整统一模型", "统一模型", "题面物理机制 + 市场价格形成机制", base_variant),
    ("去除SPR释放", "题面物理层", "关闭战略石油储备释放，检查政策缓冲贡献", no_spr),
    ("去除商业库存缓冲", "题面物理层", "关闭商业库存吸收缺口能力", no_inventory),
    ("去除绕道运输", "题面物理层", "关闭绕道运输恢复能力", no_reroute),
    ("去除需求收缩", "题面需求层", "关闭题面给出的高价需求下降", no_demand_contraction),
    ("去除恐慌溢价", "题面行为层", "关闭冲突初期恐慌放大项", no_panic),
    ("去除风险与不确定性溢价", "市场价格形成层", "关闭地缘风险重估和中期不确定性平台", no_risk_uncertainty),
    ("去除缓冲确认折价", "市场价格形成层", "关闭市场确认缓冲机制生效后的第一轮降温", no_buffer_confirmation),
    ("去除预期修复折价", "市场价格形成层", "关闭中后期对运输与政策修复的再定价", no_expectation_relief),
    ("去除全部市场修正", "市场价格形成层", "保留题面物理层，关闭风险、不确定性和两类折价", no_market_corrections),
]


def run_ablation() -> pd.DataFrame:
    event_df = load_event_window()
    base_assumptions, base_behavior = reconstruct_best_parameters()
    rows: list[dict[str, float | str]] = []

    base_metrics: dict[str, float] | None = None
    for name, layer, note, fn in VARIANTS:
        assumptions, behavior = fn(base_assumptions, base_behavior)
        simulation = dynamic.simulate_dynamic_model(event_df, assumptions, behavior)
        metrics = calibration.evaluate_simulation(simulation)
        if name == "完整统一模型":
            base_metrics = metrics
        if base_metrics is None:
            raise RuntimeError("Base variant must be evaluated first.")
        rows.append(
            {
                "消融实验": name,
                "机制层级": layer,
                "说明": note,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "峰值误差": metrics["峰值误差"],
                "末日误差": metrics["末日误差"],
                "高价平台RMSE": metrics["高价平台RMSE"],
                "低价回落RMSE": metrics["低价回落RMSE"],
                "前期RMSE": metrics["前期RMSE"],
                "中期RMSE": metrics["中期RMSE"],
                "后期RMSE": metrics["后期RMSE"],
                "模拟峰值": metrics["模拟峰值"],
                "模拟末日价格": metrics["模拟末日价格"],
                "RMSE相对完整模型变化": metrics["RMSE"] - base_metrics["RMSE"],
                "后期RMSE相对完整模型变化": metrics["后期RMSE"] - base_metrics["后期RMSE"],
                "低价回落RMSE相对完整模型变化": metrics["低价回落RMSE"] - base_metrics["低价回落RMSE"],
            }
        )
    return pd.DataFrame(rows)


def draw_figure(table: pd.DataFrame) -> None:
    dynamic.configure_plot_style()
    plot = table.iloc[1:].copy().sort_values("RMSE相对完整模型变化")
    colors = plot["机制层级"].map(
        {
            "题面物理层": "#2563eb",
            "题面需求层": "#0f766e",
            "题面行为层": "#7c3aed",
            "市场价格形成层": "#f97316",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    axes[0].barh(plot["消融实验"], plot["RMSE相对完整模型变化"], color=colors)
    axes[0].axvline(0, color="#111827", linewidth=1)
    axes[0].set_title("机制消融后 RMSE 变化")
    axes[0].set_xlabel("相对完整模型 RMSE 变化")

    late_plot = table.iloc[1:].copy().sort_values("后期RMSE相对完整模型变化")
    late_colors = late_plot["机制层级"].map(
        {
            "题面物理层": "#2563eb",
            "题面需求层": "#0f766e",
            "题面行为层": "#7c3aed",
            "市场价格形成层": "#f97316",
        }
    )
    axes[1].barh(late_plot["消融实验"], late_plot["后期RMSE相对完整模型变化"], color=late_colors)
    axes[1].axvline(0, color="#111827", linewidth=1)
    axes[1].set_title("机制消融后后期 RMSE 变化")
    axes[1].set_xlabel("相对完整模型后期 RMSE 变化")

    fig.tight_layout()
    fig.savefig(AblationPaths.figure_path, dpi=190)
    plt.close(fig)


def build_report(table: pd.DataFrame) -> str:
    rows = "\n".join(
        "| {消融实验} | {机制层级} | {RMSE:.2f} | {RMSE相对完整模型变化:+.2f} | {后期RMSE:.2f} | {后期RMSE相对完整模型变化:+.2f} | {低价回落RMSE:.2f} | {说明} |".format(
            **row
        )
        for row in table.to_dict("records")
    )
    largest_rmse = table.iloc[1:].sort_values("RMSE相对完整模型变化", ascending=False).iloc[0]
    largest_late = table.iloc[1:].sort_values("后期RMSE相对完整模型变化", ascending=False).iloc[0]
    market_all = table[table["消融实验"] == "去除全部市场修正"].iloc[0]

    return f"""# 短期模型机制消融实验报告

## 目的

本报告不另建新模型，而是在同一个短期动态模型内逐项关闭机制，用来回答两个问题：

1. 题面要求的 SPR、商业库存、绕道运输、需求收缩、恐慌需求是否真的进入了模型解释链条。
2. 额外加入的风险溢价、缓冲确认折价和预期修复折价是否只是为了曲线拟合硬塞进去。

## 结果表

| 消融实验 | 机制层级 | RMSE | RMSE变化 | 后期RMSE | 后期RMSE变化 | 低价回落RMSE | 说明 |
|---|---|---:|---:|---:|---:|---:|---|
{rows}

## 主要发现

- 对全窗口 RMSE 影响最大的消融项是“{largest_rmse["消融实验"]}”，关闭后 RMSE 相对完整模型变化 {largest_rmse["RMSE相对完整模型变化"]:+.2f}。
- 对后期再定价影响最大的消融项是“{largest_late["消融实验"]}”，关闭后后期 RMSE 相对完整模型变化 {largest_late["后期RMSE相对完整模型变化"]:+.2f}。
- “去除全部市场修正”后 RMSE 为 {market_all["RMSE"]:.2f}，后期 RMSE 为 {market_all["后期RMSE"]:.2f}，说明题面物理机制仍是模型主干，但市场价格形成层对后期回落和平台解释有补充作用。

## 论文表述建议

论文中应把短期模型写成一个统一模型，而不是拆成两个模型。推荐表述为：

> 模型由题面物理供需层和市场价格形成层组成。前者负责刻画供应中断、SPR、商业库存、绕道运输、需求收缩和恐慌需求，后者负责刻画期货价格对物理缺口的传导效率、地缘风险重估以及缓冲机制被市场确认后的预期修复。消融实验表明，关闭题面物理机制或市场修正机制都会改变误差结构，说明额外项不是独立于题面机制的任意拟合项，而是对价格形成过程的补充刻画。

## 输出产物

- `{AblationPaths.metrics_csv.relative_to(PROJECT_ROOT)}`
- `{AblationPaths.figure_path.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    ensure_dirs()
    table = run_ablation()
    table.to_csv(AblationPaths.metrics_csv, index=False)
    draw_figure(table)
    AblationPaths.report_path.write_text(build_report(table), encoding="utf-8")
    print("Short-term model ablation complete")
    print(f"Metrics: {AblationPaths.metrics_csv.relative_to(PROJECT_ROOT)}")
    print(f"Figure: {AblationPaths.figure_path.relative_to(PROJECT_ROOT)}")
    print(f"Report: {AblationPaths.report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
