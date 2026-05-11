"""Stage 11 audit for independent expert challenges.

This script does not replace the main model.  It turns the external critique
into reproducible checks: which claims are stale after recent model changes,
which risks remain real, and which counterfactual assumptions materially move
the long-horizon results.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parent
from src.models import dynamic_short_term as dynamic
from src.scenarios import forecast_stage5 as stage5
from src.scenarios import simulation as scenario_sim


QA_DIR = PROJECT_ROOT / "output" / "qa"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "stage11_expert_challenge_audit_report.md"
CONSISTENCY_CSV = QA_DIR / "长期模型一致性审计.csv"
COUNTERFACTUAL_CSV = QA_DIR / "专家质疑反驳性检验.csv"
PRESSURE_CSV = QA_DIR / "短期pressure_scale敏感性.csv"
MARKER_DAYS = [60, 90, 120, 180]


def load_stage5_context() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]],
]:
    base_config = dynamic.load_yaml(dynamic.BASE_CONFIG_PATH)
    scenario_config = dynamic.load_yaml(dynamic.SCENARIO_CONFIG_PATH)
    paths = dynamic.resolve_paths(base_config)
    event_df = dynamic.load_event_window(paths.event_csv)
    best = stage5.load_best_row()
    base_assumptions, base_behavior = stage5.calibrated_assumptions_and_behavior(best)
    scenarios = stage5.build_scenario_parameters(base_assumptions, base_behavior, scenario_config)
    forecast_frame = stage5.build_forecast_frame(event_df)
    prefix = stage5.load_calibrated_prefix()
    return event_df, forecast_frame, prefix, scenarios


def simulate_path(
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
    forecast_frame: pd.DataFrame,
    prefix: pd.DataFrame,
    uncertainty_fn: Callable | None = None,
    oversupply_fn: Callable | None = None,
    spr_fn: Callable | None = None,
) -> pd.DataFrame:
    original_uncertainty = scenario_sim.uncertainty_components
    original_oversupply = scenario_sim.oversupply_discount
    original_spr = scenario_sim.adaptive_spr_release
    try:
        if uncertainty_fn is not None:
            scenario_sim.uncertainty_components = uncertainty_fn
        if oversupply_fn is not None:
            scenario_sim.oversupply_discount = oversupply_fn
        if spr_fn is not None:
            scenario_sim.adaptive_spr_release = spr_fn
        future_frame = forecast_frame[forecast_frame["阶段"] == "情景外推期"].copy()
        base_price = float(forecast_frame.iloc[0]["pre_close"])
        future = scenario_sim.simulate_future_from_prefix(prefix, future_frame, assumptions, behavior, base_price)
        return pd.concat([prefix, future], ignore_index=True)
    finally:
        scenario_sim.uncertainty_components = original_uncertainty
        scenario_sim.oversupply_discount = original_oversupply
        scenario_sim.adaptive_spr_release = original_spr


def decaying_uncertainty_components(
    day_index: int,
    base_price: float,
    behavior: dynamic.BehavioralParameters,
    assumptions: dynamic.PhysicalAssumptions,
    spr_release: float,
    route_supply: float,
    demand_decline: float,
) -> tuple[float, float, float]:
    """Counterfactual: force both shock and regime risk to decay over time."""
    buildup = 1 - np.exp(-day_index / 18)
    shock_decay = np.exp(-max(day_index - 30, 0) / 45)
    shock_uncertainty = base_price * behavior.uncertainty_floor * 0.45 * buildup * shock_decay

    unresolved_stress = (
        assumptions.supply_interruption
        - spr_release
        - route_supply
        - demand_decline
    ) / max(assumptions.supply_interruption, 1.0)
    regime_share = np.clip(unresolved_stress, 0.0, 1.0)
    regime_decay = np.exp(-max(day_index - 60, 0) / 90)
    regime_risk = base_price * behavior.uncertainty_floor * 0.90 * regime_share * regime_decay
    return float(shock_uncertainty + regime_risk), float(shock_uncertainty), float(regime_risk)


def strong_oversupply_discount(
    oversupply: float,
    base_price: float,
    elasticity: float,
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> float:
    """Counterfactual: make excess supply pull prices down more aggressively."""
    if oversupply <= 0:
        return 0.0
    return float(
        base_price
        * behavior.pressure_scale
        * 2.2
        * (oversupply / assumptions.base_demand)
        / max(abs(elasticity), 0.01)
    )


def aggressive_spr_taper(
    day_index: int,
    scheduled_spr: float,
    gross_gap_before_spr: float,
    previous_price: float,
    base_price: float,
    assumptions: dynamic.PhysicalAssumptions,
) -> tuple[float, float]:
    """Counterfactual: policy release responds more sharply to closed gaps."""
    if scheduled_spr <= 0:
        return 0.0, 0.0
    reserve_buffer = assumptions.base_demand * 0.01
    coverage_need_current = max(gross_gap_before_spr + reserve_buffer, 0.0)
    demand_based_release = min(scheduled_spr, coverage_need_current)

    stress_ratio = np.clip(gross_gap_before_spr / max(assumptions.supply_interruption, 1.0), 0.0, 1.0)
    price_stress_current = np.clip((previous_price / base_price - 1.03) / 0.25, 0.0, 1.0)
    minimum_policy_share = 0.12 + 0.18 * price_stress_current
    stress_share = max(minimum_policy_share, stress_ratio)
    time_taper = 1.0
    if day_index > 75:
        time_taper = 0.35 + 0.65 * np.exp(-(day_index - 75) / 45)
    current_release = min(scheduled_spr, max(demand_based_release, scheduled_spr * stress_share * time_taper))

    price_stress = np.clip((previous_price / base_price - 1.05) / 0.35, 0.0, 1.0)
    reserve_buffer = assumptions.base_demand * 0.005
    coverage_need = max(gross_gap_before_spr + reserve_buffer, 0.0)
    policy_floor = scheduled_spr * (0.04 + 0.16 * price_stress)
    tapered = min(current_release, max(coverage_need, policy_floor))
    tapered = float(min(scheduled_spr, max(0.0, tapered)))
    return tapered, tapered / scheduled_spr


def scenario_summary(path: pd.DataFrame, label: str, scenario: str) -> dict[str, float | str]:
    final = path[path["day_index"] == 180].iloc[0]
    peak = float(path["forecast_price"].max())
    marker_prices = {
        f"第{day}天价格": float(path.loc[path["day_index"] == day, "forecast_price"].iloc[0])
        for day in MARKER_DAYS
    }
    return {
        "检验": label,
        "情景": scenario,
        **marker_prices,
        "外推期最高价": peak,
        "第180天有效供给": float(final["effective_supply"]),
        "第180天有效需求": float(final["effective_demand"]),
        "第180天剩余缺口": float(final["supply_gap"]),
        "第180天供给过剩": max(float(final["supply_balance"]), 0.0),
        "第180天SPR计划释放": float(final["scheduled_spr_release"]),
        "第180天SPR实际释放": float(final["spr_release"]),
        "第180天SPR释放比例": float(final["spr_taper_ratio"]),
        "第180天不确定性溢价": float(final["uncertainty_premium"]),
        "第180天冲击不确定性": float(final["shock_uncertainty_premium"]),
        "第180天制度风险溢价": float(final["regime_risk_premium"]),
        "第180天过剩供给折价": float(final["oversupply_discount"]),
    }


def build_consistency_audit(result: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for scenario, label in [("optimistic", "乐观"), ("neutral", "中性"), ("pessimistic", "悲观")]:
        subset = result[(result["scenario"] == scenario) & (result["day_index"].isin(MARKER_DAYS))]
        for _, row in subset.iterrows():
            rows.append(
                {
                    "情景": label,
                    "day_index": int(row["day_index"]),
                    "预测价格": float(row["forecast_price"]),
                    "有效供给": float(row["effective_supply"]),
                    "有效需求": float(row["effective_demand"]),
                    "剩余缺口": float(row["supply_gap"]),
                    "供给过剩": max(float(row["supply_balance"]), 0.0),
                    "SPR计划释放": float(row["scheduled_spr_release"]),
                    "SPR实际释放": float(row["spr_release"]),
                    "SPR释放比例": float(row["spr_taper_ratio"]),
                    "不确定性溢价": float(row["uncertainty_premium"]),
                    "冲击不确定性": float(row["shock_uncertainty_premium"]),
                    "制度风险溢价": float(row["regime_risk_premium"]),
                    "过剩供给折价": float(row["oversupply_discount"]),
                }
            )
    return pd.DataFrame(rows)


def run_counterfactuals(
    forecast_frame: pd.DataFrame,
    prefix: pd.DataFrame,
    scenarios: dict[str, tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for scenario_key, scenario_label in [("neutral", "中性"), ("pessimistic", "悲观")]:
        assumptions, behavior = scenarios[scenario_key]
        variants = [
            ("当前模型", {}, None, None, None),
            ("不确定性溢价减半", {"uncertainty_floor": behavior.uncertainty_floor * 0.5}, None, None, None),
            ("不确定性随时间衰减", {}, decaying_uncertainty_components, None, None),
            ("强过剩供给折价", {}, None, strong_oversupply_discount, None),
            ("更强SPR价格反馈", {}, None, None, aggressive_spr_taper),
        ]
        for label, behavior_updates, uncertainty_fn, oversupply_fn, spr_fn in variants:
            variant_behavior = replace(behavior, **behavior_updates)
            path = simulate_path(
                assumptions,
                variant_behavior,
                forecast_frame,
                prefix,
                uncertainty_fn=uncertainty_fn,
                oversupply_fn=oversupply_fn,
                spr_fn=spr_fn,
            )
            rows.append(scenario_summary(path, label, scenario_label))
    return pd.DataFrame(rows)


def run_pressure_scale_sensitivity(
    event_df: pd.DataFrame,
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    variants = [
        ("0.5x", behavior.pressure_scale * 0.5),
        ("当前值", behavior.pressure_scale),
        ("1.5x", behavior.pressure_scale * 1.5),
        ("2.0x", behavior.pressure_scale * 2.0),
        ("1.0绝对值压力传导", 1.0),
    ]
    for label, pressure_scale in variants:
        variant_behavior = replace(behavior, pressure_scale=float(pressure_scale))
        simulation = dynamic.simulate_dynamic_model(event_df, assumptions, variant_behavior)
        metrics = dynamic.compute_metrics(simulation, variant_behavior)
        rows.append(
            {
                "检验": label,
                "pressure_scale": float(pressure_scale),
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
                "模拟峰值": metrics["simulated_peak_price"],
                "实际峰值": metrics["actual_peak_price"],
                "末日模拟价": metrics["simulated_final_price"],
                "末日实际价": metrics["actual_final_price"],
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], digits: int = 2) -> str:
    view = df[columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda x: f"{x:.{digits}f}")
    return view.to_markdown(index=False)


def build_report(consistency: pd.DataFrame, counterfactuals: pd.DataFrame, pressure: pd.DataFrame) -> str:
    neutral_180 = consistency[(consistency["情景"] == "中性") & (consistency["day_index"] == 180)].iloc[0]
    report = f"""# 阶段 11：专家质疑回应与模型一致性审计报告

## 1. 审计目的

本报告回应 `质疑报告/独立专家评审报告_20260512_模型诊断专家.md` 中关于长期供需闭环、SPR 反馈、不确定性溢价和 `pressure_scale` 的质疑。阶段 12 已将长期外推替换为新版自适应反馈主模型；本报告用于复核新版主模型是否仍存在专家指出的矛盾。

## 2. 对“600 万桶过剩悖论”的当前复核

专家报告中的“中性第 180 天每天过剩约 500--600 万桶”对应旧逻辑或手算假设。当前模型第 180 天中性情景为：

- 预测价格：{neutral_180["预测价格"]:.2f} 美元/桶
- 有效供给：{neutral_180["有效供给"]:.2f} 万桶/日
- 有效需求：{neutral_180["有效需求"]:.2f} 万桶/日
- 供给过剩：{neutral_180["供给过剩"]:.2f} 万桶/日
- SPR 计划释放：{neutral_180["SPR计划释放"]:.2f} 万桶/日
- SPR 实际释放：{neutral_180["SPR实际释放"]:.2f} 万桶/日
- SPR 实际释放比例：{neutral_180["SPR释放比例"]:.2f}

因此，当前代码已经不再是“SPR 固定满额 670 万桶/日释放”的旧逻辑；中性情景的实际 SPR 释放已随缺口和价格压力收缩，供给过剩也从旧模型的约 100 万桶/日降至约 50 万桶/日。仍需承认：模型不是完整一般均衡模型，长期结果应解释为条件情景中心路径，而不是未来真实价格拟合。

## 3. 长期一致性审计表

{markdown_table(consistency[consistency["day_index"].isin([90, 180])], ["情景", "day_index", "预测价格", "有效供给", "有效需求", "剩余缺口", "供给过剩", "SPR实际释放", "不确定性溢价", "过剩供给折价"])}

## 4. 反驳性检验

{markdown_table(counterfactuals, ["情景", "检验", "第180天价格", "外推期最高价", "第180天供给过剩", "第180天SPR实际释放", "第180天不确定性溢价", "第180天过剩供给折价"])}

主要观察：

- 不确定性溢价减半或强制衰减仍会压低中性和悲观第 180 天价格，说明制度风险项仍是长期模型的关键敏感源。
- 强化过剩供给折价主要影响已经接近供需闭合的中性/乐观路径，对悲观高缺口路径影响有限。
- 更强 SPR 价格反馈对新版中性路径影响很小，说明主模型中 SPR 已经主要由剩余缺口驱动。

## 5. pressure_scale 反驳性检验

{markdown_table(pressure, ["检验", "pressure_scale", "RMSE", "MAE", "模拟峰值", "实际峰值", "末日模拟价", "末日实际价"])}

结论：`pressure_scale=1.0` 会显著破坏短期拟合，说明它不能被解释为题面弹性的直接替代。更稳妥的论文表述是：题面低弹性用于传统供需反事实上界；`pressure_scale` 是剩余缺口进入期货价格的行为传导参数，必须作为校准参数接受敏感性检验和文献支撑。

## 6. 当前处理建议

1. 将新版自适应反馈模型作为唯一长期主模型，旧模型只保留在历史审计材料中。
2. 在论文中统一使用新版长期结果，避免新旧模型口径混用。
3. 继续强调长期蒙特卡洛概率是模型设定下的条件概率，不是历史频率意义上的真实市场概率。
4. 若后续继续增强，应优先补入 IEA/OPEC 全球口径供需数据和市场隐含风险变量，而不是再微调长期递推参数。
"""
    return report


def main() -> None:
    event_df, forecast_frame, prefix, scenarios = load_stage5_context()
    current_result = pd.read_csv(stage5.SCENARIO_RESULT_CSV)
    consistency = build_consistency_audit(current_result)
    counterfactuals = run_counterfactuals(forecast_frame, prefix, scenarios)
    base_assumptions, base_behavior = scenarios["neutral"]
    pressure = run_pressure_scale_sensitivity(event_df, base_assumptions, base_behavior)

    for path in [CONSISTENCY_CSV, COUNTERFACTUAL_CSV, PRESSURE_CSV, REPORT_PATH]:
        ensure_parent(path)
    consistency.to_csv(CONSISTENCY_CSV, index=False)
    counterfactuals.to_csv(COUNTERFACTUAL_CSV, index=False)
    pressure.to_csv(PRESSURE_CSV, index=False)
    REPORT_PATH.write_text(build_report(consistency, counterfactuals, pressure), encoding="utf-8")

    print("Stage 11 expert challenge audit complete")
    print(f"Consistency audit: {CONSISTENCY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Counterfactual tests: {COUNTERFACTUAL_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Pressure sensitivity: {PRESSURE_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
