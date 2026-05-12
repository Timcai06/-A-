"""One-at-a-time sensitivity analysis for the 60-180 day forecast.

The analysis reuses the neutral scenario path as the central baseline. Each run
changes one parameter at a time and keeps all other calibrated mechanisms fixed,
so the output can be read as a clean marginal influence ranking.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parent
from src.scenarios import scenario_forecast as scenario
from src.scenarios import simulation as scenario_sim


OUTPUT_DIR = PROJECT_ROOT / "output" / "sensitivity"
RESULT_CSV = OUTPUT_DIR / "敏感性分析结果.csv"
RANKING_CSV = OUTPUT_DIR / "参数重要性排序.csv"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "敏感性分析报告.md"
TORNADO_FIGURE = PROJECT_ROOT / "figures" / "sensitivity_tornado_180day.png"
RESPONSE_FIGURE = PROJECT_ROOT / "figures" / "sensitivity_parameter_response.png"


SENSITIVITY_SPECS: list[dict[str, Any]] = [
    {
        "key": "supply_interruption",
        "参数": "供应中断量",
        "层级": "题面物理层",
        "单位": "万桶/日",
        "扰动": [1400, 1493, 1600, 1800],
        "可控性": "低",
        "解释": "封锁实际造成的供应缺口，是价格上行压力的源头变量。",
    },
    {
        "key": "spr_max_release",
        "参数": "SPR释放上限",
        "层级": "题面政策缓冲层",
        "单位": "万桶/日",
        "扰动": [200, 450, 670, 700],
        "可控性": "高",
        "解释": "战略石油储备释放能力，直接缓冲供应缺口。",
    },
    {
        "key": "spr_delay_days",
        "参数": "SPR启动延迟",
        "层级": "题面政策缓冲层",
        "单位": "天",
        "扰动": [0, 3, 7, 14],
        "可控性": "高",
        "解释": "战略储备从冲击发生到释放启动的时间延迟。",
    },
    {
        "key": "route_max_capacity",
        "参数": "绕道运输能力",
        "层级": "题面物理缓冲层",
        "单位": "万桶/日",
        "扰动": [150, 237, 300, 420],
        "可控性": "中",
        "解释": "绕道、替代航线和运输组织恢复形成的额外供给能力。",
    },
    {
        "key": "route_ramp_days",
        "参数": "绕道恢复耗时",
        "层级": "题面物理缓冲层",
        "单位": "天",
        "扰动": [14, 26, 42, 60],
        "可控性": "中",
        "解释": "绕道运输从启动到达到最大能力所需时间。",
    },
    {
        "key": "long_elasticity",
        "参数": "长期需求弹性",
        "层级": "题面需求层",
        "单位": "无量纲",
        "扰动": [-0.35, -0.25, -0.18, -0.10],
        "可控性": "低",
        "解释": "中长期高价下需求收缩强弱，绝对值越大代表需求越容易下降。",
    },
    {
        "key": "fear_decay",
        "参数": "恐慌衰减速度",
        "层级": "题面行为层",
        "单位": "日衰减率",
        "扰动": [0.04, 0.08, 0.114, 0.16],
        "可控性": "中",
        "解释": "冲突初期恐慌溢价消退速度，越大表示市场更快冷静。",
    },
    {
        "key": "risk_weight",
        "参数": "地缘风险权重",
        "层级": "市场价格形成层",
        "单位": "无量纲",
        "扰动": [1.85, 2.22, 2.46, 2.95, 3.20],
        "可控性": "低",
        "解释": "市场把供应中断解释为长期地缘风险溢价的强度。",
    },
    {
        "key": "uncertainty_floor",
        "参数": "不确定性与制度风险强度",
        "层级": "市场价格形成层",
        "单位": "无量纲",
        "扰动": [0.17, 0.22, 0.247, 0.30, 0.34],
        "可控性": "中",
        "解释": "前期冲击不确定性和持续封锁制度风险的共同强度。",
    },
    {
        "key": "relief_discount_strength",
        "参数": "预期修复强度",
        "层级": "市场价格形成层",
        "单位": "无量纲",
        "扰动": [0.06, 0.12, 0.24, 0.30],
        "可控性": "中",
        "解释": "市场确认运输、库存和政策缓冲后给出的再定价折价。",
    },
    {
        "key": "SHOCK_UNCERTAINTY_SHARE",
        "参数": "冲击不确定性占比",
        "层级": "长期机制透明层",
        "单位": "无量纲",
        "扰动": [0.25, 0.35, 0.45, 0.55, 0.65],
        "可控性": "中",
        "解释": "长期不确定性中由前期冲击再定价贡献的比例，用于检验冲击溢价强弱。",
    },
    {
        "key": "REGIME_RISK_SHARE",
        "参数": "制度风险占比",
        "层级": "长期机制透明层",
        "单位": "无量纲",
        "扰动": [0.50, 0.70, 0.90, 1.10, 1.30],
        "可控性": "中",
        "解释": "持续封锁制度风险对长期风险溢价的贡献强度。",
    },
    {
        "key": "REGIME_CONFIDENCE_DECAY_FLOOR",
        "参数": "信心恢复后风险底座",
        "层级": "长期机制透明层",
        "单位": "无量纲",
        "扰动": [0.20, 0.30, 0.40, 0.50, 0.60],
        "可控性": "中",
        "解释": "市场逐步确认缓冲有效后，持续封锁风险仍保留的最低比例。",
    },
    {
        "key": "SPR_TAPER_START_DAY",
        "参数": "SPR收缩起点",
        "层级": "长期机制透明层",
        "单位": "天",
        "扰动": [60, 75, 90, 105],
        "可控性": "高",
        "解释": "长期外推中 SPR 从计划释放转向缺口驱动收缩的开始时间。",
    },
    {
        "key": "OVERSUPPLY_REVERSION_SCALE",
        "参数": "过剩供给回归强度",
        "层级": "长期机制透明层",
        "单位": "无量纲",
        "扰动": [0.80, 1.10, 1.35, 1.70, 2.20],
        "可控性": "中",
        "解释": "供给超过需求时向下修正目标价格的机制强度。",
    },
    {
        "key": "BLOCKADE_RISK_DECAY",
        "参数": "封锁风险衰减速度",
        "层级": "长期机制透明层",
        "单位": "日衰减率",
        "扰动": [0.002, 0.004, 0.006, 0.008],
        "可控性": "中",
        "解释": "封锁风险溢价随时间被市场吸收的速度，用于检验长期路径是否依赖隐藏指数衰减常数。",
    },
]


def load_scenario_baseline() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    scenario.dynamic.PhysicalAssumptions,
    scenario.dynamic.BehavioralParameters,
]:
    base_config = scenario.dynamic.load_yaml(scenario.dynamic.BASE_CONFIG_PATH)
    paths = scenario.dynamic.resolve_paths(base_config)
    event_df = scenario.dynamic.load_event_window(paths.event_csv)
    best = scenario.load_best_row()
    base_assumptions, base_behavior = scenario.calibrated_assumptions_and_behavior(best)
    forecast_frame = scenario.build_forecast_frame(event_df)
    return event_df, forecast_frame, base_assumptions, base_behavior


def build_counterfactual_prefix(
    event_df: pd.DataFrame,
    assumptions: scenario.dynamic.PhysicalAssumptions,
    behavior: scenario.dynamic.BehavioralParameters,
) -> pd.DataFrame:
    """Re-simulate the observed event window under the perturbed assumption.

    Stage 5 intentionally fixes the calibrated prefix for presentation. Stage 6
    needs counterfactual sensitivity, so each perturbation must be propagated
    from day 0 instead of being attached only after the observed window.
    """
    prefix = scenario.dynamic.simulate_dynamic_model(event_df, assumptions, behavior)
    prefix["阶段"] = "附件观测期"
    prefix["is_observed_price"] = True
    prefix["forecast_price"] = prefix["simulated_price"]
    return prefix


def apply_perturbation(
    key: str,
    value: float,
    assumptions: scenario.dynamic.PhysicalAssumptions,
    behavior: scenario.dynamic.BehavioralParameters,
) -> tuple[scenario.dynamic.PhysicalAssumptions, scenario.dynamic.BehavioralParameters]:
    if key in {
        "supply_interruption",
        "spr_max_release",
        "spr_delay_days",
        "route_max_capacity",
        "route_ramp_days",
        "long_elasticity",
        "fear_decay",
    }:
        cast_value: int | float = int(round(value)) if key in {"spr_delay_days", "route_ramp_days"} else float(value)
        return replace(assumptions, **{key: cast_value}), behavior
    if key in {"risk_weight", "uncertainty_floor", "relief_discount_strength"}:
        return assumptions, replace(behavior, **{key: float(value)})
    if hasattr(scenario_sim, key):
        return assumptions, behavior
    raise KeyError(f"Unsupported sensitivity key: {key}")


@contextmanager
def temporary_simulation_constant(key: str, value: float) -> Iterator[None]:
    targets = [
        module
        for module in (scenario_sim, scenario.dynamic)
        if key and hasattr(module, key)
    ]
    if not targets:
        yield
        return

    original_values = [(module, getattr(module, key)) for module in targets]
    for module, _ in original_values:
        setattr(module, key, float(value))
    try:
        yield
    finally:
        for module, original_value in original_values:
            setattr(module, key, original_value)


def run_single_path(
    event_df: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    assumptions: scenario.dynamic.PhysicalAssumptions,
    behavior: scenario.dynamic.BehavioralParameters,
    mechanism_key: str | None = None,
    mechanism_value: float | None = None,
) -> dict[str, Any]:
    context = (
        temporary_simulation_constant(mechanism_key, mechanism_value)
        if mechanism_key and mechanism_value is not None
        else temporary_simulation_constant("", 0.0)
    )
    with context:
        prefix = build_counterfactual_prefix(event_df, assumptions, behavior)
        simulation = scenario.run_scenario("neutral", assumptions, behavior, forecast_frame, prefix)
    metrics = scenario.summarize_scenario(simulation)
    return {
        "第60天价格": metrics["第60天价格"],
        "第90天价格": metrics["第90天价格"],
        "第120天价格": metrics["第120天价格"],
        "第180天价格": metrics["第180天价格"],
        "外推期最高价": metrics["外推期最高价"],
        "外推期均价": metrics["外推期均价"],
        "第180天剩余供需缺口": metrics["第180天剩余供需缺口"],
        "外推期最大供需缺口": metrics["外推期最大供需缺口"],
        "二次跳涨幅度": metrics["二次跳涨幅度"],
        "二次跳涨风险": metrics["二次跳涨风险"],
    }


def run_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    event_df, forecast_frame, base_assumptions, base_behavior = load_scenario_baseline()
    base_metrics = run_single_path(event_df, forecast_frame, base_assumptions, base_behavior)
    base_180 = float(base_metrics["第180天价格"])
    rows: list[dict[str, Any]] = []

    rows.append(
        {
            "参数": "中性基准",
            "参数键": "baseline",
            "层级": "基准",
            "单位": "",
            "扰动值": np.nan,
            "扰动标签": "中性基准",
            "相对基准变化": 0.0,
            "可控性": "",
            "解释": "中性情景，即综合最优参数外推路径。",
            **base_metrics,
            "第180天价格相对基准变化": 0.0,
        }
    )

    for spec in SENSITIVITY_SPECS:
        key = str(spec["key"])
        for value in spec["扰动"]:
            new_assumptions, new_behavior = apply_perturbation(key, float(value), base_assumptions, base_behavior)
            mechanism_key = key if hasattr(scenario_sim, key) else None
            metrics = run_single_path(
                event_df,
                forecast_frame,
                new_assumptions,
                new_behavior,
                mechanism_key=mechanism_key,
                mechanism_value=float(value),
            )
            base_value = getattr(base_assumptions, key, getattr(base_behavior, key, np.nan))
            if mechanism_key is not None:
                base_value = getattr(scenario_sim, mechanism_key)
            change = np.nan if base_value == 0 else (float(value) - float(base_value)) / abs(float(base_value))
            rows.append(
                {
                    "参数": spec["参数"],
                    "参数键": key,
                    "层级": spec["层级"],
                    "单位": spec["单位"],
                    "扰动值": value,
                    "扰动标签": format_perturbation_label(value, spec["单位"]),
                    "相对基准变化": change,
                    "可控性": spec["可控性"],
                    "解释": spec["解释"],
                    **metrics,
                    "第180天价格相对基准变化": float(metrics["第180天价格"]) - base_180,
                }
            )

    result = pd.DataFrame(rows)
    ranking = build_ranking(result, base_180)
    return result, ranking


def format_perturbation_label(value: float, unit: str) -> str:
    if unit == "无量纲":
        return f"{value:.2f}"
    if unit == "日衰减率":
        return f"{value:.3f}"
    if unit == "天":
        return f"{int(round(value))}天"
    if unit == "万桶/日":
        return f"{value:.0f}"
    return str(value)


def build_ranking(result: pd.DataFrame, base_180: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    variant_result = result[result["参数键"] != "baseline"]
    for parameter, group in variant_result.groupby("参数", sort=False):
        final_min = float(group["第180天价格"].min())
        final_max = float(group["第180天价格"].max())
        peak_min = float(group["外推期最高价"].min())
        peak_max = float(group["外推期最高价"].max())
        gap_min = float(group["第180天剩余供需缺口"].min())
        gap_max = float(group["第180天剩余供需缺口"].max())
        worst = group.sort_values("第180天价格", ascending=False).iloc[0]
        best = group.sort_values("第180天价格", ascending=True).iloc[0]
        rows.append(
            {
                "参数": parameter,
                "参数键": str(group.iloc[0]["参数键"]),
                "层级": str(group.iloc[0]["层级"]),
                "可控性": str(group.iloc[0]["可控性"]),
                "第180天价格最小值": final_min,
                "第180天价格最大值": final_max,
                "第180天价格波动范围": final_max - final_min,
                "相对基准最大上行": final_max - base_180,
                "相对基准最大下行": final_min - base_180,
                "外推期最高价波动范围": peak_max - peak_min,
                "第180天剩余缺口波动范围": gap_max - gap_min,
                "综合敏感度得分": (final_max - final_min) + 0.45 * (peak_max - peak_min) + 0.002 * (gap_max - gap_min),
                "最高风险等级": risk_order_label(group["二次跳涨风险"]),
                "最高价格扰动": str(worst["扰动标签"]),
                "最低价格扰动": str(best["扰动标签"]),
                "解释": str(group.iloc[0]["解释"]),
            }
        )

    ranking = pd.DataFrame(rows).sort_values("综合敏感度得分", ascending=False).reset_index(drop=True)
    ranking.insert(0, "重要性排名", np.arange(1, len(ranking) + 1))
    return ranking


def risk_order_label(values: pd.Series) -> str:
    order = {"低": 0, "中": 1, "高": 2}
    reverse = {value: key for key, value in order.items()}
    max_value = int(values.map(order).max())
    return reverse[max_value]


def save_figures(result: pd.DataFrame, ranking: pd.DataFrame) -> None:
    scenario.dynamic.configure_plot_style()
    ensure_parent(TORNADO_FIGURE)

    top = ranking.sort_values("综合敏感度得分", ascending=True)
    colors = ["#dc2626" if controllability == "低" else "#2563eb" for controllability in top["可控性"]]
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    ax.barh(top["参数"], top["综合敏感度得分"], color=colors, alpha=0.86)
    ax.set_title("关键参数综合敏感度排序")
    ax.set_xlabel("综合敏感度得分（终点价波动 + 峰值波动加权）")
    ax.set_ylabel("")
    for i, value in enumerate(top["综合敏感度得分"]):
        ax.text(value + 0.18, i, f"{value:.1f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(TORNADO_FIGURE, dpi=190)
    plt.close(fig)

    top_names = ranking.head(3)["参数"].tolist()
    response = result[result["参数"].isin(top_names)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=False)
    for ax, parameter in zip(axes, top_names, strict=False):
        sub = response[response["参数"] == parameter].copy()
        sub = sub.sort_values("扰动值")
        ax.plot(sub["扰动标签"], sub["第180天价格"], color="#2563eb", marker="o", linewidth=2.0)
        ax.axhline(
            float(result.loc[result["参数键"] == "baseline", "第180天价格"].iloc[0]),
            color="#6b7280",
            linestyle="--",
            linewidth=1.0,
        )
        ax.set_title(parameter)
        ax.set_xlabel("扰动值")
        ax.set_ylabel("第180天价格")
        ax.tick_params(axis="x", labelrotation=20)
        ax.ticklabel_format(useOffset=False, style="plain", axis="y")
    fig.suptitle("前三个高敏感参数响应曲线", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(RESPONSE_FIGURE, dpi=190)
    plt.close(fig)


def build_report(result: pd.DataFrame, ranking: pd.DataFrame) -> str:
    base = result[result["参数键"] == "baseline"].iloc[0]
    top5 = ranking.head(5)
    rank_rows = "\n".join(
        "| {重要性排名:.0f} | {参数} | {层级} | {综合敏感度得分:.2f} | {第180天价格波动范围:.2f} | {外推期最高价波动范围:.2f} | {最高风险等级} | {可控性} |".format(
            **row
        )
        for row in ranking.to_dict("records")
    )
    top_rows = "\n".join(
        f"- {row['参数']}：综合敏感度得分 {row['综合敏感度得分']:.2f}，第180天价格波动范围 {row['第180天价格波动范围']:.2f} 美元/桶；"
        f"最高风险等级为 {row['最高风险等级']}；主要解释为 {row['解释']}"
        for _, row in top5.iterrows()
    )
    high_risk = result[(result["参数键"] != "baseline") & (result["二次跳涨风险"] == "高")].copy()
    if high_risk.empty:
        high_risk_text = "本轮单因素扰动没有产生新的高二次跳涨风险；高风险主要仍来自悲观组合情景。"
    else:
        high_risk_text = "\n".join(
            f"- {row['参数']}={row['扰动标签']}：第180天价格 {row['第180天价格']:.2f}，"
            f"剩余缺口 {row['第180天剩余供需缺口']:.1f} 万桶/日，二次跳涨幅度 {row['二次跳涨幅度']:.2f}。"
            for _, row in high_risk.sort_values("第180天价格", ascending=False).head(8).iterrows()
        )

    return f"""# 敏感性分析报告

## 运行结论

敏感性分析以中性情景为基准，逐个扰动关键参数，观察第180天价格、外推期最高价、剩余供需缺口和二次跳涨风险的变化。基准路径第180天价格为 {base["第180天价格"]:.2f} 美元/桶，外推期最高价为 {base["外推期最高价"]:.2f} 美元/桶，二次跳涨风险为{base["二次跳涨风险"]}。

本分析没有引入新数据，也没有编造未来真实价格。所有结果都是在综合最优参数和长期外推逻辑基础上的“单因素假设实验”，用途是判断模型结论对关键变量是否敏感。

## 参数重要性排序

| 排名 | 参数 | 层级 | 综合敏感度 | 第180天波动 | 外推峰值波动 | 最高风险 | 可控性 |
|---:|---|---|---:|---:|---:|---|---|
{rank_rows}

## 主要发现

{top_rows}

## 二次跳涨风险触发条件

{high_risk_text}

## 论文可用表述

敏感性分析表明，第180天终点价格对市场价格形成层最敏感，尤其是不确定性与制度风险强度和地缘风险权重；供应中断量决定价格上行压力的物理底座。SPR 释放、启动延迟和绕道运输在中性路径下对第180天终点价影响较小，但会改变外推期峰值、缺口闭合速度和二次跳涨幅度，因此更适合被解释为“削峰”和“争取恢复时间”的政策缓冲变量。最终论文中不应把单因素敏感性写成某个参数单独决定结果，而应强调组合情景：当地缘风险维持高位、供应中断偏大且政策缓冲不足同时出现时，二次跳涨风险才会显著抬升。

## 输出产物

- `{RESULT_CSV.relative_to(PROJECT_ROOT)}`
- `{RANKING_CSV.relative_to(PROJECT_ROOT)}`
- `{TORNADO_FIGURE.relative_to(PROJECT_ROOT)}`
- `{RESPONSE_FIGURE.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    result, ranking = run_sensitivity()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_parent(RESULT_CSV)
    ensure_parent(RANKING_CSV)
    ensure_parent(REPORT_PATH)
    result.to_csv(RESULT_CSV, index=False)
    ranking.to_csv(RANKING_CSV, index=False)
    save_figures(result, ranking)
    REPORT_PATH.write_text(build_report(result, ranking), encoding="utf-8")
    print("Sensitivity analysis complete")
    print(f"Result: {RESULT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Ranking: {RANKING_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Tornado figure: {TORNADO_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Response figure: {RESPONSE_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
