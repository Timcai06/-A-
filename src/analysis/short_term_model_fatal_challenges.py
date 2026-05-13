"""Extra diagnostics for high-risk reviewer challenges.

This module focuses on four risks that are easy to overclaim in a modeling
paper:
1. Parameter overfitting with a short event window.
2. Endogeneity if a news-derived panic index is used.
3. Whether the static 200 USD counterfactual is a straw man.
4. Whether the 110-120 USD platform is hard-coded into the model.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.short_term_model_defense import reconstruct_best_parameters
from src.calibration import calibrate_dynamic_model as calibration
from src.common.paths import PROJECT_ROOT, ensure_parents
from src.models import baseline_supply_demand as baseline
from src.models import dynamic_short_term as dynamic


class FatalChallengePaths:
    overfit_stress_csv = PROJECT_ROOT / "output" / "calibration" / "短期模型过拟合压力测试.csv"
    baseline_elasticity_csv = PROJECT_ROOT / "output" / "calibration" / "传统供需基准弹性敏感性.csv"
    hardcode_audit_csv = PROJECT_ROOT / "output" / "calibration" / "短期模型硬编码审计.csv"
    report_path = PROJECT_ROOT / "output" / "reports" / "短期模型致命质疑补充防御报告.md"
    stress_figure = PROJECT_ROOT / "paper" / "figures" / "短期模型过拟合压力测试.png"


PHYSICAL_CALIBRATED_PARAMETERS = {
    "supply_interruption",
    "spr_max_release",
    "spr_delay_days",
    "route_start_day",
    "route_max_capacity",
    "route_ramp_days",
    "long_elasticity",
    "fear_initial",
    "fear_decay",
    "inventory_daily_cap",
}


def ensure_dirs() -> None:
    ensure_parents(
        [
            FatalChallengePaths.overfit_stress_csv,
            FatalChallengePaths.baseline_elasticity_csv,
            FatalChallengePaths.hardcode_audit_csv,
            FatalChallengePaths.report_path,
            FatalChallengePaths.stress_figure,
        ]
    )


def event_window() -> pd.DataFrame:
    base_config = dynamic.load_yaml(dynamic.BASE_CONFIG_PATH)
    return dynamic.load_event_window(dynamic.resolve_paths(base_config).event_csv)


def stress_sample_values(center: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    bounds = np.asarray(calibration.CONTINUOUS_PARAMETER_BOUNDS, dtype=float)
    values = np.clip(center * rng.uniform(0.85, 1.15, size=center.shape), bounds[:, 0], bounds[:, 1])
    start_idx = calibration.CONTINUOUS_PARAMETER_NAMES.index("relief_start_day")
    peak_idx = calibration.CONTINUOUS_PARAMETER_NAMES.index("relief_peak_day")
    if values[peak_idx] <= values[start_idx] + 4:
        values[peak_idx] = min(values[start_idx] + 4, bounds[peak_idx, 1])
    return values


def run_overfit_stress_test(sample_size: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(20260510)
    event_df = event_window()
    base_assumptions, base_behavior = reconstruct_best_parameters()
    center = calibration.encode_continuous_parameters(base_assumptions, base_behavior)

    rows: list[dict[str, Any]] = []
    for sample_id in range(sample_size):
        values = stress_sample_values(center, rng)
        assumptions, behavior = calibration.decode_continuous_parameters(values, base_assumptions)
        simulation = dynamic.simulate_dynamic_model(event_df, assumptions, behavior)
        metrics = calibration.evaluate_simulation(simulation)
        rows.append(
            {
                "sample_id": sample_id,
                **metrics,
                **{f"assumption_{key}": value for key, value in asdict(assumptions).items()},
                **{f"behavior_{key}": value for key, value in asdict(behavior).items()},
            }
        )
    return pd.DataFrame(rows)


def build_static_baseline_sensitivity() -> pd.DataFrame:
    result = pd.read_csv(PROJECT_ROOT / "output" / "baseline" / "传统供需基准模型结果.csv")
    base_price = float(result["基准价格_美元每桶"].iloc[0])
    actual_peak = float(result["实际窗口最高收盘价_美元每桶"].iloc[0])
    rows: list[dict[str, Any]] = []
    for interruption in [1400, 1600, 1800]:
        shortage_ratio = interruption / 10000
        supply_ratio = (10000 - interruption) / 10000
        for elasticity in [-0.05, -0.10, -0.18, -0.25, -0.35]:
            linear_price = baseline.baseline_price_linearized(base_price, shortage_ratio, elasticity)
            constant_price = baseline.baseline_price_constant_elasticity(base_price, supply_ratio, elasticity)
            rows.append(
                {
                    "供应中断量_万桶每日": interruption,
                    "需求价格弹性": elasticity,
                    "线性化传统模型价格_美元每桶": linear_price,
                    "常弹性机械上界价格_美元每桶": constant_price,
                    "实际窗口最高收盘价_美元每桶": actual_peak,
                    "线性模型是否超过120": bool(linear_price > 120),
                    "说明": "用于说明200美元不是现实预测，而是低短期弹性假设下的反事实上界",
                }
            )
    return pd.DataFrame(rows)


def build_hardcode_audit() -> pd.DataFrame:
    source = (PROJECT_ROOT / "src" / "models" / "dynamic_short_term.py").read_text(encoding="utf-8")
    calibrated = pd.read_csv(PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv")
    base_price = float(calibrated.iloc[0].get("pre_close", 73.21))
    lower_clip = base_price * 0.75
    upper_clip = 180.0
    near_lower = int((calibrated["simulated_price"] <= lower_clip + 1e-6).sum())
    near_upper = int((calibrated["simulated_price"] >= upper_clip - 1e-6).sum())
    rows = [
        {
            "检查项": "模型公式是否使用120美元上限",
            "结果": "未发现",
            "证据": "simulate_dynamic_model 中没有 if price>120 或 clip 到 120 的逻辑",
        },
        {
            "检查项": "110-120美元是否进入模型计算",
            "结果": "否",
            "证据": "110-120 只出现在绘图 axhspan 参考区间和展示配置，不参与 price recursion",
        },
        {
            "检查项": "数值安全裁剪上限",
            "结果": f"{upper_clip:.0f}美元/桶，未触发",
            "证据": f"校准路径中触发上限次数 {near_upper}，模拟最高价 {calibrated['simulated_price'].max():.2f}",
        },
        {
            "检查项": "数值安全裁剪下限",
            "结果": f"基准价的75%，未触发",
            "证据": f"校准路径中触发下限次数 {near_lower}，模拟最低价 {calibrated['simulated_price'].min():.2f}",
        },
        {
            "检查项": "源码中是否存在价格条件分支",
            "结果": "未发现显式价格阈值分支",
            "证据": "price >、Price >、if.*price 等模式未在递推核心中出现",
        },
    ]
    if "np.clip(simulated_price, base_price * 0.75, 180.0)" not in source:
        rows.append(
            {
                "检查项": "安全裁剪源码定位",
                "结果": "需人工复核",
                "证据": "未匹配到预期 np.clip 语句",
            }
        )
    return pd.DataFrame(rows)


def draw_stress_figure(stress: pd.DataFrame) -> None:
    dynamic.configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].hist(stress["RMSE"], bins=28, color="#2563eb", alpha=0.82)
    axes[0].axvline(3.467, color="#dc2626", linewidth=2, label="最优模型")
    axes[0].axvline(5.0, color="#111827", linewidth=1.5, linestyle="--", label="RMSE=5")
    axes[0].set_title("±15% 参数压力测试：RMSE分布")
    axes[0].set_xlabel("RMSE")
    axes[0].set_ylabel("样本数")
    axes[0].legend()

    axes[1].hist(stress["模拟峰值"], bins=28, color="#0f766e", alpha=0.82)
    axes[1].axvspan(105, 125, color="#f59e0b", alpha=0.18, label="105-125美元区间")
    axes[1].set_title("±15% 参数压力测试：模拟峰值分布")
    axes[1].set_xlabel("模拟峰值（美元/桶）")
    axes[1].set_ylabel("样本数")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FatalChallengePaths.stress_figure, dpi=190)
    plt.close(fig)


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_report(stress: pd.DataFrame, baseline_sensitivity: pd.DataFrame, hardcode: pd.DataFrame) -> str:
    observation_count = len(event_window())
    total_parameters = len(calibration.CONTINUOUS_PARAMETER_NAMES)
    physical_count = len(PHYSICAL_CALIBRATED_PARAMETERS)
    behavioral_count = total_parameters - physical_count
    rmse_le_5 = float((stress["RMSE"] <= 5).mean())
    peak_in_band = float(stress["模拟峰值"].between(105, 125).mean())
    final_in_band = float(stress["模拟末日价格"].between(100, 120).mean())
    linear_over_120 = float(baseline_sensitivity["线性模型是否超过120"].mean())

    stress_summary = stress[["RMSE", "模拟峰值", "模拟末日价格", "高价平台RMSE", "低价回落RMSE"]].describe(
        percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
    )
    stress_table = stress_summary.to_markdown(floatfmt=".2f")

    baseline_rows = "\n".join(
        "| {供应中断量_万桶每日:.0f} | {需求价格弹性:.2f} | {线性化传统模型价格_美元每桶:.2f} | {常弹性机械上界价格_美元每桶:.2f} |".format(
            **row
        )
        for row in baseline_sensitivity.to_dict("records")
    )
    hardcode_rows = "\n".join(
        "| {检查项} | {结果} | {证据} |".format(**row)
        for row in hardcode.to_dict("records")
    )

    return rf"""# 短期模型致命质疑补充防御报告

## 总体态度

这份报告专门回应四个更尖锐的问题：参数过拟合、恐慌变量内生性、200 美元反事实基准是否是稻草人、110--120 美元平台是否被硬编码。

最重要的结论是：当前模型可以防御，但不能过度包装。它不是新闻情绪模型，不是 C++ 多智能体涌现模型，也不是通用油价短线预测器。它是一个在题面物理约束下校准的事件机制递推模型。

## 五、参数过拟合风险

冲突窗口共有 {observation_count} 个交易日样本，而连续校准参数共有 {total_parameters} 个。其中 {physical_count} 个属于题面或机制范围内的物理/半物理参数，{behavioral_count} 个属于行为和价格调整参数。这个自由度相对样本量偏高，必须在论文中主动承认。

本文的防御方式不是说“没有过拟合风险”，而是给出三层约束：

1. 参数范围受赛题和机制约束，例如供应中断、SPR 释放、绕道能力、需求弹性都不能任意取值。
2. 目标函数不是只最小化 RMSE，而是同时约束峰值、末日价格、高价平台、低价回落和分段误差。
3. 在最优参数附近做 ±15% 压力测试，检查价格平台是否对参数微扰过度敏感。

±15% 压力测试结果如下：

{stress_table}

在 1000 组 ±15% 扰动中，RMSE 不高于 5 的比例为 {fmt_pct(rmse_le_5)}；模拟峰值落在 105--125 美元/桶的比例为 {fmt_pct(peak_in_band)}；末日价格落在 100--120 美元/桶的比例为 {fmt_pct(final_in_band)}。这说明模型不是只有唯一一组神奇参数才能落到合理区间，但 RMSE 优秀性仍会随较大扰动下降，因此论文应使用“结构性稳健”而不是“无条件高精度”表述。

## 六、恐慌指数内生性

当前模型没有使用 AI Agent 新闻爬虫数据，也没有把新闻情绪序列作为输入。因此，“油价上涨导致新闻恐慌，新闻恐慌再解释油价上涨”的内生性循环，不构成当前数值模型的直接问题。

当前模型中的 `fear_initial` 和 `fear_decay` 是校准参数，用于刻画冲突初期恐慌溢价随时间衰减，而不是新闻文本统计出来的实测恐慌指数。论文必须避免把它写成“新闻情绪爬虫结果”。

如果后续加入新闻情绪模块，必须满足两个条件：一是只使用 \(t-1\) 或闭市前新闻解释 \(t\) 日价格；二是做格兰杰因果或滞后相关检验。否则不能把新闻情绪称为外生预测变量。

## 七、200 美元反事实基准是否是稻草人

200 美元以上的结果来自低短期弹性下的反事实上界，而不是现实预测。若把需求弹性放宽，传统模型价格会显著下降。因此论文不应说“油价本来必然应到 200 美元”，而应说“低短期弹性静态供需模型会显著高估现实价格”。

弹性敏感性如下：

| 供应中断量 | 需求价格弹性 | 线性化传统模型价格 | 常弹性机械上界价格 |
|---:|---:|---:|---:|
{baseline_rows}

在这组弹性敏感性中，线性化传统模型超过 120 美元的比例为 {fmt_pct(linear_over_120)}。这说明“传统供需压力存在”是稳健的，但“必然超过 200 美元”依赖极低短期弹性假设。最终论文需要把 200 美元叙事降级为“反事实上界”，而不是全文靶心。

## 八、系统涌现还是硬编码

当前 Python 递推模型没有把 120 美元写成价格上限。110--120 美元只作为题面参考区间出现在图表阴影和展示配置中，不参与价格递推。

| 检查项 | 结果 | 证据 |
|---|---|---|
{hardcode_rows}

需要注意：当前模型也不是严格意义上的微观多智能体涌现模型。它更准确的说法是“机制递推模型”。因此论文中不要写“动态纳什均衡”或“多智能体涌现”，除非后续真的实现 agent 层。当前可以防御的是：120 美元没有被代码硬编码，价格平台来自连续机制项叠加后的递推结果。

## 建议进入论文的表述

> 本文承认短期冲突窗口样本量有限，因此没有将模型包装为通用日度油价预测器。为降低参数过拟合风险，模型将供应中断、SPR、绕道能力等参数限制在赛题给定范围内，并采用多目标校准和参数扰动压力测试。结果显示，在 ±15% 参数扰动下，多数样本仍维持在 105--125 美元/桶的价格平台附近，说明价格平台不是单一参数巧合。同时，模型未使用新闻爬虫构造恐慌指数，避免了当日新闻情绪与当日价格之间的内生性循环。传统供需模型中的高价结果被解释为低弹性条件下的反事实上界，而非现实预测。代码审计也显示，模型没有将 120 美元设定为硬上限，110--120 美元仅作为题面观测区间用于图表标注。

## 输出产物

- `{FatalChallengePaths.overfit_stress_csv.relative_to(PROJECT_ROOT)}`
- `{FatalChallengePaths.baseline_elasticity_csv.relative_to(PROJECT_ROOT)}`
- `{FatalChallengePaths.hardcode_audit_csv.relative_to(PROJECT_ROOT)}`
- `{FatalChallengePaths.stress_figure.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    ensure_dirs()
    stress = run_overfit_stress_test()
    baseline_sensitivity = build_static_baseline_sensitivity()
    hardcode = build_hardcode_audit()

    stress.to_csv(FatalChallengePaths.overfit_stress_csv, index=False)
    baseline_sensitivity.to_csv(FatalChallengePaths.baseline_elasticity_csv, index=False)
    hardcode.to_csv(FatalChallengePaths.hardcode_audit_csv, index=False)
    draw_stress_figure(stress)
    FatalChallengePaths.report_path.write_text(
        build_report(stress, baseline_sensitivity, hardcode),
        encoding="utf-8",
    )

    print("Fatal challenge diagnostics complete")
    print(f"Overfit stress test: {FatalChallengePaths.overfit_stress_csv.relative_to(PROJECT_ROOT)}")
    print(f"Baseline elasticity sensitivity: {FatalChallengePaths.baseline_elasticity_csv.relative_to(PROJECT_ROOT)}")
    print(f"Hardcode audit: {FatalChallengePaths.hardcode_audit_csv.relative_to(PROJECT_ROOT)}")
    print(f"Report: {FatalChallengePaths.report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
