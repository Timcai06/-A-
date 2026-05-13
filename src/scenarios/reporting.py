"""Stage 5 scenario output writers and report builders."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parent
from src.models import dynamic_short_term as dynamic
from src.scenarios.settings import (
    BUFFER_FIGURE,
    EXTERNAL_CONSTRAINTS_CSV,
    FORECAST_END_DAY,
    REPORT_PATH,
    SCENARIO_METRICS_CSV,
    SCENARIO_NAMES,
    SCENARIO_PARAMS_CSV,
    SCENARIO_PRICE_FIGURE,
    SCENARIO_RESULT_CSV,
)


def save_figures(result: pd.DataFrame) -> None:
    dynamic.configure_plot_style()
    ensure_parent(SCENARIO_PRICE_FIGURE)

    colors = {
        "optimistic": "#059669",
        "neutral": "#2563eb",
        "pessimistic": "#dc2626",
    }
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    observed = result[(result["scenario"] == "neutral") & (result["is_observed_price"])]
    observed_end = observed["trade_date"].max()
    ax.plot(
        observed["trade_date"],
        observed["actual_price"],
        color="#111827",
        linewidth=1.8,
        marker="o",
        markersize=2.5,
        label="附件真实收盘价",
    )
    for key, label in SCENARIO_NAMES.items():
        sub = result[result["scenario"] == key]
        plot_sub = sub[sub["trade_date"] >= observed_end]
        ax.plot(
            plot_sub["trade_date"],
            plot_sub["forecast_price"],
            color=colors[key],
            linewidth=2.0,
            label=label,
        )
    ax.axvline(observed_end, color="#6b7280", linestyle="--", linewidth=1.0, label="附件数据截止")
    ax.axhspan(110, 120, color="#f59e0b", alpha=0.10, label="110-120美元/桶参考区间")
    ax.set_title("60-180天三情景原油价格路径")
    ax.set_xlabel("日期")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper left", ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(SCENARIO_PRICE_FIGURE, dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True)
    for key, label in SCENARIO_NAMES.items():
        sub = result[result["scenario"] == key]
        plot_sub = sub[sub["trade_date"] >= observed_end]
        axes[0].plot(plot_sub["trade_date"], plot_sub["inventory_remaining"], color=colors[key], linewidth=2.0, label=label)
        axes[1].plot(plot_sub["trade_date"], plot_sub["supply_gap"], color=colors[key], linewidth=2.0, label=label)
    axes[0].set_title("商业库存剩余量")
    axes[0].set_ylabel("万桶")
    axes[0].legend(loc="upper right")
    axes[1].set_title("剩余供需缺口")
    axes[1].set_xlabel("日期")
    axes[1].set_ylabel("万桶/日")
    for ax in axes:
        ax.axvline(observed_end, color="#6b7280", linestyle="--", linewidth=1.0)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(BUFFER_FIGURE, dpi=190)
    plt.close(fig)


def build_report(metrics: pd.DataFrame, params: pd.DataFrame, event_df: pd.DataFrame) -> str:
    event_start = event_df["trade_date"].min().date()
    observed_end = event_df["trade_date"].max().date()
    forecast_end = (event_df["trade_date"].min() + pd.Timedelta(days=FORECAST_END_DAY)).date()

    metric_rows = "\n".join(
        "| {情景} | {第60天价格:.2f} | {第90天价格:.2f} | {第120天价格:.2f} | {第180天价格:.2f} | {外推期最高价:.2f} | {外推期均价:.2f} | {二次跳涨风险} |".format(
            **row
        )
        for row in metrics.to_dict("records")
    )
    param_rows = "\n".join(
        "| {情景} | {assumption_supply_interruption:.0f} | {assumption_spr_max_release:.0f} | {assumption_route_max_capacity:.0f} | {assumption_long_elasticity:.2f} | {assumption_fear_decay:.2f} | {behavior_risk_weight:.2f} | {behavior_uncertainty_floor:.2f} |".format(
            **row
        )
        for row in params.to_dict("records")
    )
    constraints_text = ""
    if EXTERNAL_CONSTRAINTS_CSV.exists():
        constraints = pd.read_csv(EXTERNAL_CONSTRAINTS_CSV)
        constraint_rows = "\n".join(
            "| {中文含义} | {数值:.4f} |".format(**row)
            for row in constraints.to_dict("records")
        )
        constraints_text = f"""
## 官方外生约束因子

本轮长期预测已把 EIA/JODI/OPEC 和 OVX 从论文证据层推进到模型参数层。约束因子只调整长期情景参数，不替代附件真实价格，也不把 JODI 多国上报口径包装成完整全球总量。OVX 只约束风险权重和不确定性强度，不作为同日价格方向预测变量。

| 约束项 | 乘数 |
|---|---:|
{constraint_rows}
"""

    return f"""# 三情景预测报告

## 运行结论

阶段 5 已基于阶段 4 综合最优短期动态模型，构建乐观、中性、悲观三条 60-180 天价格路径。预测口径为：从冲突窗口起点 {event_start} 推进到第 {FORECAST_END_DAY} 天，即 {forecast_end}；附件真实价格只覆盖到 {observed_end}，之后均为情景外推结果，不作为真实观测数据。

本阶段没有新增新闻爬虫数据，也没有编造未来真实价格。模型继续使用附件 CSV 的历史价格作为校准基础，情景差异来自赛题范围内的供应中断、SPR 释放、绕道恢复、需求弹性、恐慌衰减和市场预期修复强度。本轮引入 EIA/JODI/OPEC 官方外生数据和 OVX 市场隐含波动率生成长期参数约束因子，使绕道能力、库存缓冲、需求调整、风险权重和不确定性强度同时受到数量级审计。

为回应长期外推中“SPR 是否被机械满额释放”和“不确定性溢价是否形成常数托底”的风险，本阶段对外推逻辑做了两点增强：第一，SPR 释放不再等同于计划上限，而是随剩余物理缺口、价格压力和持续时间自动收缩；第二，不确定性溢价拆分为前期冲击不确定性和持续封锁制度风险，前者随时间衰减，后者由未解决的封锁压力决定。

## 关键预测结果

| 情景 | 第60天 | 第90天 | 第120天 | 第180天 | 外推期最高价 | 外推期均价 | 二次跳涨风险 |
|---|---:|---:|---:|---:|---:|---:|---|
{metric_rows}

## 情景参数差异

| 情景 | 供应中断 | SPR释放上限 | 绕道能力 | 长期需求弹性 | 恐慌衰减 | 风险权重 | 不确定性/制度风险强度 |
|---|---:|---:|---:|---:|---:|---:|---:|
{param_rows}

{constraints_text}

## 解释

- 乐观情景假设 SPR 可用上限更高、绕道运输恢复更快、需求收缩更充分；当缺口被覆盖后，实际 SPR 释放会自动回落，因此外推期价格逐步向低位收敛。
- 中性情景沿用阶段 4 的综合最优校准参数，是后续论文和敏感性分析的主基准路径。
- 悲观情景假设 SPR 释放弱、绕道运输慢、需求收缩不足且恐慌衰减较慢，因此物理缺口和制度风险溢价同时存在，价格维持更高平台，并存在更强的二次跳涨风险。

## 输出产物

- `{SCENARIO_RESULT_CSV.relative_to(PROJECT_ROOT)}`
- `{SCENARIO_METRICS_CSV.relative_to(PROJECT_ROOT)}`
- `{SCENARIO_PARAMS_CSV.relative_to(PROJECT_ROOT)}`
- `{EXTERNAL_CONSTRAINTS_CSV.relative_to(PROJECT_ROOT)}`
- `{SCENARIO_PRICE_FIGURE.relative_to(PROJECT_ROOT)}`
- `{BUFFER_FIGURE.relative_to(PROJECT_ROOT)}`

## 后续作用

阶段 5 已回答“封锁持续 60-180 天时油价路径如何演化”的预测问题。阶段 6 可以在此基础上围绕供应中断、SPR 释放、需求弹性、风险权重和预期修复强度做敏感性分析。
"""


def write_outputs(result: pd.DataFrame, metrics: pd.DataFrame, params: pd.DataFrame, event_df: pd.DataFrame) -> None:
    ensure_parent(SCENARIO_RESULT_CSV)
    ensure_parent(SCENARIO_METRICS_CSV)
    ensure_parent(SCENARIO_PARAMS_CSV)
    ensure_parent(REPORT_PATH)
    result.to_csv(SCENARIO_RESULT_CSV, index=False)
    metrics.to_csv(SCENARIO_METRICS_CSV, index=False)
    params.to_csv(SCENARIO_PARAMS_CSV, index=False)
    save_figures(result)
    REPORT_PATH.write_text(build_report(metrics, params, event_df), encoding="utf-8")
