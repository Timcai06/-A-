"""Short-term dynamic model quality audit.

This audit follows the Cursor review suggestions:
- compare calendar-day and trading-day mechanism clocks with the same parameters;
- run same-parameter leave-one-chunk-out diagnostics;
- save reproducible CSV, figure, and Markdown evidence.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration.evaluation import evaluate_simulation, segment_error_rows
from src.common.metrics import mae, rmse
from src.common.paths import PROJECT_ROOT, ensure_parent
from src.common.plotting import PAPER_COLORS, SCENARIO_COLORS, configure_plot_style
from src.models import dynamic_short_term as dynamic


BEST_PARAMETERS_CSV = PROJECT_ROOT / "output" / "calibration" / "动态模型最优参数.csv"
TIME_AXIS_CSV = PROJECT_ROOT / "output" / "calibration" / "短期模型时间轴对比.csv"
TIME_AXIS_SEGMENT_CSV = PROJECT_ROOT / "output" / "calibration" / "短期模型时间轴分段误差.csv"
CHUNK_CSV = PROJECT_ROOT / "output" / "calibration" / "短期模型删块稳健性.csv"
FIGURE_PATH = PROJECT_ROOT / "paper" / "figures" / "短期模型时间轴与删块稳健性.png"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "短期递推模型质量加固报告.md"

INT_ASSUMPTION_FIELDS = {
    "spr_delay_days",
    "spr_ramp_days",
    "route_start_day",
    "route_ramp_days",
    "demand_decline_ramp_days",
}
INT_BEHAVIOR_FIELDS = {
    "buffer_relief_decay_days",
    "relief_start_day",
    "relief_peak_day",
    "relief_decay_days",
}


def load_best_parameters() -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]:
    if not BEST_PARAMETERS_CSV.exists():
        raise FileNotFoundError(f"Best-parameter CSV not found: {BEST_PARAMETERS_CSV}")

    row = pd.read_csv(BEST_PARAMETERS_CSV).iloc[0]
    assumption_kwargs = {}
    for field in fields(dynamic.PhysicalAssumptions):
        value = row[f"assumption_{field.name}"]
        assumption_kwargs[field.name] = int(round(value)) if field.name in INT_ASSUMPTION_FIELDS else float(value)

    behavior_kwargs = {}
    for field in fields(dynamic.BehavioralParameters):
        value = row[f"behavior_{field.name}"]
        behavior_kwargs[field.name] = int(round(value)) if field.name in INT_BEHAVIOR_FIELDS else float(value)

    return dynamic.PhysicalAssumptions(**assumption_kwargs), dynamic.BehavioralParameters(**behavior_kwargs)


def load_event_window() -> pd.DataFrame:
    base_config = dynamic.load_yaml(dynamic.BASE_CONFIG_PATH)
    paths = dynamic.resolve_paths(base_config)
    return dynamic.load_event_window(paths.event_csv)


def build_metric_row(label: str, simulation: pd.DataFrame) -> dict[str, float | str]:
    metrics = evaluate_simulation(simulation)
    return {
        "时间轴口径": label,
        "样本数": len(simulation),
        "最大day_index": int(simulation["day_index"].max()),
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "峰值误差": metrics["峰值误差"],
        "末日误差": metrics["末日误差"],
        "高价平台RMSE": metrics["高价平台RMSE"],
        "低价回落RMSE": metrics["低价回落RMSE"],
        "前期RMSE": metrics["前期RMSE"],
        "中期RMSE": metrics["中期RMSE"],
        "后期RMSE": metrics["后期RMSE"],
        "综合得分": metrics["综合得分"],
        "平台解释得分": metrics["平台解释得分"],
    }


def compare_time_axes(
    event_df: pd.DataFrame,
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    simulations = {
        "日历日口径": dynamic.simulate_dynamic_model(event_df, assumptions, behavior, time_axis="calendar"),
        "交易日序号口径": dynamic.simulate_dynamic_model(event_df, assumptions, behavior, time_axis="trading_day"),
    }
    metric_rows = [build_metric_row(label, simulation) for label, simulation in simulations.items()]
    segment_rows: list[dict[str, float | str]] = []
    for label, simulation in simulations.items():
        for row in segment_error_rows(simulation):
            segment_rows.append({"时间轴口径": label, **row})
    return pd.DataFrame(metric_rows), pd.DataFrame(segment_rows), simulations


def evaluate_slice(label: str, subset: pd.DataFrame) -> dict[str, float | str]:
    error = subset["simulated_price"] - subset["actual_price"]
    return {
        "样本口径": label,
        "样本数": len(subset),
        "起始日期": subset["trade_date"].min().date().isoformat(),
        "结束日期": subset["trade_date"].max().date().isoformat(),
        "RMSE": rmse(error),
        "MAE": mae(error),
        "平均偏差": float(error.mean()),
        "最大绝对误差": float(error.abs().max()),
        "实际最高价": float(subset["actual_price"].max()),
        "模拟最高价": float(subset["simulated_price"].max()),
    }


def leave_one_chunk_out(simulation: pd.DataFrame, chunks: int = 5) -> pd.DataFrame:
    indices = np.arange(len(simulation))
    rows: list[dict[str, float | str]] = []
    for chunk_id, holdout_idx in enumerate(np.array_split(indices, chunks), start=1):
        holdout = simulation.iloc[holdout_idx].copy()
        train_like = simulation.drop(simulation.index[holdout_idx]).copy()
        rows.append({"删块编号": chunk_id, **evaluate_slice("被删除块", holdout)})
        rows.append({"删块编号": chunk_id, **evaluate_slice("保留样本", train_like)})
    return pd.DataFrame(rows)


def save_figure(simulations: dict[str, pd.DataFrame], chunk_df: pd.DataFrame) -> None:
    ensure_parent(FIGURE_PATH)
    configure_plot_style()

    calendar = simulations["日历日口径"]
    trading_day = simulations["交易日序号口径"]
    holdout = chunk_df[chunk_df["样本口径"] == "被删除块"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    ax.plot(calendar["trade_date"], calendar["actual_price"], color=SCENARIO_COLORS["actual"], linewidth=2.0, label="实际收盘价")
    ax.plot(calendar["trade_date"], calendar["simulated_price"], color=SCENARIO_COLORS["fit"], linewidth=1.8, label="日历日口径")
    ax.plot(
        trading_day["trade_date"],
        trading_day["simulated_price"],
        color=SCENARIO_COLORS["neutral"],
        linewidth=1.8,
        linestyle="--",
        label="交易日序号口径",
    )
    ax.set_title("时间轴口径对短期递推路径的影响")
    ax.set_xlabel("日期")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.bar(holdout["删块编号"].astype(str), holdout["RMSE"], color=SCENARIO_COLORS["buffer"])
    ax.axhline(calendar_pipe_rmse := rmse(calendar["simulated_price"] - calendar["actual_price"]), color=PAPER_COLORS["ink"], linewidth=1.5)
    ax.text(0.02, 0.93, f"全窗口RMSE={calendar_pipe_rmse:.2f}", transform=ax.transAxes, fontsize=10)
    ax.set_title("同参删块检验：被删除块误差")
    ax.set_xlabel("被删除块编号")
    ax.set_ylabel("RMSE")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)


def build_report(time_axis_df: pd.DataFrame, segment_df: pd.DataFrame, chunk_df: pd.DataFrame) -> str:
    calendar = time_axis_df[time_axis_df["时间轴口径"] == "日历日口径"].iloc[0]
    trading = time_axis_df[time_axis_df["时间轴口径"] == "交易日序号口径"].iloc[0]
    rmse_delta = float(trading["RMSE"] - calendar["RMSE"])
    better = "更低" if rmse_delta < 0 else "更高"
    holdout = chunk_df[chunk_df["样本口径"] == "被删除块"].copy()
    holdout_rmse_std = float(holdout["RMSE"].std(ddof=0))
    worst_holdout = holdout.sort_values("RMSE", ascending=False).iloc[0]

    time_axis_table = time_axis_df[
        ["时间轴口径", "最大day_index", "RMSE", "MAE", "高价平台RMSE", "低价回落RMSE", "综合得分"]
    ].to_markdown(index=False, floatfmt=".3f")
    chunk_table = holdout[
        ["删块编号", "起始日期", "结束日期", "样本数", "RMSE", "MAE", "最大绝对误差"]
    ].to_markdown(index=False, floatfmt=".3f")
    segment_table = segment_df[
        (segment_df["时间轴口径"] == "日历日口径") & (segment_df["分段"] != "全窗口")
    ][["分段", "样本数", "RMSE", "MAE", "平均偏差", "最大绝对误差"]].to_markdown(index=False, floatfmt=".3f")

    return f"""# 短期递推模型质量加固报告

## 结论摘要

本轮针对 Cursor 审计提出的 A 级建议，完成了时间轴语义复核和同参删块稳健性检验。所有检验均使用当前 `output/calibration/动态模型最优参数.csv` 中的综合最优参数，不重新调参，因此结果反映的是现有短期主模型的结构稳定性。

- 日历日口径仍是当前主口径，RMSE 为 {float(calendar["RMSE"]):.3f}。
- 交易日序号口径在同参条件下 RMSE 为 {float(trading["RMSE"]):.3f}，比日历日口径{better} {abs(rmse_delta):.3f}。
- 同参删块检验中，被删除块 RMSE 标准差为 {holdout_rmse_std:.3f}，最难片段为第 {int(worst_holdout["删块编号"])} 块（{worst_holdout["起始日期"]} 至 {worst_holdout["结束日期"]}），RMSE 为 {float(worst_holdout["RMSE"]):.3f}。

## 时间轴语义复核

当前短期递推以日历日差作为 `day_index`，因此周末和节假日会让 SPR、绕道、恐慌衰减等机制按真实经过时间推进。交易日序号口径则把每一行行情都视为一步，适合纯金融收益序列，但会低估周末期间政策和运输机制的真实推进。

{time_axis_table}

解释：若交易日序号口径显著优于日历日口径，说明当前政策 ramp 可能被周末跳变扭曲；若日历日口径持平或更优，则说明现有机制更适合按真实经过时间解释封锁冲击。

## 日历日口径分段误差

{segment_table}

## 同参删块稳健性

将 46 个交易日按时间顺序切为 5 个连续块，每次只评估一个被删除块的误差，不重新调参。这个检验用于观察当前参数是否只服务于某一小段数据。

{chunk_table}

## 对 Cursor 审计的落实

- 时间轴问题：已提供 `calendar` 与 `trading_day` 两个可复算口径。
- 硬编码问题：本轮先不继续增参，避免把审计变成新一轮过拟合；后续应优先将恐慌通道系数和恐慌差分项纳入有界敏感性分析。
- 损失函数问题：当前分段误差已输出，下一轮若继续校准，应优先针对最差分段调整权重，而不是无边界扩展模型维度。

## 输出产物

- `{TIME_AXIS_CSV.relative_to(PROJECT_ROOT)}`
- `{TIME_AXIS_SEGMENT_CSV.relative_to(PROJECT_ROOT)}`
- `{CHUNK_CSV.relative_to(PROJECT_ROOT)}`
- `{FIGURE_PATH.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    event_df = load_event_window()
    assumptions, behavior = load_best_parameters()
    time_axis_df, segment_df, simulations = compare_time_axes(event_df, assumptions, behavior)
    chunk_df = leave_one_chunk_out(simulations["日历日口径"])

    ensure_parent(TIME_AXIS_CSV)
    time_axis_df.to_csv(TIME_AXIS_CSV, index=False)
    segment_df.to_csv(TIME_AXIS_SEGMENT_CSV, index=False)
    chunk_df.to_csv(CHUNK_CSV, index=False)
    save_figure(simulations, chunk_df)

    ensure_parent(REPORT_PATH)
    REPORT_PATH.write_text(build_report(time_axis_df, segment_df, chunk_df), encoding="utf-8")

    print("Short-term model quality audit complete")
    print(f"Time-axis comparison: {TIME_AXIS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Chunk robustness: {CHUNK_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
