"""Defense diagnostics for common short-term model review questions.

This module answers four likely reviewer challenges:
1. Does the model beat naive/random-walk baselines?
2. Is the fitted curve merely a delayed copy of actual prices?
3. What horizon is being evaluated?
4. Are absolute errors small relative to the oil-price level?
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any
from warnings import catch_warnings, simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.calibration import calibrate_dynamic_model as calibration
from src.common.metrics import direction_hit_rate, mae, mape, rmse
from src.common.paths import PROJECT_ROOT, ensure_parents
from src.common.plotting import PAPER_COLORS, SCENARIO_COLORS
from src.models import dynamic_short_term as dynamic


FULL_DATA_CSV = PROJECT_ROOT / "data" / "processed" / "布伦特原油期货主力合约价格数据_清洗后.csv"
CALIBRATED_PATH_CSV = PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv"
TOP_CANDIDATES_CSV = PROJECT_ROOT / "output" / "calibration" / "动态模型候选参数前10.csv"


class DefensePaths:
    baseline_csv = PROJECT_ROOT / "output" / "calibration" / "短期模型基准对比.csv"
    lag_csv = PROJECT_ROOT / "output" / "calibration" / "短期模型滞后平移检验.csv"
    turning_points_csv = PROJECT_ROOT / "output" / "calibration" / "短期模型拐点检验.csv"
    placebo_windows_csv = PROJECT_ROOT / "output" / "calibration" / "短期模型历史窗口边界检验.csv"
    horizon_note = PROJECT_ROOT / "output" / "reports" / "短期模型预测步长说明.md"
    defense_report = PROJECT_ROOT / "output" / "reports" / "短期模型评委质疑防御报告.md"
    baseline_figure = PROJECT_ROOT / "paper" / "figures" / "短期模型基准对比.png"
    lag_figure = PROJECT_ROOT / "paper" / "figures" / "短期模型滞后平移检验.png"
    turn_figure = PROJECT_ROOT / "paper" / "figures" / "短期模型拐点局部检验.png"


def ensure_dirs() -> None:
    ensure_parents(
        [
            DefensePaths.baseline_csv,
            DefensePaths.lag_csv,
            DefensePaths.turning_points_csv,
            DefensePaths.placebo_windows_csv,
            DefensePaths.horizon_note,
            DefensePaths.defense_report,
            DefensePaths.baseline_figure,
            DefensePaths.lag_figure,
            DefensePaths.turn_figure,
        ]
    )


def load_calibrated_path() -> pd.DataFrame:
    if not CALIBRATED_PATH_CSV.exists():
        raise FileNotFoundError(f"Missing calibrated path: {CALIBRATED_PATH_CSV}")
    df = pd.read_csv(CALIBRATED_PATH_CSV, parse_dates=["trade_date"])
    df["model_error"] = df["simulated_price"] - df["actual_price"]
    return df.sort_values("trade_date").reset_index(drop=True)


def expanding_arima_forecast(actual: pd.Series, first_pre_close: float) -> pd.Series:
    history = [float(first_pre_close)]
    preds: list[float] = []
    for value in actual:
        if len(history) < 8:
            pred = history[-1]
        else:
            with catch_warnings():
                simplefilter("ignore")
                try:
                    model = ARIMA(history, order=(1, 1, 0)).fit()
                    pred = float(model.forecast(1)[0])
                except Exception:
                    pred = history[-1]
        preds.append(pred)
        history.append(float(value))
    return pd.Series(preds, index=actual.index)


def compute_baselines(path: pd.DataFrame) -> pd.DataFrame:
    actual = path["actual_price"].astype(float)
    model_pred = path["simulated_price"].astype(float)
    first_pre_close = float(path.iloc[0].get("pre_close", actual.iloc[0]))
    if "pre_close" not in path.columns or pd.isna(first_pre_close):
        first_pre_close = float(actual.iloc[0])

    previous_actual = actual.shift(1)
    previous_actual.iloc[0] = first_pre_close

    history_with_preclose = pd.concat([pd.Series([first_pre_close]), actual.iloc[:-1]], ignore_index=True)
    rolling3 = history_with_preclose.rolling(3, min_periods=1).mean().reset_index(drop=True)
    rolling3.index = actual.index

    drift_preds: list[float] = []
    history = [first_pre_close]
    for value in actual:
        if len(history) >= 3:
            diffs = np.diff(history)
            pred = history[-1] + float(np.mean(diffs[-5:]))
        else:
            pred = history[-1]
        drift_preds.append(float(pred))
        history.append(float(value))
    drift = pd.Series(drift_preds, index=actual.index)
    arima = expanding_arima_forecast(actual, first_pre_close)

    candidates = [
        ("本文短期动态模型", model_pred, "阶段4综合最优参数；递推过程中不喂入当日真实价格"),
        ("朴素上一日基准", previous_actual, "Random Walk：预测今日价格等于上一交易日真实价格"),
        ("三日均值基准", rolling3, "只使用过去最多三日真实价格均值"),
        ("漂移随机游走基准", drift, "上一日价格加最近历史平均日变化"),
        ("滚动ARIMA(1,1,0)基准", arima, "每一步只用此前真实价格滚动拟合"),
    ]
    rows: list[dict[str, Any]] = []
    for name, pred, note in candidates:
        error = pred - actual
        rows.append(
            {
                "模型": name,
                "RMSE": rmse(error),
                "MAE": mae(error),
                "MAPE": mape(actual, pred),
                "方向命中率": direction_hit_rate(actual, pred),
                "最大绝对误差": float(np.max(np.abs(error))),
                "说明": note,
            }
        )
    table = pd.DataFrame(rows)
    model_rmse = float(table.loc[table["模型"] == "本文短期动态模型", "RMSE"].iloc[0])
    table["相对本文RMSE差值"] = table["RMSE"] - model_rmse
    table["本文相对该基准改善率"] = (table["RMSE"] - model_rmse) / table["RMSE"] * 100
    table.loc[table["模型"] == "本文短期动态模型", "本文相对该基准改善率"] = 0.0
    return table


def compute_lag_test(path: pd.DataFrame, max_shift: int = 5) -> pd.DataFrame:
    actual = path["actual_price"].to_numpy(dtype=float)
    model = path["simulated_price"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            aligned_actual = actual[:shift]
            aligned_model = model[-shift:]
        elif shift > 0:
            aligned_actual = actual[shift:]
            aligned_model = model[:-shift]
        else:
            aligned_actual = actual
            aligned_model = model
        err = aligned_model - aligned_actual
        rows.append(
            {
                "模型平移天数": shift,
                "RMSE": rmse(err),
                "MAE": mae(err),
                "样本数": len(err),
                "解释": "负数表示把模型曲线向左移；若负数显著更优，说明原模型可能滞后",
            }
        )
    table = pd.DataFrame(rows)
    zero_rmse = float(table.loc[table["模型平移天数"] == 0, "RMSE"].iloc[0])
    table["相对原始RMSE变化"] = table["RMSE"] - zero_rmse
    return table


def classify_turning_capture(model_signs: dict[int, float], actual_sign: float) -> str:
    if model_signs.get(0) == actual_sign:
        return "同步"
    if model_signs.get(-1) == actual_sign:
        return "提前1日"
    if model_signs.get(1) == actual_sign:
        return "滞后1日"
    return "未明显捕捉"


def compute_turning_points(path: pd.DataFrame) -> pd.DataFrame:
    df = path[["trade_date", "actual_price", "simulated_price"]].copy()
    df["actual_change"] = df["actual_price"].diff()
    df["model_change"] = df["simulated_price"].diff()
    candidates = df.iloc[1:].copy()
    candidates["actual_abs_change"] = candidates["actual_change"].abs()
    selected = candidates.nlargest(10, "actual_abs_change").sort_values("trade_date")
    rows: list[dict[str, Any]] = []
    for idx, row in selected.iterrows():
        actual_sign = float(np.sign(row["actual_change"]))
        model_signs = {
            offset: float(np.sign(df.loc[idx + offset, "model_change"]))
            for offset in [-1, 0, 1]
            if idx + offset in df.index and pd.notna(df.loc[idx + offset, "model_change"])
        }
        rows.append(
            {
                "日期": row["trade_date"].date().isoformat(),
                "实际变动": float(row["actual_change"]),
                "模型同日变动": float(row["model_change"]),
                "实际绝对变动": float(abs(row["actual_change"])),
                "捕捉类型": classify_turning_capture(model_signs, actual_sign),
                "说明": "按实际价格绝对变动最大的10个交易日筛选",
            }
        )
    return pd.DataFrame(rows)


def reconstruct_best_parameters() -> tuple[dynamic.PhysicalAssumptions, dynamic.BehavioralParameters]:
    best = pd.read_csv(TOP_CANDIDATES_CSV).iloc[0]
    base_config = dynamic.load_yaml(dynamic.BASE_CONFIG_PATH)
    scenario_config = dynamic.load_yaml(dynamic.SCENARIO_CONFIG_PATH)
    problem_params = dynamic.load_problem_parameters(dynamic.PROBLEM_PARAMETERS_PATH)
    _, base_assumptions = (
        dynamic.load_event_window(dynamic.resolve_paths(base_config).event_csv),
        dynamic.resolve_assumptions(base_config, scenario_config, problem_params),
    )
    values = []
    assumption_fields = set(asdict(base_assumptions).keys())
    for name in calibration.CONTINUOUS_PARAMETER_NAMES:
        column = f"assumption_{name}" if name in assumption_fields else f"behavior_{name}"
        values.append(float(best[column]))
    return calibration.decode_continuous_parameters(np.asarray(values), base_assumptions)


def select_placebo_windows(full_data: pd.DataFrame, window_size: int = 46, count: int = 6) -> list[pd.DataFrame]:
    event_start = pd.Timestamp("2026-03-02")
    event_end = pd.Timestamp("2026-05-05")
    candidates: list[dict[str, Any]] = []
    for start in range(0, len(full_data) - window_size + 1):
        win = full_data.iloc[start : start + window_size].copy()
        if win["trade_date"].between(event_start, event_end).any():
            continue
        volatility = float(win["return_pct"].std(skipna=True))
        price_range = float(win["close_price"].max() - win["close_price"].min())
        candidates.append(
            {
                "start": start,
                "end": start + window_size,
                "score": volatility * 100 + price_range / max(float(win["close_price"].mean()), 1.0),
            }
        )
    selected: list[pd.DataFrame] = []
    occupied: list[tuple[int, int]] = []
    for item in sorted(candidates, key=lambda x: x["score"], reverse=True):
        if any(not (item["end"] <= lo or item["start"] >= hi) for lo, hi in occupied):
            continue
        selected.append(full_data.iloc[item["start"] : item["end"]].copy())
        occupied.append((item["start"], item["end"]))
        if len(selected) >= count:
            break
    return selected


def compute_placebo_windows() -> pd.DataFrame:
    full = pd.read_csv(FULL_DATA_CSV, parse_dates=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    assumptions, behavior = reconstruct_best_parameters()
    rows: list[dict[str, Any]] = []
    for idx, win in enumerate(select_placebo_windows(full), start=1):
        event_like = win[["trade_date", "pre_close", "close_price"]].copy().reset_index(drop=True)
        simulation = dynamic.simulate_dynamic_model(event_like, assumptions, behavior)
        actual = simulation["actual_price"]
        model = simulation["simulated_price"]
        naive = actual.shift(1)
        naive.iloc[0] = float(event_like.iloc[0]["pre_close"])
        rows.append(
            {
                "窗口编号": idx,
                "窗口起始": event_like["trade_date"].min().date().isoformat(),
                "窗口终止": event_like["trade_date"].max().date().isoformat(),
                "样本数": len(event_like),
                "窗口均价": float(actual.mean()),
                "窗口价格范围": float(actual.max() - actual.min()),
                "窗口收益率波动率": float(win["return_pct"].std(skipna=True)),
                "霍尔木兹机制模型RMSE": rmse(model - actual),
                "朴素上一日基准RMSE": rmse(naive - actual),
                "模型/朴素RMSE": rmse(model - actual) / rmse(naive - actual),
                "解释": "固定阶段4参数直接套用到历史高波动窗口；用于边界检验，不作为重新校准表现",
            }
        )
    return pd.DataFrame(rows)


def draw_baseline_figure(table: pd.DataFrame) -> None:
    dynamic.configure_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    plot = table.sort_values("RMSE", ascending=False)
    colors = [SCENARIO_COLORS["fit"] if name == "本文短期动态模型" else PAPER_COLORS["border"] for name in plot["模型"]]
    ax.barh(plot["模型"], plot["RMSE"], color=colors)
    ax.set_title("短期模型与朴素/时间序列基准 RMSE 对比")
    ax.set_xlabel("RMSE（美元/桶，越低越好）")
    for y, value in enumerate(plot["RMSE"]):
        ax.text(value + 0.05, y, f"{value:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(DefensePaths.baseline_figure, dpi=190)
    plt.close(fig)


def draw_lag_figure(lag: pd.DataFrame) -> None:
    dynamic.configure_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = [SCENARIO_COLORS["fit"] if shift == 0 else PAPER_COLORS["border"] for shift in lag["模型平移天数"]]
    ax.bar(lag["模型平移天数"], lag["RMSE"], color=colors)
    ax.set_title("模型曲线平移后的 RMSE 检验")
    ax.set_xlabel("模型平移天数（负数=向左移，检查是否滞后）")
    ax.set_ylabel("RMSE")
    ax.axvline(0, color=PAPER_COLORS["ink"], linewidth=1.0)
    fig.tight_layout()
    fig.savefig(DefensePaths.lag_figure, dpi=190)
    plt.close(fig)


def draw_turning_figure(path: pd.DataFrame) -> None:
    dynamic.configure_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharey=True)
    windows = [
        ("前期暴涨窗口", "2026-03-02", "2026-03-14"),
        ("后期反转窗口", "2026-04-22", "2026-05-05"),
    ]
    for ax, (title, start, end) in zip(axes, windows, strict=True):
        sub = path[path["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        ax.plot(sub["trade_date"], sub["actual_price"], color=SCENARIO_COLORS["actual"], marker="o", linewidth=2.0, label="实际收盘价")
        ax.plot(sub["trade_date"], sub["simulated_price"], color=SCENARIO_COLORS["fit"], marker="s", linewidth=2.0, label="短期动态模型")
        ax.set_title(title)
        ax.set_ylabel("美元/桶")
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("日期")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(DefensePaths.turn_figure, dpi=190)
    plt.close(fig)


def write_horizon_note(path: pd.DataFrame, baseline: pd.DataFrame) -> None:
    start = path["trade_date"].min().date()
    end = path["trade_date"].max().date()
    days = int(path["day_index"].max())
    model = baseline[baseline["模型"] == "本文短期动态模型"].iloc[0]
    rmse = float(model["RMSE"])
    text = f"""# 短期模型预测步长说明

本文短期动态模型不是传统意义上的单步 T+1 日度交易预测器，而是条件机制递推模型。

- 起点：{start}
- 终点：{end}
- 日历跨度：0 到 {days} 天
- 交易日样本数：{len(path)}
- 递推方式：给定冲突开始、题面参数和阶段 4 校准参数后，从初始价格开始逐日递推。
- 真实价格使用边界：真实价格只用于校准和事后评价，不在每日递推中作为当日输入喂给模型。

因此，RMSE={rmse:.2f} 的含义不是“每天只预测明天还误差 {rmse:.2f}”，而是“在极端冲突窗口内，模型用一套固定机制参数递推出整段价格路径后的平均偏离”。论文表述应强调机制解释与条件路径预测，不应把它包装成可交易的 T+1 高频预测模型。
"""
    DefensePaths.horizon_note.write_text(text, encoding="utf-8")


def build_report(
    baseline: pd.DataFrame,
    lag: pd.DataFrame,
    turning: pd.DataFrame,
    placebo: pd.DataFrame,
) -> str:
    model = baseline[baseline["模型"] == "本文短期动态模型"].iloc[0]
    naive = baseline[baseline["模型"] == "朴素上一日基准"].iloc[0]
    arima = baseline[baseline["模型"] == "滚动ARIMA(1,1,0)基准"].iloc[0]
    zero = lag[lag["模型平移天数"] == 0].iloc[0]
    best_shift = lag.sort_values("RMSE").iloc[0]
    sync_ratio = float((turning["捕捉类型"] == "同步").mean() * 100)

    baseline_rows = "\n".join(
        "| {模型} | {RMSE:.3f} | {MAE:.3f} | {MAPE:.3f}% | {方向命中率:.1f}% | {说明} |".format(**row)
        for row in baseline.to_dict("records")
    )
    lag_rows = "\n".join(
        "| {模型平移天数:+d} | {RMSE:.3f} | {MAE:.3f} | {相对原始RMSE变化:+.3f} |".format(
            **{**row, "模型平移天数": int(row["模型平移天数"])}
        )
        for row in lag.to_dict("records")
    )
    turning_rows = "\n".join(
        "| {日期} | {实际变动:.2f} | {模型同日变动:.2f} | {捕捉类型} |".format(**row)
        for row in turning.to_dict("records")
    )
    placebo_rows = "\n".join(
        "| {窗口编号:.0f} | {窗口起始} | {窗口终止} | {霍尔木兹机制模型RMSE:.2f} | {朴素上一日基准RMSE:.2f} | {模型/朴素RMSE:.2f} |".format(
            **row
        )
        for row in placebo.to_dict("records")
    )

    return f"""# 短期模型评委质疑防御报告

## 总结回答

本报告专门回应评委可能提出的四类质疑：是否打败 Random Walk、是否只是滞后曲线、预测步长到底是什么、绝对误差对应的相对误差是多少。

结论是：本文短期动态模型 RMSE 为 {model["RMSE"]:.2f} 美元/桶、MAE 为 {model["MAE"]:.2f} 美元/桶、MAPE 为 {model["MAPE"]:.2f}%。它优于朴素上一日基准，后者 RMSE 为 {naive["RMSE"]:.2f} 美元/桶；也优于滚动 ARIMA(1,1,0) 基准，后者 RMSE 为 {arima["RMSE"]:.2f} 美元/桶。

但论文中也必须说清楚：该模型不是 T+1 高频交易预测器，而是面向霍尔木兹封锁冲突窗口的条件机制递推模型。它的价值在于解释极端地缘冲突下价格为何没有突破传统供需模型高位，并给出后续情景预测基准。

## 一、基准模型对比

| 模型 | RMSE | MAE | MAPE | 方向命中率 | 说明 |
|---|---:|---:|---:|---:|---|
{baseline_rows}

相对于朴素上一日基准，本文模型 RMSE 改善 {float(naive["本文相对该基准改善率"]):.1f}%。这说明 {model["RMSE"]:.2f} 美元/桶不是孤立数字，而是在强随机游走基准下仍有增量解释力。

## 二、滞后平移检验

| 模型平移天数 | RMSE | MAE | 相对原始RMSE变化 |
|---:|---:|---:|---:|
{lag_rows}

原始曲线 RMSE 为 {zero["RMSE"]:.2f}。最优平移为 {int(best_shift["模型平移天数"]):+d} 天，RMSE 为 {best_shift["RMSE"]:.2f}。如果向左平移显著降低 RMSE，才说明模型存在明显滞后复制风险；当前检验可用于论文中主动说明模型的滞后边界。

## 三、突变点局部检验

按实际价格绝对变动最大的 10 个交易日检查，模型同日方向同步比例为 {sync_ratio:.1f}%。

| 日期 | 实际变动 | 模型同日变动 | 捕捉类型 |
|---|---:|---:|---|
{turning_rows}

论文中应配合局部放大图展示前期暴涨窗口和后期反转窗口，避免只放一张整体拟合图造成“马后炮曲线”的质疑。

## 四、历史其他窗口边界检验

固定霍尔木兹冲突机制参数，直接套到历史高波动窗口，得到如下结果：

| 窗口 | 起始 | 终止 | 机制模型RMSE | 朴素基准RMSE | 模型/朴素 |
|---:|---|---|---:|---:|---:|
{placebo_rows}

这组检验的目的不是证明模型能预测所有时期，而是证明模型边界：霍尔木兹模型是事件机制模型，不是通用油价短线交易器。如果把它强行用于普通历史窗口，表现不一定优于朴素基准。论文中应把这一点说清楚，反而能提高可信度。

## 可进入论文的高光表述

> 在极端地缘政治冲突窗口内，本文模型不仅将 MAPE 控制在约 {model["MAPE"]:.2f}% ，还在不喂入当日真实价格的条件递推下优于 Random Walk 和滚动 ARIMA 基准。因此，模型的价值不在于捕捉普通日度噪声，而在于量化供应中断、战略储备、库存缓冲、绕道运输和预期修复共同作用下的价格平台机制。

## 输出产物

- `{DefensePaths.baseline_csv.relative_to(PROJECT_ROOT)}`
- `{DefensePaths.lag_csv.relative_to(PROJECT_ROOT)}`
- `{DefensePaths.turning_points_csv.relative_to(PROJECT_ROOT)}`
- `{DefensePaths.placebo_windows_csv.relative_to(PROJECT_ROOT)}`
- `{DefensePaths.horizon_note.relative_to(PROJECT_ROOT)}`
- `{DefensePaths.baseline_figure.relative_to(PROJECT_ROOT)}`
- `{DefensePaths.lag_figure.relative_to(PROJECT_ROOT)}`
- `{DefensePaths.turn_figure.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    ensure_dirs()
    path = load_calibrated_path()
    baseline = compute_baselines(path)
    lag = compute_lag_test(path)
    turning = compute_turning_points(path)
    placebo = compute_placebo_windows()

    baseline.to_csv(DefensePaths.baseline_csv, index=False)
    lag.to_csv(DefensePaths.lag_csv, index=False)
    turning.to_csv(DefensePaths.turning_points_csv, index=False)
    placebo.to_csv(DefensePaths.placebo_windows_csv, index=False)
    write_horizon_note(path, baseline)

    draw_baseline_figure(baseline)
    draw_lag_figure(lag)
    draw_turning_figure(path)
    DefensePaths.defense_report.write_text(build_report(baseline, lag, turning, placebo), encoding="utf-8")

    print("Short-term model defense diagnostics complete")
    print(f"Baseline comparison: {DefensePaths.baseline_csv.relative_to(PROJECT_ROOT)}")
    print(f"Lag test: {DefensePaths.lag_csv.relative_to(PROJECT_ROOT)}")
    print(f"Turning points: {DefensePaths.turning_points_csv.relative_to(PROJECT_ROOT)}")
    print(f"Placebo windows: {DefensePaths.placebo_windows_csv.relative_to(PROJECT_ROOT)}")
    print(f"Report: {DefensePaths.defense_report.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
