"""Short-term model error attribution audit.

This audit does not recalibrate the short-term model. It decomposes the saved
best path by date, event window, and mechanism channel so the residual risk can
be explained without changing the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common.metrics import mae, rmse
from src.common.paths import PROJECT_ROOT, ensure_parents
from src.common.plotting import PAPER_COLORS, SCENARIO_COLORS, configure_plot_style


CALIBRATED_PATH = PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv"
ML_PATH = PROJECT_ROOT / "output" / "calibration" / "短期机器学习残差修正路径.csv"
DAILY_ATTRIBUTION_CSV = PROJECT_ROOT / "output" / "calibration" / "短期模型逐日误差归因.csv"
TOP_ERROR_CSV = PROJECT_ROOT / "output" / "calibration" / "短期模型最大误差日.csv"
WINDOW_ATTRIBUTION_CSV = PROJECT_ROOT / "output" / "calibration" / "短期模型事件段误差归因.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "短期模型误差归因.png"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "短期模型误差归因复核报告.md"


EVENT_WINDOWS = [
    ("初始冲击吸收", "2026-03-02", "2026-03-14"),
    ("高位平台形成", "2026-03-17", "2026-03-31"),
    ("中期再定价回落", "2026-04-01", "2026-04-17"),
    ("后期二次抬升", "2026-04-20", "2026-05-05"),
]

MECHANISM_COLUMNS = {
    "物理缺口压力": "shortage_pressure",
    "封锁风险溢价": "blockade_risk_premium",
    "不确定性溢价": "uncertainty_premium",
    "恐慌溢价": "panic_premium",
    "缓冲确认折价": "buffer_confirmation_discount",
    "预期修复折价": "expectation_relief_discount",
}


def ensure_dirs() -> None:
    ensure_parents([DAILY_ATTRIBUTION_CSV, TOP_ERROR_CSV, WINDOW_ATTRIBUTION_CSV, FIGURE_PATH, REPORT_PATH])


def load_path() -> pd.DataFrame:
    path = pd.read_csv(CALIBRATED_PATH, parse_dates=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    path["模型误差"] = path["simulated_price"] - path["actual_price"]
    path["绝对误差"] = path["模型误差"].abs()
    path["相对误差"] = path["绝对误差"] / path["actual_price"] * 100
    path["朴素预测"] = path["actual_price"].shift(1)
    path.loc[0, "朴素预测"] = path.loc[0, "actual_price"]
    path["朴素误差"] = path["朴素预测"] - path["actual_price"]
    path["事件段"] = "未分段"
    for name, start, end in EVENT_WINDOWS:
        mask = path["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))
        path.loc[mask, "事件段"] = name

    if ML_PATH.exists():
        ml = pd.read_csv(ML_PATH, parse_dates=["trade_date"])
        keep = [
            "trade_date",
            "mechanism_plus_ridge_return",
            "mechanism_plus_phase_ridge",
            "phase_gated_ridge_correction",
            "ridge_assist_phase",
            "online_corrected_price",
            "regime_corrected_price",
            "ridge_price",
        ]
        path = path.merge(ml[[column for column in keep if column in ml.columns]], on="trade_date", how="left")
        if "mechanism_plus_ridge_return" in path:
            path["机制加Ridge误差"] = path["mechanism_plus_ridge_return"] - path["actual_price"]
        if "mechanism_plus_phase_ridge" in path:
            path["阶段Ridge误差"] = path["mechanism_plus_phase_ridge"] - path["actual_price"]
        if "online_corrected_price" in path:
            path["在线校正误差"] = path["online_corrected_price"] - path["actual_price"]
        if "regime_corrected_price" in path:
            path["分段校正误差"] = path["regime_corrected_price"] - path["actual_price"]

    positive = path[["shortage_pressure", "blockade_risk_premium", "uncertainty_premium", "panic_premium"]].sum(axis=1)
    discounts = path[["buffer_confirmation_discount", "expectation_relief_discount"]].sum(axis=1)
    path["风险溢价合计"] = positive
    path["折价合计"] = discounts
    path["净机制压力"] = positive - discounts
    path["缓冲供给合计"] = path[["spr_release", "route_supply", "inventory_buffer"]].sum(axis=1)
    path["剩余缺口占需求"] = path["supply_gap"] / path["effective_demand"]
    path["误差方向"] = np.where(path["模型误差"] > 0, "高估", np.where(path["模型误差"] < 0, "低估", "基本准确"))
    return path


def build_daily_attribution(path: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trade_date",
        "day_index",
        "事件段",
        "actual_price",
        "simulated_price",
        "模型误差",
        "绝对误差",
        "相对误差",
        "误差方向",
        "朴素预测",
        "朴素误差",
        "supply_gap",
        "剩余缺口占需求",
        "缓冲供给合计",
        "风险溢价合计",
        "折价合计",
        "净机制压力",
        *MECHANISM_COLUMNS.values(),
    ]
    optional = [
        "mechanism_plus_ridge_return",
        "机制加Ridge误差",
        "online_corrected_price",
        "在线校正误差",
        "regime_corrected_price",
        "分段校正误差",
        "mechanism_plus_phase_ridge",
        "阶段Ridge误差",
        "phase_gated_ridge_correction",
        "ridge_assist_phase",
    ]
    columns.extend([column for column in optional if column in path.columns])
    return path[columns].copy()


def summarize_windows(path: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for name, start, end in EVENT_WINDOWS:
        sub = path[path["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
        error = sub["模型误差"]
        naive_error = sub["朴素误差"]
        row: dict[str, float | str | int] = {
            "事件段": name,
            "起始日期": start,
            "结束日期": end,
            "样本数": len(sub),
            "RMSE": rmse(error),
            "MAE": mae(error),
            "平均误差": float(error.mean()),
            "最大绝对误差": float(error.abs().max()),
            "低估天数": int((error < 0).sum()),
            "高估天数": int((error > 0).sum()),
            "朴素RMSE": rmse(naive_error),
            "相对朴素改善率": (rmse(naive_error) - rmse(error)) / rmse(naive_error) * 100,
            "平均剩余缺口": float(sub["supply_gap"].mean()),
            "平均缓冲供给": float(sub["缓冲供给合计"].mean()),
            "平均风险溢价": float(sub["风险溢价合计"].mean()),
            "平均折价": float(sub["折价合计"].mean()),
            "平均净机制压力": float(sub["净机制压力"].mean()),
        }
        if "机制加Ridge误差" in sub:
            row["机制加Ridge_RMSE"] = rmse(sub["机制加Ridge误差"])
            row["机制加Ridge相对主模型改善率"] = (rmse(error) - rmse(sub["机制加Ridge误差"])) / rmse(error) * 100
        if "阶段Ridge误差" in sub:
            row["阶段Ridge_RMSE"] = rmse(sub["阶段Ridge误差"])
            row["阶段Ridge相对主模型改善率"] = (rmse(error) - rmse(sub["阶段Ridge误差"])) / rmse(error) * 100
        if "在线校正误差" in sub:
            row["在线校正_RMSE"] = rmse(sub["在线校正误差"])
            row["在线校正相对主模型改善率"] = (rmse(error) - rmse(sub["在线校正误差"])) / rmse(error) * 100
        if "分段校正误差" in sub:
            row["分段校正_RMSE"] = rmse(sub["分段校正误差"])
            row["分段校正相对主模型改善率"] = (rmse(error) - rmse(sub["分段校正误差"])) / rmse(error) * 100
        rows.append(row)
    return pd.DataFrame(rows)


def top_errors(path: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    columns = [
        "trade_date",
        "事件段",
        "actual_price",
        "simulated_price",
        "模型误差",
        "绝对误差",
        "误差方向",
        "supply_gap",
        "缓冲供给合计",
        "风险溢价合计",
        "折价合计",
        "净机制压力",
    ]
    return path.sort_values("绝对误差", ascending=False).head(n)[columns].copy()


def draw_figure(path: pd.DataFrame, windows: pd.DataFrame) -> None:
    configure_plot_style(savefig_dpi=260, figure_dpi=150, title_size=13)
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.2))
    ax = axes[0, 0]
    colors = path["模型误差"].map(lambda value: SCENARIO_COLORS["risk"] if value > 0 else SCENARIO_COLORS["optimistic"])
    ax.bar(path["trade_date"], path["模型误差"], color=colors, width=0.8)
    ax.axhline(0, color=PAPER_COLORS["ink"], lw=1.0)
    ax.set_title("逐日残差：正值为模型高估，负值为模型低估")
    ax.set_ylabel("误差（美元/桶）")

    ax = axes[0, 1]
    ax.barh(windows["事件段"], windows["RMSE"], color=SCENARIO_COLORS["fit"], alpha=0.9, label="主模型")
    if "阶段Ridge_RMSE" in windows:
        ax.scatter(windows["阶段Ridge_RMSE"], windows["事件段"], color=SCENARIO_COLORS["buffer"], s=52, marker="^", label="最终增强")
    ax.set_title("事件段 RMSE 对比")
    ax.set_xlabel("RMSE（美元/桶）")
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    mechanism = path[
        [
            "trade_date",
            "shortage_pressure",
            "blockade_risk_premium",
            "uncertainty_premium",
            "panic_premium",
        ]
    ].copy()
    mechanism = mechanism.rename(columns={value: key for key, value in MECHANISM_COLUMNS.items()})
    ax.stackplot(
        mechanism["trade_date"],
        mechanism["物理缺口压力"],
        mechanism["封锁风险溢价"],
        mechanism["不确定性溢价"],
        mechanism["恐慌溢价"],
        labels=["物理缺口", "封锁风险", "不确定性", "恐慌溢价"],
        colors=[SCENARIO_COLORS["buffer"], SCENARIO_COLORS["fit"], SCENARIO_COLORS["risk"], SCENARIO_COLORS["highlight"]],
        alpha=0.72,
    )
    ax.set_title("上行机制压力分解")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    ax = axes[1, 1]
    ax.plot(path["trade_date"], path["风险溢价合计"], color=SCENARIO_COLORS["risk"], lw=2.0, label="风险溢价合计")
    ax.plot(path["trade_date"], path["折价合计"], color=SCENARIO_COLORS["optimistic"], lw=2.0, label="缓冲/预期折价")
    ax.plot(path["trade_date"], path["净机制压力"], color=PAPER_COLORS["ink"], lw=1.8, label="净机制压力")
    ax.set_title("风险溢价与缓冲折价的拉扯")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper right", fontsize=8)

    for axis in axes.ravel():
        axis.grid(True, alpha=0.45)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURE_PATH)
    plt.close(fig)


def build_report(path: pd.DataFrame, daily: pd.DataFrame, top: pd.DataFrame, windows: pd.DataFrame) -> str:
    model_rmse = rmse(path["模型误差"])
    naive_rmse = rmse(path["朴素误差"])
    model_mae = mae(path["模型误差"])
    phase_ridge_rmse = rmse(path["阶段Ridge误差"]) if "阶段Ridge误差" in path else np.nan
    phase_ridge_mae = mae(path["阶段Ridge误差"]) if "阶段Ridge误差" in path else np.nan
    worst = top.iloc[0]
    worst_window = windows.sort_values("RMSE", ascending=False).iloc[0]
    best_window = windows.sort_values("RMSE").iloc[0]
    top_rows = "\n".join(
        "| {trade_date:%Y-%m-%d} | {事件段} | {actual_price:.2f} | {simulated_price:.2f} | {模型误差:.2f} | {误差方向} | {净机制压力:.2f} |".format(
            **row
        )
        for row in top.to_dict("records")
    )
    window_rows = "\n".join(
        f"| {row['事件段']} | {row['RMSE']:.3f} | {row['MAE']:.3f} | {row['相对朴素改善率']:.1f}% | {row.get('阶段Ridge_RMSE', np.nan):.3f} | {row.get('阶段Ridge相对主模型改善率', np.nan):.1f}% |"
        for row in windows.to_dict("records")
    )
    conclusion = (
        "短期机制主模型可以继续承担经济解释任务，但质量表述应区分机制解释路径和增强预测路径。"
        f"针对高位平台与中期再定价两个薄弱段启用阶段 Ridge 后，RMSE 降至 {phase_ridge_rmse:.3f}，"
        f"MAE 降至 {phase_ridge_mae:.3f}。这说明低自由度数据驱动层可以改善平台期日度震荡，"
        "但不应替代供需、风险溢价和预期修复机制。"
    )
    return f"""# 短期模型误差归因复核报告

## 核心结论

本轮复核不重新调参，只基于当前最终短期路径进行逐日误差、事件段误差和机制项归因。当前机制主模型全窗口 RMSE 为 {model_rmse:.3f} 美元/桶，MAE 为 {model_mae:.3f} 美元/桶；最终短期增强路径 RMSE 为 {phase_ridge_rmse:.3f} 美元/桶，MAE 为 {phase_ridge_mae:.3f} 美元/桶；朴素上一日基准 RMSE 为 {naive_rmse:.3f} 美元/桶。

最大单日误差出现在 {worst['trade_date']:%Y-%m-%d}，属于“{worst['事件段']}”，模型{worst['误差方向']} {abs(float(worst['模型误差'])):.2f} 美元/桶。事件段层面，表现最好的是“{best_window['事件段']}”（RMSE={best_window['RMSE']:.2f}），最难的是“{worst_window['事件段']}”（RMSE={worst_window['RMSE']:.2f}）。

{conclusion}

## 最大误差日

| 日期 | 事件段 | 实际价 | 模型价 | 模型误差 | 方向 | 净机制压力 |
|---|---|---:|---:|---:|---|---:|
{top_rows}

## 事件段复核

    | 事件段 | 机制主模型RMSE | MAE | 相对朴素改善 | 最终增强RMSE | 相对主模型改善 |
    |---|---:|---:|---:|---:|---:|
{window_rows}

## 解释口径

- “高估”表示模型给出的机制压力偏强，通常需要检查风险溢价是否下降得不够快，或缓冲折价是否释放不足。
- “低估”表示模型没有完全捕捉真实市场的二次风险重定价，通常出现在尾段重新上行或平台震荡阶段。
- 阶段 Ridge 只作为低自由度辅助修正，不替代主模型；若某些事件段改善明显，说明这些片段仍存在短期收益惯性或残差相关结构。

## 输出文件

- `{DAILY_ATTRIBUTION_CSV.relative_to(PROJECT_ROOT)}`
- `{TOP_ERROR_CSV.relative_to(PROJECT_ROOT)}`
- `{WINDOW_ATTRIBUTION_CSV.relative_to(PROJECT_ROOT)}`
- `{FIGURE_PATH.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    ensure_dirs()
    path = load_path()
    daily = build_daily_attribution(path)
    windows = summarize_windows(path)
    top = top_errors(path)
    daily.to_csv(DAILY_ATTRIBUTION_CSV, index=False)
    windows.to_csv(WINDOW_ATTRIBUTION_CSV, index=False)
    top.to_csv(TOP_ERROR_CSV, index=False)
    draw_figure(path, windows)
    REPORT_PATH.write_text(build_report(path, daily, top, windows), encoding="utf-8")

    print("Short-term error attribution audit complete")
    print(f"Daily attribution: {DAILY_ATTRIBUTION_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Top errors: {TOP_ERROR_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Window attribution: {WINDOW_ATTRIBUTION_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
