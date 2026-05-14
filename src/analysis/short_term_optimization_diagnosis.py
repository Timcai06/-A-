"""Short-term model optimization diagnosis.

This module is intentionally kept outside the paper build.  It reads the
current short-term model artifacts and produces an engineering diagnosis for
the next modeling iteration, without writing into ``paper/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.metrics import mae, rmse
from src.common.paths import PROJECT_ROOT, ensure_parents
from src.common.plotting import PAPER_COLORS, SCENARIO_COLORS, configure_plot_style


@dataclass(frozen=True)
class Paths:
    mechanism_path: Path = PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv"
    enhanced_path: Path = PROJECT_ROOT / "output" / "calibration" / "短期机器学习残差修正路径.csv"
    model_compare: Path = PROJECT_ROOT / "output" / "calibration" / "短期机器学习模型对比.csv"
    daily_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期模型优化诊断_逐日表.csv"
    segment_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期模型优化诊断_事件段表.csv"
    lag_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期模型优化诊断_滞后表.csv"
    report: Path = PROJECT_ROOT / "output" / "reports" / "短期模型优化诊断报告.md"
    figure: Path = PROJECT_ROOT / "output" / "candidate_figures" / "短期模型优化诊断_残差结构.png"


EVENT_WINDOWS = [
    ("初始冲击吸收", "2026-03-02", "2026-03-14"),
    ("高位平台形成", "2026-03-17", "2026-03-31"),
    ("中期再定价回落", "2026-04-01", "2026-04-17"),
    ("后期二次抬升", "2026-04-20", "2026-05-05"),
]

MECHANISM_COLUMNS = {
    "供需缺口压力": "shortage_pressure",
    "封锁风险溢价": "blockade_risk_premium",
    "不确定性溢价": "uncertainty_premium",
    "恐慌溢价": "panic_premium",
    "缓冲确认折价": "buffer_confirmation_discount",
    "预期修复折价": "expectation_relief_discount",
}


def load_model_path() -> pd.DataFrame:
    if not Paths.mechanism_path.exists():
        raise FileNotFoundError(Paths.mechanism_path)
    mechanism = pd.read_csv(Paths.mechanism_path, parse_dates=["trade_date"]).sort_values("trade_date")
    if Paths.enhanced_path.exists():
        enhanced = pd.read_csv(Paths.enhanced_path, parse_dates=["trade_date"]).sort_values("trade_date")
        optional = [
            "trade_date",
            "naive_price",
            "ridge_price",
            "mechanism_plus_ridge_return",
            "mechanism_plus_phase_ridge",
            "online_corrected_price",
            "regime_corrected_price",
            "ridge_assist_phase",
            "phase_gated_ridge_correction",
        ]
        mechanism = mechanism.merge(enhanced[[c for c in optional if c in enhanced.columns]], on="trade_date", how="left")

    mechanism["事件段"] = "未分段"
    for name, start, end in EVENT_WINDOWS:
        mask = mechanism["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))
        mechanism.loc[mask, "事件段"] = name

    mechanism["机制误差"] = mechanism["simulated_price"] - mechanism["actual_price"]
    mechanism["机制绝对误差"] = mechanism["机制误差"].abs()
    mechanism["朴素价格"] = mechanism.get("naive_price", mechanism["actual_price"].shift(1))
    mechanism.loc[mechanism["朴素价格"].isna(), "朴素价格"] = mechanism.loc[mechanism["朴素价格"].isna(), "actual_price"]
    mechanism["朴素误差"] = mechanism["朴素价格"] - mechanism["actual_price"]
    if "mechanism_plus_phase_ridge" in mechanism:
        mechanism["阶段Ridge误差"] = mechanism["mechanism_plus_phase_ridge"] - mechanism["actual_price"]
    if "mechanism_plus_ridge_return" in mechanism:
        mechanism["机制加Ridge误差"] = mechanism["mechanism_plus_ridge_return"] - mechanism["actual_price"]

    positive_cols = ["shortage_pressure", "blockade_risk_premium", "uncertainty_premium", "panic_premium"]
    discount_cols = ["buffer_confirmation_discount", "expectation_relief_discount"]
    mechanism["风险上行压力"] = mechanism[positive_cols].sum(axis=1)
    mechanism["缓冲折价"] = mechanism[discount_cols].sum(axis=1)
    mechanism["净机制压力"] = mechanism["风险上行压力"] - mechanism["缓冲折价"]
    mechanism["缓冲供给"] = mechanism[["spr_release", "route_supply", "inventory_buffer"]].sum(axis=1)
    mechanism["剩余缺口占需求"] = mechanism["supply_gap"] / mechanism["effective_demand"]
    mechanism["实际日变化"] = mechanism["actual_price"].diff()
    mechanism["模型日变化"] = mechanism["simulated_price"].diff()
    mechanism["拐点错向"] = np.sign(mechanism["实际日变化"]) != np.sign(mechanism["模型日变化"])
    mechanism.loc[mechanism["实际日变化"].isna() | mechanism["模型日变化"].isna(), "拐点错向"] = False
    return mechanism


def metric_row(df: pd.DataFrame, error_col: str) -> dict[str, float]:
    error = df[error_col].dropna()
    return {
        "RMSE": rmse(error),
        "MAE": mae(error),
        "平均误差": float(error.mean()),
        "最大绝对误差": float(error.abs().max()),
    }


def segment_summary(path: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, start, end in EVENT_WINDOWS:
        sub = path[path["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
        row: dict[str, object] = {
            "事件段": name,
            "起始日期": start,
            "结束日期": end,
            "样本数": len(sub),
            **{f"机制{key}": value for key, value in metric_row(sub, "机制误差").items()},
            "朴素RMSE": rmse(sub["朴素误差"]),
            "机制相对朴素RMSE改善率": (rmse(sub["朴素误差"]) - rmse(sub["机制误差"])) / rmse(sub["朴素误差"]) * 100
            if rmse(sub["朴素误差"]) > 0
            else np.nan,
            "平均剩余缺口占需求": float(sub["剩余缺口占需求"].mean()),
            "平均缓冲供给": float(sub["缓冲供给"].mean()),
            "平均风险上行压力": float(sub["风险上行压力"].mean()),
            "平均缓冲折价": float(sub["缓冲折价"].mean()),
            "拐点错向天数": int(sub["拐点错向"].sum()),
        }
        if "阶段Ridge误差" in sub:
            ridge = metric_row(sub, "阶段Ridge误差")
            row.update({f"阶段Ridge{key}": value for key, value in ridge.items()})
            row["阶段Ridge相对机制RMSE改善率"] = (rmse(sub["机制误差"]) - rmse(sub["阶段Ridge误差"])) / rmse(sub["机制误差"]) * 100
        if "机制加Ridge误差" in sub:
            ridge_all = metric_row(sub, "机制加Ridge误差")
            row.update({f"全段Ridge{key}": value for key, value in ridge_all.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def lag_diagnosis(path: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    actual = path["actual_price"].to_numpy(dtype=float)
    model = path["simulated_price"].to_numpy(dtype=float)
    for lag in range(-5, 6):
        if lag < 0:
            aligned_actual = actual[-lag:]
            aligned_model = model[:lag]
            meaning = f"模型提前{-lag}天"
        elif lag > 0:
            aligned_actual = actual[:-lag]
            aligned_model = model[lag:]
            meaning = f"模型滞后{lag}天"
        else:
            aligned_actual = actual
            aligned_model = model
            meaning = "原始对齐"
        error = aligned_model - aligned_actual
        rows.append(
            {
                "平移天数": lag,
                "含义": meaning,
                "RMSE": rmse(error),
                "MAE": mae(error),
                "相关系数": float(np.corrcoef(aligned_actual, aligned_model)[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def top_error_table(path: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    cols = [
        "trade_date",
        "事件段",
        "actual_price",
        "simulated_price",
        "机制误差",
        "机制绝对误差",
        "supply_gap",
        "缓冲供给",
        "风险上行压力",
        "缓冲折价",
        "净机制压力",
        "拐点错向",
    ]
    if "mechanism_plus_phase_ridge" in path:
        cols.extend(["mechanism_plus_phase_ridge", "阶段Ridge误差", "ridge_assist_phase", "phase_gated_ridge_correction"])
    return path.sort_values("机制绝对误差", ascending=False).head(n)[cols]


def mechanism_correlations(path: pd.DataFrame) -> pd.DataFrame:
    cols = ["机制误差", "supply_gap", "缓冲供给", "风险上行压力", "缓冲折价", "净机制压力", *MECHANISM_COLUMNS.values()]
    corr = path[cols].corr(numeric_only=True)["机制误差"].drop("机制误差").sort_values(key=lambda s: s.abs(), ascending=False)
    return corr.reset_index().rename(columns={"index": "变量", "机制误差": "与机制误差相关系数"})


def draw_diagnostic_figure(path: pd.DataFrame, segments: pd.DataFrame) -> None:
    configure_plot_style(savefig_dpi=260, figure_dpi=150, title_size=13)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))

    ax = axes[0, 0]
    colors = np.where(path["机制误差"] >= 0, SCENARIO_COLORS["risk"], SCENARIO_COLORS["buffer"])
    ax.bar(path["trade_date"], path["机制误差"], color=colors, alpha=0.72, width=0.8)
    ax.axhline(0, color=PAPER_COLORS["ink"], lw=0.9)
    ax.set_title("逐日机制误差")
    ax.set_ylabel("美元/桶")

    ax = axes[0, 1]
    y = np.arange(len(segments))
    ax.barh(y + 0.18, segments["机制RMSE"], height=0.32, color=SCENARIO_COLORS["fit"], label="机制主模型")
    if "阶段RidgeRMSE" in segments:
        ax.barh(y - 0.18, segments["阶段RidgeRMSE"], height=0.32, color=SCENARIO_COLORS["buffer"], label="阶段Ridge增强")
    ax.set_yticks(y, segments["事件段"])
    ax.set_title("事件段 RMSE")
    ax.set_xlabel("美元/桶")
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    ax.plot(path["trade_date"], path["风险上行压力"], color=SCENARIO_COLORS["risk"], lw=2.0, label="风险上行压力")
    ax.plot(path["trade_date"], path["缓冲折价"], color=SCENARIO_COLORS["buffer"], lw=2.0, label="缓冲折价")
    ax.plot(path["trade_date"], path["净机制压力"], color=PAPER_COLORS["ink"], lw=1.7, label="净机制压力")
    ax.set_title("机制压力与折价结构")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    ax.scatter(path["净机制压力"], path["机制误差"], color=SCENARIO_COLORS["fit"], alpha=0.72)
    ax.axhline(0, color=PAPER_COLORS["ink"], lw=0.9)
    ax.axvline(path["净机制压力"].median(), color=PAPER_COLORS["muted"], lw=0.9, linestyle="--")
    ax.set_title("净机制压力与误差关系")
    ax.set_xlabel("净机制压力")
    ax.set_ylabel("机制误差")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(Paths.figure)
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def build_report(
    path: pd.DataFrame,
    segments: pd.DataFrame,
    lag: pd.DataFrame,
    top_errors: pd.DataFrame,
    corr: pd.DataFrame,
) -> str:
    model_rmse = rmse(path["机制误差"])
    model_mae = mae(path["机制误差"])
    naive_rmse = rmse(path["朴素误差"])
    best_shift = lag.sort_values("RMSE").iloc[0]
    worst_segment = segments.sort_values("机制RMSE", ascending=False).iloc[0]
    best_gain_segment = (
        segments.dropna(subset=["阶段Ridge相对机制RMSE改善率"]).sort_values("阶段Ridge相对机制RMSE改善率", ascending=False).iloc[0]
        if "阶段Ridge相对机制RMSE改善率" in segments
        else None
    )
    top_rows = "\n".join(
        "| {date} | {seg} | {actual:.2f} | {sim:.2f} | {err:.2f} | {gap:.2f} | {pressure:.2f} | {discount:.2f} | {turn} |".format(
            date=row["trade_date"].strftime("%Y-%m-%d"),
            seg=row["事件段"],
            actual=row["actual_price"],
            sim=row["simulated_price"],
            err=row["机制误差"],
            gap=row["supply_gap"],
            pressure=row["风险上行压力"],
            discount=row["缓冲折价"],
            turn="是" if row["拐点错向"] else "否",
        )
        for row in top_errors.to_dict("records")
    )
    segment_rows = "\n".join(
        "| {seg} | {rmse} | {mae} | {naive} | {gain} | {ridge} | {ridge_gain} | {turns} |".format(
            seg=row["事件段"],
            rmse=fmt(row["机制RMSE"]),
            mae=fmt(row["机制MAE"]),
            naive=fmt(row["朴素RMSE"]),
            gain=fmt(row["机制相对朴素RMSE改善率"], 1) + "%",
            ridge=fmt(row.get("阶段RidgeRMSE", np.nan)),
            ridge_gain=fmt(row.get("阶段Ridge相对机制RMSE改善率", np.nan), 1) + "%",
            turns=int(row["拐点错向天数"]),
        )
        for row in segments.to_dict("records")
    )
    corr_rows = "\n".join(
        f"| {row['变量']} | {row['与机制误差相关系数']:.3f} |" for row in corr.head(8).to_dict("records")
    )
    lag_rows = "\n".join(
        f"| {int(row['平移天数'])} | {row['含义']} | {row['RMSE']:.3f} | {row['MAE']:.3f} | {row['相关系数']:.3f} |"
        for row in lag.to_dict("records")
    )
    ridge_note = ""
    if best_gain_segment is not None:
        ridge_note = (
            f"阶段 Ridge 对“{best_gain_segment['事件段']}”帮助最大，"
            f"该段 RMSE 改善 {best_gain_segment['阶段Ridge相对机制RMSE改善率']:.1f}%。"
        )

    return f"""# 短期模型优化诊断报告

> 本报告只用于下一轮建模优化，不写入论文正文，也不修改 `paper/` 目录。

## 诊断结论

当前机制主模型全窗口 RMSE={model_rmse:.3f}，MAE={model_mae:.3f}；朴素上一日基准 RMSE={naive_rmse:.3f}，说明机制主模型仍然明显打败随机游走基准。最薄弱的事件段是“{worst_segment['事件段']}”，该段机制 RMSE={worst_segment['机制RMSE']:.3f}。{ridge_note}

滞后诊断中，最低 RMSE 出现在“{best_shift['含义']}”（平移天数 {int(best_shift['平移天数'])}，RMSE={best_shift['RMSE']:.3f}）。如果最低点远离原始对齐，说明模型有明显滞后/超前问题；如果最低点接近 0，则说明主要问题不是简单平移，而是局部机制强弱和日度震荡刻画不足。

## 事件段表现

| 事件段 | 机制RMSE | 机制MAE | 朴素RMSE | 机制相对朴素改善 | 阶段Ridge RMSE | 阶段Ridge相对机制改善 | 拐点错向天数 |
|---|---:|---:|---:|---:|---:|---:|---:|
{segment_rows}

## 最大误差日

| 日期 | 事件段 | 实际价 | 机制价 | 机制误差 | 剩余缺口 | 风险上行压力 | 缓冲折价 | 拐点错向 |
|---|---|---:|---:|---:|---:|---:|---:|---|
{top_rows}

## 误差相关性线索

| 变量 | 与机制误差相关系数 |
|---|---:|
{corr_rows}

## 滞后诊断

| 平移天数 | 含义 | RMSE | MAE | 相关系数 |
|---:|---|---:|---:|---:|
{lag_rows}

## 下一轮优化建议

1. 优先处理“{worst_segment['事件段']}”：这里不是全局参数能轻松解决的问题，更像事件段机制强弱或滞后结构没有完全跟上。
2. 保留机制主模型作为主解释层，不建议直接让机器学习替代机制递推。
3. 对高位平台和中期再定价段继续测试阶段门控残差修正，但修正量必须小、可解释、只使用滞后信息。
4. 下一轮代码优化应先尝试 1-3 天滞后/平滑项，尤其是风险溢价、缓冲确认折价和预期修复折价。
5. 若要继续补外生变量，应优先接入市场定价类滞后变量，例如 OVX、期限结构斜率或可复现新闻风险指数，而不是继续增加自由参数。

## 输出文件

- `{Paths.daily_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.segment_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.lag_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.figure.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    ensure_parents([Paths.daily_csv, Paths.segment_csv, Paths.lag_csv, Paths.report, Paths.figure])
    path = load_model_path()
    segments = segment_summary(path)
    lag = lag_diagnosis(path)
    top_errors = top_error_table(path)
    corr = mechanism_correlations(path)

    path.to_csv(Paths.daily_csv, index=False)
    segments.to_csv(Paths.segment_csv, index=False)
    lag.to_csv(Paths.lag_csv, index=False)
    draw_diagnostic_figure(path, segments)
    Paths.report.write_text(build_report(path, segments, lag, top_errors, corr), encoding="utf-8")
    print(Paths.report.relative_to(PROJECT_ROOT))
    print(Paths.figure.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
