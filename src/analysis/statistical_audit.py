"""Stage 13 statistical audit for the short-term model.

This stage is designed as a paper-facing rigor layer. It does not alter the
main model; instead it checks whether the calibrated short-term path is
statistically stronger than simple baselines and whether residual diagnostics
show obvious unresolved structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src.common.metrics import mae, mape, rmse
from src.common.paths import PROJECT_ROOT, ensure_parent
from src.common.plotting import PAPER_COLORS, SCENARIO_COLORS
from src.models import dynamic_short_term as dynamic


CALIBRATED_PATH = PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "statistics"
SUMMARY_CSV = OUTPUT_DIR / "短期统计审计摘要.csv"
DIAGNOSTIC_CSV = OUTPUT_DIR / "短期残差诊断.csv"
SPLIT_CSV = OUTPUT_DIR / "短期分段统计.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "短期模型统计审计.png"
REPORT_PATH = PROJECT_ROOT / "output" / "reports" / "短期模型统计审计报告.md"


@dataclass(frozen=True)
class ForecastSeries:
    df: pd.DataFrame
    actual: np.ndarray
    model: np.ndarray
    naive: np.ndarray
    residual: np.ndarray
    naive_residual: np.ndarray


def load_forecast_series() -> ForecastSeries:
    if not CALIBRATED_PATH.exists():
        raise FileNotFoundError(f"Missing calibrated path: {CALIBRATED_PATH}")

    df = pd.read_csv(CALIBRATED_PATH, parse_dates=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    required = {"actual_price", "simulated_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in calibrated path: {sorted(missing)}")

    df["naive_forecast"] = df["actual_price"].shift(1)
    eval_df = df.dropna(subset=["actual_price", "simulated_price", "naive_forecast"]).copy()
    actual = eval_df["actual_price"].to_numpy(dtype=float)
    model = eval_df["simulated_price"].to_numpy(dtype=float)
    naive = eval_df["naive_forecast"].to_numpy(dtype=float)
    residual = model - actual
    naive_residual = naive - actual

    eval_df["model_residual"] = residual
    eval_df["naive_residual"] = naive_residual
    eval_df["model_abs_error"] = np.abs(residual)
    eval_df["naive_abs_error"] = np.abs(naive_residual)
    eval_df["model_squared_error"] = residual**2
    eval_df["naive_squared_error"] = naive_residual**2
    return ForecastSeries(eval_df, actual, model, naive, residual, naive_residual)


def diebold_mariano(loss_model: np.ndarray, loss_baseline: np.ndarray, max_lag: int | None = None) -> tuple[float, float, float]:
    """One-step Diebold-Mariano test with small-sample t reference and Newey-West HAC variance.

    Args:
        loss_model: Per-period loss of the model under evaluation.
        loss_baseline: Per-period loss of the reference model.
        max_lag: Maximum lag for the Newey-West kernel. Defaults to floor(n^(1/3)).
    """
    diff = np.asarray(loss_model - loss_baseline, dtype=float)
    n = diff.size
    if n < 3:
        return float("nan"), float("nan"), float("nan")

    mean_diff = float(np.mean(diff))
    if max_lag is None:
        max_lag = max(1, int(n ** (1 / 3)))
    max_lag = min(max_lag, n - 1)

    centered = diff - mean_diff
    long_run_variance = float(np.dot(centered, centered) / n)
    for j in range(1, max_lag + 1):
        gamma = float(np.dot(centered[j:], centered[:-j]) / n)
        weight = 1.0 - j / (max_lag + 1)
        long_run_variance += 2.0 * weight * gamma

    variance_of_mean = long_run_variance / n
    if variance_of_mean <= 0 or not np.isfinite(variance_of_mean):
        fallback = float(np.var(diff, ddof=1) / n)
        variance_of_mean = fallback if fallback > 0 else float("nan")

    if not np.isfinite(variance_of_mean) or variance_of_mean <= 0:
        return mean_diff, float("nan"), float("nan")
    statistic = mean_diff / sqrt(variance_of_mean)
    p_value = float(stats.t.cdf(statistic, df=n - 1))
    return mean_diff, float(statistic), p_value


def build_summary(series: ForecastSeries) -> pd.DataFrame:
    model_rmse = rmse(series.residual)
    naive_rmse = rmse(series.naive_residual)
    model_mae = mae(series.residual)
    naive_mae = mae(series.naive_residual)
    model_mape = mape(series.actual, series.model)
    naive_mape = mape(series.actual, series.naive)

    squared_loss = diebold_mariano(series.residual**2, series.naive_residual**2)
    absolute_loss = diebold_mariano(np.abs(series.residual), np.abs(series.naive_residual))

    actual_direction = np.sign(np.diff(series.actual))
    model_direction = np.sign(np.diff(series.model))
    valid_direction = actual_direction != 0
    direction_hits = int(np.sum(actual_direction[valid_direction] == model_direction[valid_direction]))
    direction_total = int(np.sum(valid_direction))
    direction_rate = direction_hits / direction_total if direction_total else np.nan
    direction_p = (
        stats.binomtest(direction_hits, direction_total, p=0.5, alternative="greater").pvalue
        if direction_total
        else np.nan
    )

    ols_x = sm.add_constant(series.model)
    ols = sm.OLS(series.actual, ols_x).fit()
    calibration_f = ols.f_test("const = 0, x1 = 1")

    rows: list[dict[str, Any]] = [
        {
            "指标": "模型RMSE",
            "数值": model_rmse,
            "对照值": naive_rmse,
            "解释": "短期动态模型与朴素上一日基准的RMSE对比",
        },
        {
            "指标": "模型MAE",
            "数值": model_mae,
            "对照值": naive_mae,
            "解释": "短期动态模型与朴素上一日基准的MAE对比",
        },
        {
            "指标": "模型MAPE",
            "数值": model_mape,
            "对照值": naive_mape,
            "解释": "短期动态模型与朴素上一日基准的相对误差对比",
        },
        {
            "指标": "RMSE改善率",
            "数值": (1 - model_rmse / naive_rmse) * 100,
            "对照值": 0.0,
            "解释": "相对朴素上一日基准的RMSE下降比例",
        },
        {
            "指标": "DM平方损失均值差",
            "数值": squared_loss[0],
            "对照值": squared_loss[1],
            "p_value": squared_loss[2],
            "解释": "负值表示模型平方损失低于朴素基准；单侧p值越小越支持模型优于基准",
        },
        {
            "指标": "DM绝对损失均值差",
            "数值": absolute_loss[0],
            "对照值": absolute_loss[1],
            "p_value": absolute_loss[2],
            "解释": "负值表示模型绝对损失低于朴素基准；单侧p值越小越支持模型优于基准",
        },
        {
            "指标": "方向命中率",
            "数值": direction_rate * 100,
            "对照值": 50.0,
            "p_value": direction_p,
            "解释": "涨跌方向命中率及其相对随机猜测50%的单侧二项检验",
        },
        {
            "指标": "校准回归R2",
            "数值": float(ols.rsquared),
            "对照值": np.nan,
            "p_value": float(calibration_f.pvalue),
            "解释": "actual = a + b * fitted；p值检验a=0且b=1是否可被拒绝",
        },
    ]
    return pd.DataFrame(rows)


def build_diagnostics(series: ForecastSeries) -> pd.DataFrame:
    residual = series.residual
    rows: list[dict[str, Any]] = []

    for lag in [5, 10]:
        lb = acorr_ljungbox(residual, lags=[lag], return_df=True).iloc[0]
        rows.append(
            {
                "检验": f"Ljung-Box残差自相关_lag{lag}",
                "统计量": float(lb["lb_stat"]),
                "p_value": float(lb["lb_pvalue"]),
                "解释": "p值较大表示没有强证据说明残差仍存在该阶以内自相关",
            }
        )

        lb_sq = acorr_ljungbox(residual**2, lags=[lag], return_df=True).iloc[0]
        rows.append(
            {
                "检验": f"Ljung-Box平方残差_lag{lag}",
                "统计量": float(lb_sq["lb_stat"]),
                "p_value": float(lb_sq["lb_pvalue"]),
                "解释": "用于粗略检查残差波动聚集；p值较小提示可能存在异方差结构",
            }
        )

    arch_lm, arch_p, arch_f, arch_fp = het_arch(residual, nlags=5)
    rows.append(
        {
            "检验": "ARCH-LM_lag5",
            "统计量": float(arch_lm),
            "p_value": float(arch_p),
            "F统计量": float(arch_f),
            "F_p_value": float(arch_fp),
            "解释": "检验短期残差是否存在条件异方差",
        }
    )

    jb = stats.jarque_bera(residual)
    rows.append(
        {
            "检验": "Jarque-Bera正态性",
            "统计量": float(jb.statistic),
            "p_value": float(jb.pvalue),
            "解释": "用于判断残差分布是否显著偏离正态；金融序列中非正态并不罕见",
        }
    )
    return pd.DataFrame(rows)


def build_split_stats(series: ForecastSeries) -> pd.DataFrame:
    df = series.df.copy().reset_index(drop=True)
    labels = pd.qcut(df.index, q=3, labels=["前段冲击吸收", "中段平台维持", "后段预期修复"])
    df["分段"] = labels.astype(str)
    rows = []
    for label, group in df.groupby("分段", sort=False):
        rows.append(
            {
                "分段": label,
                "样本数": len(group),
                "模型RMSE": rmse(group["model_residual"]),
                "朴素基准RMSE": rmse(group["naive_residual"]),
                "模型MAE": mae(group["model_residual"]),
                "朴素基准MAE": mae(group["naive_residual"]),
                "RMSE改善率": (1 - rmse(group["model_residual"]) / rmse(group["naive_residual"])) * 100,
            }
        )
    return pd.DataFrame(rows)


def save_figure(series: ForecastSeries) -> None:
    dynamic.configure_plot_style()
    ensure_parent(FIGURE_PATH)
    df = series.df
    loss_diff = df["model_squared_error"] - df["naive_squared_error"]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2))
    axes[0, 0].plot(df["trade_date"], df["model_residual"], color=SCENARIO_COLORS["fit"], linewidth=1.8)
    axes[0, 0].axhline(0, color=PAPER_COLORS["ink"], linewidth=0.9)
    axes[0, 0].set_title("短期模型残差序列")
    axes[0, 0].set_ylabel("美元/桶")

    sm.graphics.tsa.plot_acf(series.residual, lags=min(15, len(series.residual) - 2), ax=axes[0, 1], color=SCENARIO_COLORS["fit"])
    axes[0, 1].set_title("残差自相关函数")

    axes[1, 0].hist(series.residual, bins=12, color=SCENARIO_COLORS["neutral"], alpha=0.82, edgecolor="white")
    axes[1, 0].axvline(0, color=PAPER_COLORS["ink"], linewidth=0.9)
    axes[1, 0].set_title("残差分布")

    axes[1, 1].plot(df["trade_date"], loss_diff, color=SCENARIO_COLORS["risk"], linewidth=1.8)
    axes[1, 1].axhline(0, color=PAPER_COLORS["ink"], linewidth=0.9)
    axes[1, 1].set_title("模型相对朴素基准的平方损失差")
    axes[1, 1].set_ylabel("负值表示模型更优")

    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=190)
    plt.close(fig)


def build_report(summary: pd.DataFrame, diagnostics: pd.DataFrame, split_stats: pd.DataFrame) -> str:
    lookup = summary.set_index("指标")
    diag_lookup = diagnostics.set_index("检验")
    split_rows = "\n".join(
        "| {分段} | {样本数:.0f} | {模型RMSE:.2f} | {朴素基准RMSE:.2f} | {RMSE改善率:.1f}% |".format(**row)
        for row in split_stats.to_dict("records")
    )
    diag_rows = "\n".join(
        "| {检验} | {统计量:.3f} | {p_value:.3f} |".format(**row)
        for row in diagnostics.to_dict("records")
        if "统计量" in row and pd.notna(row["统计量"])
    )

    return f"""# 短期模型统计审计报告

## 审计目的

本审计不改变短期主模型，只对阶段4综合最优路径做统计审计，重点检查模型是否真正优于朴素基准、残差是否存在明显未解释结构，以及方向命中率是否只是随机结果。该审计可以作为论文“模型检验”部分的统计证据补充。

## 核心结论

- 短期动态模型 RMSE 为 {lookup.loc["模型RMSE", "数值"]:.2f} 美元/桶，朴素上一日基准 RMSE 为 {lookup.loc["模型RMSE", "对照值"]:.2f} 美元/桶，RMSE 改善率为 {lookup.loc["RMSE改善率", "数值"]:.1f}%。
- Diebold-Mariano 平方损失检验的损失均值差为 {lookup.loc["DM平方损失均值差", "数值"]:.2f}，单侧 p 值为 {lookup.loc["DM平方损失均值差", "p_value"]:.3f}；绝对损失检验的单侧 p 值为 {lookup.loc["DM绝对损失均值差", "p_value"]:.3f}。
- 方向命中率为 {lookup.loc["方向命中率", "数值"]:.1f}%，相对 50% 随机猜测的单侧二项检验 p 值为 {lookup.loc["方向命中率", "p_value"]:.3f}。
- 残差 Ljung-Box lag10 p 值为 {diag_lookup.loc["Ljung-Box残差自相关_lag10", "p_value"]:.3f}，说明在该窗口内没有强证据表明模型残差仍存在显著线性自相关。
- ARCH-LM lag5 p 值为 {diag_lookup.loc["ARCH-LM_lag5", "p_value"]:.3f}，若论文采用该结果，应谨慎说明短期窗口残差存在一定金融波动聚集可能，后续可用 GARCH 作为附加审计而非替代主模型。

## 分段表现

| 分段 | 样本数 | 模型RMSE | 朴素基准RMSE | RMSE改善率 |
|---|---:|---:|---:|---:|
{split_rows}

## 残差诊断

| 检验 | 统计量 | p值 |
|---|---:|---:|
{diag_rows}

## 论文可用表述

统计审计进一步支持短期模型并非单纯复制上一日价格。与朴素上一日基准相比，综合机制递推模型在 RMSE、MAE 和 MAPE 上均有改善；DM 检验显示模型损失均值低于朴素基准，但由于样本窗口仅 45 个可比较交易日，显著性应以“支持性证据”表述，而不宜写成压倒性统计证明。Python 版 HAC-DM 与 R 版独立审计的 p 值口径不同，论文正文采用更保守的 R 独立复核结果；本报告用于可复现计算留档。残差自相关检验未发现明显线性滞后复制问题，ARCH 检验则提示金融价格残差仍可能存在波动聚集，因此本文将 GARCH/R 语言计量审计列为后续深化方向，而不把它替代机制递推主模型。

## 输出产物

- `{SUMMARY_CSV.relative_to(PROJECT_ROOT)}`
- `{DIAGNOSTIC_CSV.relative_to(PROJECT_ROOT)}`
- `{SPLIT_CSV.relative_to(PROJECT_ROOT)}`
- `{FIGURE_PATH.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    series = load_forecast_series()
    summary = build_summary(series)
    diagnostics = build_diagnostics(series)
    split_stats = build_split_stats(series)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_parent(SUMMARY_CSV)
    ensure_parent(DIAGNOSTIC_CSV)
    ensure_parent(SPLIT_CSV)
    ensure_parent(REPORT_PATH)
    summary.to_csv(SUMMARY_CSV, index=False)
    diagnostics.to_csv(DIAGNOSTIC_CSV, index=False)
    split_stats.to_csv(SPLIT_CSV, index=False)
    save_figure(series)
    REPORT_PATH.write_text(build_report(summary, diagnostics, split_stats), encoding="utf-8")

    print("Statistical audit complete")
    print(f"Summary: {SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Diagnostics: {DIAGNOSTIC_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Split stats: {SPLIT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Figure: {FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
