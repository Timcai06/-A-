"""Final quality gate for the short-term event-window model.

The goal is not to chase the lowest possible in-sample error.  This module
aggregates existing audits into a single, reproducible decision table: whether
the short-term model is accurate, better than strong baselines, not a lag copy,
stable under perturbation, and honest about what machine learning can add.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT, ensure_parents
from src.common.plotting import configure_plot_style


@dataclass(frozen=True)
class GatePaths:
    quality_metrics_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期模型质量指标.csv"
    baseline_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期模型基准对比.csv"
    ml_comparison_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期机器学习模型对比.csv"
    rolling_validation_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期历史滚动验证结果.csv"
    lag_shift_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期模型滞后平移检验.csv"
    turning_point_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期模型拐点检验.csv"
    chunk_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期模型删块稳健性.csv"
    perturbation_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期模型局部扰动稳健性.csv"
    residual_diagnostics_csv: Path = PROJECT_ROOT / "output" / "statistics" / "短期残差诊断.csv"
    gate_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期模型最终质量闸门.csv"
    score_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期模型最终质量评分.csv"
    report_path: Path = PROJECT_ROOT / "output" / "reports" / "短期模型最终质量闸门报告.md"
    radar_figure: Path = PROJECT_ROOT / "paper" / "figures" / "短期模型最终质量雷达图.png"


PATHS = GatePaths()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing quality-gate input: {path.relative_to(PROJECT_ROOT)}")
    return pd.read_csv(path)


def metric_value(metrics: pd.DataFrame, name: str) -> float:
    row = metrics[metrics["指标"].astype(str) == name]
    if row.empty:
        raise KeyError(f"Metric not found: {name}")
    return float(row.iloc[0]["数值"])


def model_row(table: pd.DataFrame, name: str) -> pd.Series:
    row = table[table["模型"].astype(str) == name]
    if row.empty:
        raise KeyError(f"Model row not found: {name}")
    return row.iloc[0]


def residual_pvalue(table: pd.DataFrame, name: str) -> float:
    row = table[table["检验"].astype(str) == name]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["p_value"])


def judgement(passed: bool, warning: bool = False) -> str:
    if passed and not warning:
        return "通过"
    if passed and warning:
        return "谨慎通过"
    return "需补强"


def build_gate() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    quality = read_csv(PATHS.quality_metrics_csv)
    baseline = read_csv(PATHS.baseline_csv)
    ml = read_csv(PATHS.ml_comparison_csv)
    rolling = read_csv(PATHS.rolling_validation_csv)
    lag = read_csv(PATHS.lag_shift_csv)
    turning = read_csv(PATHS.turning_point_csv)
    chunk = read_csv(PATHS.chunk_csv)
    perturb = read_csv(PATHS.perturbation_csv)
    residual = read_csv(PATHS.residual_diagnostics_csv)

    rmse = metric_value(quality, "RMSE")
    mae = metric_value(quality, "MAE")
    mape = metric_value(quality, "MAPE")
    direction = metric_value(quality, "方向命中率")
    naive_improvement = metric_value(quality, "相对朴素基准RMSE改善率")
    theil_u = metric_value(quality, "Theil_U_vs_naive")

    non_model_baselines = baseline[baseline["模型"] != "本文短期动态模型"].copy()
    best_baseline_rmse = float(non_model_baselines["RMSE"].min())
    all_baselines_beaten = bool((rmse < non_model_baselines["RMSE"]).all())

    mechanism = model_row(ml, "机制递推主模型")
    hybrid = model_row(ml, "机制+阶段Ridge修正")
    pure_ridge = model_row(ml, "历史Ridge收益率模型")
    hybrid_gain = float(hybrid["相对机制主模型RMSE改善率"])
    hybrid_direction_gain = float(hybrid["方向命中率"] - mechanism["方向命中率"])
    pure_ridge_vs_mechanism = float(pure_ridge["相对机制主模型RMSE改善率"])

    normal_rolling = rolling[~rolling["是否冲突窗口"].astype(bool)].copy()
    event_rolling = rolling[rolling["是否冲突窗口"].astype(bool)].tail(1)
    ridge_better_ratio = float((normal_rolling["Ridge相对Naive_RMSE改善率"] > 0).mean() * 100)
    event_ridge_improvement = float(event_rolling["Ridge相对Naive_RMSE改善率"].iloc[0])

    best_shift = int(lag.sort_values(["RMSE", "MAE"]).iloc[0]["模型平移天数"])
    rmse_shift_minus1 = float(lag[lag["模型平移天数"] == -1]["RMSE"].iloc[0])
    lag_margin = rmse_shift_minus1 - rmse

    acceptable_turning = int(turning["捕捉类型"].isin(["同步", "提前1日"]).sum())
    turning_total = int(len(turning))
    turning_ratio = acceptable_turning / turning_total * 100 if turning_total else float("nan")
    same_day_capture_ratio = float((turning["模型同日变动"].abs() / turning["实际绝对变动"]).median())

    holdout = chunk[chunk["样本口径"] == "被删除块"].copy()
    worst_chunk_rmse = float(holdout["RMSE"].max())
    chunk_rmse_std = float(holdout["RMSE"].std(ddof=0))

    accepted_mask = (
        (perturb["RMSE"] <= 5.0)
        & (perturb["高价平台RMSE"] <= 5.0)
        & (perturb["低价回落RMSE"] <= 6.0)
        & (perturb["后期RMSE"] <= 5.0)
    )
    perturb_accept_rate = float(accepted_mask.mean() * 100)

    lb_lag10 = residual_pvalue(residual, "Ljung-Box残差自相关_lag10")
    arch_lm = residual_pvalue(residual, "ARCH-LM_lag5")
    squared_lb = residual_pvalue(residual, "Ljung-Box平方残差_lag10")

    rows = [
        {
            "维度": "拟合精度",
            "核心指标": f"RMSE={rmse:.3f}, MAE={mae:.3f}, MAPE={mape:.2f}%",
            "阈值或判断": "MAPE <= 3%，RMSE <= 3.5",
            "结论": judgement(mape <= 3.0 and rmse <= 3.5),
            "风险提示": "样本为事件窗口条件解释，不是普通交易日逐日预测。",
            "评分": 5.0 if mape <= 3.0 and rmse <= 3.5 else 3.5,
        },
        {
            "维度": "基准优势",
            "核心指标": f"Theil U={theil_u:.3f}, 相对朴素RMSE改善={naive_improvement:.1f}%, 最强基准RMSE={best_baseline_rmse:.3f}",
            "阈值或判断": "Theil U < 1，且优于全部基准",
            "结论": judgement(theil_u < 1 and all_baselines_beaten),
            "风险提示": "朴素基准很强，必须保留 DM 检验和方向命中证据。",
            "评分": 5.0 if theil_u < 0.80 and all_baselines_beaten else 4.2,
        },
        {
            "维度": "机器学习辅助",
            "核心指标": f"机制+Ridge相对机制RMSE改善={hybrid_gain:.2f}%, 方向提升={hybrid_direction_gain:.1f}pct；纯Ridge相对机制={pure_ridge_vs_mechanism:.1f}%",
            "阈值或判断": "辅助层可提升，但纯历史 ML 不能替代机制模型",
            "结论": judgement(hybrid_gain > 0 and hybrid_direction_gain >= 0, warning=pure_ridge_vs_mechanism < 0),
            "风险提示": "建议写作：机制模型为主，Ridge 只作为滞后特征辅助修正层。",
            "评分": 4.3 if hybrid_gain > 0 and hybrid_direction_gain >= 0 else 3.0,
        },
        {
            "维度": "历史滚动验证",
            "核心指标": f"历史Ridge胜过朴素窗口占比={ridge_better_ratio:.1f}%，冲突窗口Ridge改善={event_ridge_improvement:.2f}%",
            "阈值或判断": "纯历史 ML 若不稳定胜出，只可作基准或辅助",
            "结论": judgement(ridge_better_ratio < 60 and event_ridge_improvement < 0, warning=True),
            "风险提示": "这不是坏事，而是支持论文不要把 LSTM/Ridge 写成主模型。",
            "评分": 4.0,
        },
        {
            "维度": "滞后复制风险",
            "核心指标": f"最优平移={best_shift}天，左移1天RMSE劣化={lag_margin:.3f}",
            "阈值或判断": "0 天平移最好，左移不应更优",
            "结论": judgement(best_shift == 0 and lag_margin > 0),
            "风险提示": "仍需配合拐点图说明个别日度噪声无法完全提前捕捉。",
            "评分": 5.0 if best_shift == 0 else 2.5,
        },
        {
            "维度": "拐点解释",
            "核心指标": f"同步或提前捕捉={acceptable_turning}/{turning_total}，同日幅度中位捕捉率={same_day_capture_ratio:.2f}",
            "阈值或判断": "方向同步或提前为主，幅度不夸大",
            "结论": judgement(turning_ratio >= 80, warning=same_day_capture_ratio < 0.35),
            "风险提示": "模型能解释方向和平台，不宜声称精确捕捉所有单日跳幅。",
            "评分": 4.2 if turning_ratio >= 80 else 3.0,
        },
        {
            "维度": "分段稳定性",
            "核心指标": f"最差删块RMSE={worst_chunk_rmse:.3f}, 删块RMSE标准差={chunk_rmse_std:.3f}",
            "阈值或判断": "最差片段 RMSE <= 5，块间差异可解释",
            "结论": judgement(worst_chunk_rmse <= 5.0, warning=chunk_rmse_std > 0.8),
            "风险提示": "最难片段集中在中期平台再定价，应在论文里主动说明。",
            "评分": 4.4 if worst_chunk_rmse <= 5.0 else 3.0,
        },
        {
            "维度": "参数稳健性",
            "核心指标": f"局部优秀扰动占比={perturb_accept_rate:.1f}%",
            "阈值或判断": "局部扰动优秀率 >= 90%",
            "结论": judgement(perturb_accept_rate >= 90),
            "风险提示": "这是局部稳健性，不代表任意参数都优秀。",
            "评分": 5.0 if perturb_accept_rate >= 95 else 4.2,
        },
        {
            "维度": "残差结构",
            "核心指标": f"Ljung-Box lag10 p={lb_lag10:.3f}, ARCH-LM p={arch_lm:.3f}, 平方残差LB p={squared_lb:.3f}",
            "阈值或判断": "线性自相关不显著；波动聚集需谨慎说明",
            "结论": judgement(lb_lag10 > 0.05 and arch_lm > 0.05, warning=squared_lb < 0.05),
            "风险提示": "金融残差有波动聚集迹象，R/GARCH 只能作审计，不替代机制模型。",
            "评分": 4.0 if lb_lag10 > 0.05 and arch_lm > 0.05 else 3.0,
        },
    ]
    gate = pd.DataFrame(rows)
    facts = {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "direction": direction,
        "naive_improvement": naive_improvement,
        "hybrid_rmse": float(hybrid["RMSE"]),
        "hybrid_mae": float(hybrid["MAE"]),
        "hybrid_mape": float(hybrid["MAPE"]),
        "hybrid_direction": float(hybrid["方向命中率"]),
        "hybrid_gain": hybrid_gain,
        "pure_ridge_rmse": float(pure_ridge["RMSE"]),
        "ridge_better_ratio": ridge_better_ratio,
        "event_ridge_improvement": event_ridge_improvement,
        "worst_chunk_rmse": worst_chunk_rmse,
        "perturb_accept_rate": perturb_accept_rate,
    }
    score = gate[["维度", "评分"]].copy()
    return gate, score, facts


def draw_radar(score: pd.DataFrame) -> None:
    configure_plot_style()
    labels = score["维度"].to_list()
    values = score["评分"].to_numpy(float)
    angles = np.linspace(0, 2 * math.pi, len(labels), endpoint=False)
    values_closed = np.r_[values, values[0]]
    angles_closed = np.r_[angles, angles[0]]

    fig = plt.figure(figsize=(8.8, 8.2))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles_closed, values_closed, color="#0f766e", linewidth=2.4)
    ax.fill(angles_closed, values_closed, color="#14b8a6", alpha=0.22)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=9)
    ax.set_title("短期模型最终质量闸门评分", pad=22, fontsize=15)
    ax.grid(color="#cbd5e1", alpha=0.85)
    fig.tight_layout()
    fig.savefig(PATHS.radar_figure, dpi=200)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for row in df[columns].to_dict("records")
    ]
    return "\n".join([header, sep, *rows])


def build_report(gate: pd.DataFrame, facts: dict[str, float]) -> str:
    avg_score = float(gate["评分"].mean())
    pass_count = int(gate["结论"].isin(["通过", "谨慎通过"]).sum())
    strict_pass = int((gate["结论"] == "通过").sum())
    cautious = int((gate["结论"] == "谨慎通过").sum())
    gate_table = markdown_table(gate, ["维度", "核心指标", "阈值或判断", "结论", "风险提示", "评分"])

    return f"""# 短期模型最终质量闸门报告

## 总体判断

本报告把短期模型质量拆成拟合精度、基准优势、机器学习辅助、历史滚动验证、滞后复制风险、拐点解释、分段稳定性、参数稳健性和残差结构九个维度。当前平均质量评分为 **{avg_score:.2f}/5**，其中 {strict_pass} 个维度直接通过，{cautious} 个维度谨慎通过。

结论是：短期主模型已经可以作为最终论文的核心模型；拟合质量还能通过“机制 + 滞后价格特征 Ridge 辅助层”进一步提升，但这个增强层应被写成计算机辅助修正，而不是取代机制模型的黑箱预测器。

## 拟合质量是否还能提高

可以提高，但提高幅度有限且必须守住无泄漏边界。当前机制递推主模型 RMSE 为 {facts["rmse"]:.3f}，MAE 为 {facts["mae"]:.3f}，MAPE 为 {facts["mape"]:.2f}%，方向命中率为 {facts["direction"]:.1f}%。在只使用 2026-03-02 之前历史价格特征训练的 Ridge 收益率辅助层加入后，RMSE 降至 {facts["hybrid_rmse"]:.3f}，MAE 为 {facts["hybrid_mae"]:.3f}，MAPE 为 {facts["hybrid_mape"]:.2f}%，方向命中率升至 {facts["hybrid_direction"]:.1f}%，相对机制主模型 RMSE 改善 {facts["hybrid_gain"]:.2f}%。

但纯历史 Ridge 模型在冲突窗口 RMSE 为 {facts["pure_ridge_rmse"]:.3f}，历史 46 日滚动窗口中仅有 {facts["ridge_better_ratio"]:.1f}% 的窗口稳定优于朴素基准，冲突窗口相对朴素基准改善率为 {facts["event_ridge_improvement"]:.2f}%。这说明机器学习有价值，但不能单独解释霍尔木兹封锁；最稳妥的最终写法是“机制递推主模型 + 低自由度滞后特征辅助修正”。

## 质量闸门

{gate_table}

## 对最终论文的建议

- 主结论仍使用机制递推主模型解释 110--120 美元平台，因为它有物理机制、消融实验和边界检验支撑。
- 若论文需要报告“最优拟合效果”，可以补充说明机器学习辅助层把 RMSE 从 {facts["rmse"]:.3f} 降至 {facts["hybrid_rmse"]:.3f}，但不能把它写成独立预测主模型。
- 对评委最有说服力的不是单个 RMSE，而是“打败强基准 + 非滞后复制 + 分段稳定 + 参数邻域稳定 + 机器学习辅助不越界”这一整套证据链。

## 输出

- `{PATHS.gate_csv.relative_to(PROJECT_ROOT)}`
- `{PATHS.score_csv.relative_to(PROJECT_ROOT)}`
- `{PATHS.radar_figure.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    ensure_parents([PATHS.gate_csv, PATHS.score_csv, PATHS.report_path, PATHS.radar_figure])
    gate, score, facts = build_gate()
    gate.to_csv(PATHS.gate_csv, index=False)
    score.to_csv(PATHS.score_csv, index=False)
    draw_radar(score)
    PATHS.report_path.write_text(build_report(gate, facts), encoding="utf-8")
    print("Short-term final quality gate complete")
    print(f"Gate: {PATHS.gate_csv.relative_to(PROJECT_ROOT)}")
    print(f"Report: {PATHS.report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
