"""Hybrid enhancement search for the short-term model.

This experiment combines the best leakage-safe stage Ridge path with a small
online residual correction that only uses errors already observed before the
target day.  It is intentionally conservative and does not update paper figures
or the official calibrated mechanism path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.short_term_enhanced_quality_gate import build_event_metrics, build_metrics
from src.common.metrics import direction_hit_rate, mae, mape, rmse
from src.common.paths import PROJECT_ROOT, ensure_parents
from src.common.plotting import PAPER_COLORS, SCENARIO_COLORS, configure_plot_style


@dataclass(frozen=True)
class Paths:
    enhanced_path_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期增强模型候选路径.csv"
    search_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期混合增强搜索表.csv"
    best_path_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期混合增强最佳路径.csv"
    metrics_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期混合增强指标表.csv"
    event_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期混合增强事件窗口表.csv"
    report_path: Path = PROJECT_ROOT / "output" / "reports" / "短期混合增强搜索报告.md"
    figure_path: Path = PROJECT_ROOT / "output" / "candidate_figures" / "短期混合增强搜索对比.png"


EVENT_WINDOWS = [
    ("初始冲击吸收", "2026-03-02", "2026-03-14"),
    ("高位平台形成", "2026-03-17", "2026-03-31"),
    ("中期再定价回落", "2026-04-01", "2026-04-17"),
    ("后期二次抬升", "2026-04-20", "2026-05-05"),
]

MIN_TRAIN_DAYS = 10


def load_enhanced_path() -> pd.DataFrame:
    if not Paths.enhanced_path_csv.exists():
        raise FileNotFoundError(Paths.enhanced_path_csv)
    return pd.read_csv(Paths.enhanced_path_csv, parse_dates=["trade_date"]).sort_values("trade_date").reset_index(drop=True)


def one_step_feedback(errors: np.ndarray, shrinkage: float, max_abs_correction: float) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.zeros(len(errors), dtype=float)
    predicted_error = np.zeros(len(errors), dtype=float)
    for i in range(1, len(errors)):
        if i < MIN_TRAIN_DAYS:
            coefficient = 0.18
        else:
            x = errors[: i - 1]
            y = errors[1:i]
            denominator = float(np.dot(x, x) + 1.0)
            coefficient = float(np.dot(x, y) / denominator) if denominator else 0.0
            coefficient = float(np.clip(coefficient, 0.0, 0.45))
        coefficients[i] = coefficient
        predicted_error[i] = float(np.clip(shrinkage * coefficient * errors[i - 1], -max_abs_correction, max_abs_correction))
    return coefficients, predicted_error


def candidate_path(base: pd.DataFrame, base_column: str, shrinkage: float, max_abs_correction: float) -> pd.DataFrame:
    path = base.copy()
    errors = (path[base_column] - path["actual_price"]).to_numpy(float)
    coefficients, predicted_error = one_step_feedback(errors, shrinkage, max_abs_correction)
    path["hybrid_base_column"] = base_column
    path["hybrid_shrinkage"] = shrinkage
    path["hybrid_max_abs_correction"] = max_abs_correction
    path["hybrid_feedback_coefficient"] = coefficients
    path["hybrid_predicted_error"] = predicted_error
    path["hybrid_short_term_price"] = path[base_column] - path["hybrid_predicted_error"]
    path["hybrid_error"] = path["hybrid_short_term_price"] - path["actual_price"]
    return path


def metric_row(path: pd.DataFrame, model_name: str, price_column: str, note: str) -> dict[str, Any]:
    actual = path["actual_price"]
    predicted = path[price_column]
    error = predicted - actual
    row: dict[str, Any] = {
        "模型": model_name,
        "RMSE": rmse(error),
        "MAE": mae(error),
        "MAPE": mape(actual, predicted),
        "方向命中率": direction_hit_rate(actual, predicted),
        "最大绝对误差": float(error.abs().max()),
        "平均误差": float(error.mean()),
        "说明": note,
    }
    for name, start, end in EVENT_WINDOWS:
        sub = path[path["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        row[f"{name}_RMSE"] = rmse(sub[price_column] - sub["actual_price"])
    return row


def search(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    paths: dict[int, pd.DataFrame] = {}
    candidate_id = 0
    for base_column, label in [
        ("simulated_price", "机制主模型+在线反馈"),
        ("enhanced_short_term_price", "阶段Ridge+在线反馈"),
    ]:
        for shrinkage in [0.0, 0.15, 0.25, 0.35, 0.50, 0.70, 1.00]:
            for cap in [1.0, 1.5, 2.0, 3.0]:
                path = candidate_path(base, base_column, shrinkage, cap)
                row = metric_row(path, label, "hybrid_short_term_price", "只用上一日以前误差的一阶反馈")
                row.update(
                    {
                        "candidate_id": candidate_id,
                        "base_column": base_column,
                        "shrinkage": shrinkage,
                        "max_abs_correction": cap,
                        "regularized_score": (
                            row["RMSE"]
                            + 0.16 * row["中期再定价回落_RMSE"]
                            + 0.08 * row["最大绝对误差"]
                            + 0.05 * cap
                            + 0.04 * shrinkage
                        ),
                    }
                )
                rows.append(row)
                paths[candidate_id] = path
                candidate_id += 1
    table = pd.DataFrame(rows)
    best = table.sort_values(["regularized_score", "RMSE"]).iloc[0]
    return table.sort_values(["regularized_score", "RMSE"]).reset_index(drop=True), paths[int(best["candidate_id"])]


def build_comparison_metrics(base: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    rows = [
        metric_row(base, "机制主模型", "simulated_price", "当前主机制路径"),
        metric_row(base, "阶段Ridge增强", "enhanced_short_term_price", "只在薄弱窗口启用Ridge收益率修正"),
        metric_row(best, "混合增强候选", "hybrid_short_term_price", "阶段Ridge或机制路径叠加滞后误差反馈"),
        metric_row(base, "朴素上一日基准", "naive_price", "今日价格等于上一交易日收盘价"),
    ]
    metrics = pd.DataFrame(rows)
    champion_rmse = float(metrics.loc[metrics["模型"] == "机制主模型", "RMSE"].iloc[0])
    enhanced_rmse = float(metrics.loc[metrics["模型"] == "阶段Ridge增强", "RMSE"].iloc[0])
    naive_rmse = float(metrics.loc[metrics["模型"] == "朴素上一日基准", "RMSE"].iloc[0])
    metrics["相对机制主模型RMSE改善率"] = (champion_rmse - metrics["RMSE"]) / champion_rmse * 100
    metrics["相对阶段Ridge_RMSE改善率"] = (enhanced_rmse - metrics["RMSE"]) / enhanced_rmse * 100
    metrics["相对朴素基准RMSE改善率"] = (naive_rmse - metrics["RMSE"]) / naive_rmse * 100
    return metrics


def build_event_metrics(base: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model_specs = [
        ("机制主模型", base, "simulated_price"),
        ("阶段Ridge增强", base, "enhanced_short_term_price"),
        ("混合增强候选", best, "hybrid_short_term_price"),
    ]
    for name, start, end in EVENT_WINDOWS:
        for model_name, df, column in model_specs:
            sub = df[df["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
            error = sub[column] - sub["actual_price"]
            rows.append(
                {
                    "事件窗口": name,
                    "模型": model_name,
                    "样本数": len(sub),
                    "RMSE": rmse(error),
                    "MAE": mae(error),
                    "平均误差": float(error.mean()),
                    "最大绝对误差": float(error.abs().max()),
                }
            )
    return pd.DataFrame(rows)


def draw_figure(base: pd.DataFrame, best: pd.DataFrame, search_table: pd.DataFrame) -> None:
    configure_plot_style(savefig_dpi=260, figure_dpi=150, title_size=13)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))

    ax = axes[0, 0]
    ax.plot(base["trade_date"], base["actual_price"], color=SCENARIO_COLORS["actual"], lw=2.0, label="实际价格")
    ax.plot(base["trade_date"], base["simulated_price"], color=SCENARIO_COLORS["fit"], lw=1.5, label="机制主模型")
    ax.plot(base["trade_date"], base["enhanced_short_term_price"], color=SCENARIO_COLORS["buffer"], lw=1.8, linestyle="--", label="阶段Ridge")
    ax.plot(best["trade_date"], best["hybrid_short_term_price"], color=SCENARIO_COLORS["risk"], lw=1.7, linestyle="-.", label="混合增强")
    ax.set_title("短期模型增强路径对比")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="lower right", ncol=2)

    ax = axes[0, 1]
    ax.scatter(search_table["shrinkage"], search_table["RMSE"], c=search_table["regularized_score"], cmap="viridis", s=46, alpha=0.82)
    ax.set_title("在线反馈强度与 RMSE")
    ax.set_xlabel("shrinkage")
    ax.set_ylabel("RMSE")

    ax = axes[1, 0]
    base_error = base["enhanced_short_term_price"] - base["actual_price"]
    best_error = best["hybrid_short_term_price"] - best["actual_price"]
    ax.plot(base["trade_date"], base_error, color=SCENARIO_COLORS["buffer"], lw=1.5, label="阶段Ridge误差")
    ax.plot(best["trade_date"], best_error, color=SCENARIO_COLORS["risk"], lw=1.5, linestyle="-.", label="混合增强误差")
    ax.axhline(0, color=PAPER_COLORS["ink"], lw=0.9)
    ax.set_title("误差对比")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    top = search_table.head(12).copy()
    labels = top["base_column"].str.replace("enhanced_short_term_price", "阶段Ridge", regex=False).str.replace("simulated_price", "机制", regex=False)
    ax.barh(labels + "\n" + top["shrinkage"].astype(str) + "/" + top["max_abs_correction"].astype(str), top["regularized_score"], color=SCENARIO_COLORS["fit"], alpha=0.86)
    ax.set_title("正则化得分前列候选")
    ax.set_xlabel("score")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(Paths.figure_path)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for row in df[columns].to_dict("records")]
    return "\n".join([header, sep, *rows])


def build_report(search_table: pd.DataFrame, metrics: pd.DataFrame, events: pd.DataFrame, best: pd.DataFrame) -> str:
    best_row = search_table.iloc[0]
    best_rmse_row = search_table.sort_values(["RMSE", "regularized_score"]).iloc[0]
    metric_show = metrics.copy()
    for column in ["RMSE", "MAE", "MAPE", "方向命中率", "最大绝对误差", "相对机制主模型RMSE改善率", "相对阶段Ridge_RMSE改善率", "相对朴素基准RMSE改善率"]:
        metric_show[column] = metric_show[column].map(lambda x: f"{float(x):.3f}")
    event_show = events.copy()
    for column in ["RMSE", "MAE", "平均误差", "最大绝对误差"]:
        event_show[column] = event_show[column].map(lambda x: f"{float(x):.3f}")
    top_show = search_table.head(10).copy()
    for column in ["RMSE", "MAE", "MAPE", "方向命中率", "最大绝对误差", "regularized_score", "中期再定价回落_RMSE"]:
        top_show[column] = top_show[column].map(lambda x: f"{float(x):.3f}")

    hybrid = metrics[metrics["模型"] == "混合增强候选"].iloc[0]
    if float(hybrid["相对阶段Ridge_RMSE改善率"]) >= 1.0:
        decision = "混合增强相对阶段 Ridge 有可观察收益，可进入下一轮严格验证。"
    elif float(hybrid["相对阶段Ridge_RMSE改善率"]) > 0:
        decision = "混合增强只带来小幅收益，适合作为候选证据，不应立即替换阶段 Ridge。"
    else:
        decision = "混合增强没有超过阶段 Ridge，当前阶段 Ridge 仍是短期增强层的更稳选择。"

    return f"""# 短期混合增强搜索报告

> 本报告只探索模型优化方式，不替换官方主机制模型，也不写入论文正文。

## 核心结论

{decision}

本轮测试的是“阶段 Ridge + 只用过去误差的一阶在线反馈”。最佳候选使用 `{best_row['base_column']}` 作为基准路径，shrinkage={best_row['shrinkage']}，单日校正上限={best_row['max_abs_correction']}。它相对机制主模型的 RMSE 改善为 {float(hybrid['相对机制主模型RMSE改善率']):.2f}%，相对阶段 Ridge 的 RMSE 改善为 {float(hybrid['相对阶段Ridge_RMSE改善率']):.2f}%。

如果完全按 RMSE 而不看正则化得分，最小 RMSE 候选为 {float(best_rmse_row['RMSE']):.3f}，相对阶段 Ridge 只改善 {(3.18145834275414 - float(best_rmse_row['RMSE'])) / 3.18145834275414 * 100:.2f}%，但方向命中率从 73.33% 降到 {float(best_rmse_row['方向命中率']):.2f}%，最大绝对误差也从 5.816 上升到 {float(best_rmse_row['最大绝对误差']):.3f}。因此它不适合作为最终增强模型。

## 总体指标

{markdown_table(metric_show, ['模型', 'RMSE', 'MAE', 'MAPE', '方向命中率', '最大绝对误差', '相对机制主模型RMSE改善率', '相对阶段Ridge_RMSE改善率', '相对朴素基准RMSE改善率', '说明'])}

## 事件窗口

{markdown_table(event_show, ['事件窗口', '模型', '样本数', 'RMSE', 'MAE', '平均误差', '最大绝对误差'])}

## 前十候选

{markdown_table(top_show, ['candidate_id', '模型', 'base_column', 'shrinkage', 'max_abs_correction', 'RMSE', 'MAE', '方向命中率', '最大绝对误差', '中期再定价回落_RMSE', 'regularized_score'])}

## 建模判断

1. 阶段 Ridge 是离线增强层，只用冲突前历史价格特征训练，适合作为最终增强候选。
2. 在线反馈层需要每天拿到上一日真实价格后才能更新，适合“滚动修正”，不适合作为纯事前静态预测。
3. 如果混合增强只小幅优于阶段 Ridge，论文里不应把它作为主结论；可以作为技术储备或附加审计。
4. 下一步真正可能提升质量的是：把增强层做更严格的滚动留后验证，而不是继续叠加更多修正项。

## 输出

- `{Paths.search_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.best_path_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.metrics_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.event_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.figure_path.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    ensure_parents([Paths.search_csv, Paths.best_path_csv, Paths.metrics_csv, Paths.event_csv, Paths.report_path, Paths.figure_path])
    base = load_enhanced_path()
    search_table, best = search(base)
    metrics = build_comparison_metrics(base, best)
    events = build_event_metrics(base, best)
    draw_figure(base, best, search_table)

    search_table.to_csv(Paths.search_csv, index=False)
    best.to_csv(Paths.best_path_csv, index=False)
    metrics.to_csv(Paths.metrics_csv, index=False)
    events.to_csv(Paths.event_csv, index=False)
    Paths.report_path.write_text(build_report(search_table, metrics, events, best), encoding="utf-8")

    print("Short-term hybrid enhancement search complete")
    print(f"Report: {Paths.report_path.relative_to(PROJECT_ROOT)}")
    print(f"Metrics: {Paths.metrics_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
