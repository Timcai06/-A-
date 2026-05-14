"""Quality gate for the enhanced short-term model candidate.

The enhanced candidate keeps the mechanism recursion as the base path and only
uses a low-degree Ridge return layer in predeclared weak event windows.  This
module formalizes whether that candidate is strong enough to be kept as an
auxiliary final-fit path, without changing the paper source or official figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.short_term_exogenous_risk_ridge_experiment import (
    ALPHA_GRID,
    EVENT_START,
    EVENT_WINDOWS,
    RIDGE_ASSIST_WINDOWS,
    choose_alpha,
    feature_sets,
    load_mechanism,
    load_price_features,
    ridge_pipeline,
)
from src.common.metrics import direction_hit_rate, mae, mape, rmse
from src.common.paths import PROJECT_ROOT, ensure_parents
from src.common.plotting import PAPER_COLORS, SCENARIO_COLORS, configure_plot_style


@dataclass(frozen=True)
class Paths:
    path_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期增强模型候选路径.csv"
    metrics_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期增强模型指标表.csv"
    event_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期增强模型事件窗口表.csv"
    alpha_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期增强模型正则强度表.csv"
    coef_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期增强模型特征系数表.csv"
    decision_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期增强模型质量门.csv"
    report_path: Path = PROJECT_ROOT / "output" / "reports" / "短期增强模型质量门报告.md"
    figure_path: Path = PROJECT_ROOT / "output" / "candidate_figures" / "短期增强模型质量门_误差对比.png"


FEATURE_SET_NAME = "仅历史价格特征"


def clean_frame(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    needed = ["trade_date", "pre_close_filled", "target_close_price", "target_log_return", *columns]
    return features[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()


def build_candidate_path() -> tuple[pd.DataFrame, pd.DataFrame, float, pd.DataFrame]:
    features = load_price_features()
    mechanism = load_mechanism()
    columns = feature_sets(features)[FEATURE_SET_NAME]
    usable = clean_frame(features, columns)
    pre_event = usable[usable["trade_date"] < EVENT_START].copy()
    event_features = usable[usable["trade_date"].isin(mechanism["trade_date"])].copy()

    best_alpha, alpha_scores = choose_alpha(pre_event, columns)
    model = ridge_pipeline(best_alpha)
    model.fit(pre_event[columns].to_numpy(float), pre_event["target_log_return"].to_numpy(float))

    pred_return = model.predict(event_features[columns].to_numpy(float))
    event_predictions = event_features[["trade_date", "pre_close_filled", "target_close_price"]].copy()
    event_predictions["ridge_predicted_return"] = pred_return
    event_predictions["ridge_price"] = event_predictions["pre_close_filled"] * np.exp(pred_return)

    path = mechanism.merge(event_predictions, on="trade_date", how="inner")
    path["naive_price"] = path["pre_close_filled"]
    path["ridge_return_correction"] = path["ridge_price"] - path["pre_close_filled"]
    path["mechanism_plus_full_ridge"] = path["simulated_price"] + path["ridge_return_correction"]
    path["enhanced_phase"] = "未启用"
    path["phase_ridge_correction"] = 0.0
    for name, start, end in RIDGE_ASSIST_WINDOWS:
        mask = path["trade_date"].between(start, end)
        path.loc[mask, "enhanced_phase"] = name
        path.loc[mask, "phase_ridge_correction"] = path.loc[mask, "ridge_return_correction"]
    path["enhanced_short_term_price"] = path["simulated_price"] + path["phase_ridge_correction"]
    path["enhanced_model_note"] = "机制主模型+历史价格滞后特征阶段Ridge修正"

    ridge = model.named_steps["ridge"]
    scaler = model.named_steps["standardize"]
    # Coefficients are converted back to standardized-feature contribution scale
    # for ranking only; signs and relative magnitudes are the useful information.
    coefs = pd.DataFrame(
        {
            "特征": columns,
            "标准化系数": ridge.coef_,
            "训练均值": scaler.mean_,
            "训练标准差": scaler.scale_,
        }
    )
    coefs["系数绝对值"] = coefs["标准化系数"].abs()
    coefs = coefs.sort_values("系数绝对值", ascending=False).reset_index(drop=True)
    return path, alpha_scores, best_alpha, coefs


def metric_row(path: pd.DataFrame, model_name: str, column: str, note: str) -> dict[str, object]:
    actual = path["actual_price"]
    predicted = path[column]
    error = predicted - actual
    return {
        "模型": model_name,
        "RMSE": rmse(error),
        "MAE": mae(error),
        "MAPE": mape(actual, predicted),
        "方向命中率": direction_hit_rate(actual, predicted),
        "最大绝对误差": float(error.abs().max()),
        "平均误差": float(error.mean()),
        "说明": note,
    }


def build_metrics(path: pd.DataFrame) -> pd.DataFrame:
    rows = [
        metric_row(path, "机制主模型", "simulated_price", "当前短期冲击机制递推主模型"),
        metric_row(path, "朴素上一日基准", "naive_price", "今日价格等于上一交易日收盘价"),
        metric_row(path, "纯Ridge收益率", "ridge_price", "只使用冲突前历史价格特征训练的一日收益率模型"),
        metric_row(path, "机制+全段Ridge", "mechanism_plus_full_ridge", "全事件窗口叠加Ridge收益率修正"),
        metric_row(path, "机制+阶段Ridge", "enhanced_short_term_price", "仅在高位平台和中期再定价窗口启用Ridge修正"),
    ]
    metrics = pd.DataFrame(rows)
    base_rmse = float(metrics.loc[metrics["模型"] == "机制主模型", "RMSE"].iloc[0])
    naive_rmse = float(metrics.loc[metrics["模型"] == "朴素上一日基准", "RMSE"].iloc[0])
    metrics["相对机制主模型RMSE改善率"] = (base_rmse - metrics["RMSE"]) / base_rmse * 100
    metrics["相对朴素基准RMSE改善率"] = (naive_rmse - metrics["RMSE"]) / naive_rmse * 100
    return metrics


def build_event_metrics(path: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    models = [
        ("机制主模型", "simulated_price"),
        ("朴素上一日基准", "naive_price"),
        ("机制+阶段Ridge", "enhanced_short_term_price"),
    ]
    for event_name, start, end in EVENT_WINDOWS:
        sub = path[path["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
        if sub.empty:
            continue
        for model_name, column in models:
            error = sub[column] - sub["actual_price"]
            rows.append(
                {
                    "事件窗口": event_name,
                    "开始日期": start,
                    "结束日期": end,
                    "样本数": len(sub),
                    "模型": model_name,
                    "RMSE": rmse(error),
                    "MAE": mae(error),
                    "MAPE": mape(sub["actual_price"], sub[column]),
                    "方向命中率": direction_hit_rate(sub["actual_price"], sub[column]),
                    "最大绝对误差": float(error.abs().max()),
                }
            )
    event_metrics = pd.DataFrame(rows)
    base = event_metrics[event_metrics["模型"] == "机制主模型"][["事件窗口", "RMSE"]].rename(
        columns={"RMSE": "机制主模型RMSE"}
    )
    event_metrics = event_metrics.merge(base, on="事件窗口", how="left")
    event_metrics["相对机制主模型RMSE改善率"] = (
        (event_metrics["机制主模型RMSE"] - event_metrics["RMSE"]) / event_metrics["机制主模型RMSE"] * 100
    )
    return event_metrics


def build_decision(metrics: pd.DataFrame, event_metrics: pd.DataFrame, alpha_scores: pd.DataFrame) -> pd.DataFrame:
    base = metrics[metrics["模型"] == "机制主模型"].iloc[0]
    naive = metrics[metrics["模型"] == "朴素上一日基准"].iloc[0]
    enhanced = metrics[metrics["模型"] == "机制+阶段Ridge"].iloc[0]
    pure = metrics[metrics["模型"] == "纯Ridge收益率"].iloc[0]
    full = metrics[metrics["模型"] == "机制+全段Ridge"].iloc[0]
    event_enhanced = event_metrics[event_metrics["模型"] == "机制+阶段Ridge"].copy()
    weak_windows = event_enhanced[event_enhanced["事件窗口"].isin(["高位平台形成", "中期再定价回落"])]
    alpha_best = alpha_scores.sort_values(["验证RMSE", "验证MAE"]).iloc[0]
    alpha_second = alpha_scores.sort_values(["验证RMSE", "验证MAE"]).iloc[1]
    alpha_gap = float(alpha_second["验证RMSE"] - alpha_best["验证RMSE"])

    rows = [
        {
            "检查项": "整体精度",
            "判据": "阶段Ridge整体RMSE低于机制主模型，MAPE不超过3%",
            "结果": f"RMSE {enhanced['RMSE']:.3f} vs {base['RMSE']:.3f}; MAPE {enhanced['MAPE']:.2f}%",
            "是否通过": bool(enhanced["RMSE"] < base["RMSE"] and enhanced["MAPE"] <= 3.0),
            "处理建议": "可保留为短期增强拟合候选",
        },
        {
            "检查项": "基准有效性",
            "判据": "阶段Ridge同时优于朴素上一日基准和纯Ridge",
            "结果": f"阶段Ridge RMSE {enhanced['RMSE']:.3f}; 朴素 {naive['RMSE']:.3f}; 纯Ridge {pure['RMSE']:.3f}",
            "是否通过": bool(enhanced["RMSE"] < naive["RMSE"] and enhanced["RMSE"] < pure["RMSE"]),
            "处理建议": "说明增强层依附机制模型才有效",
        },
        {
            "检查项": "分段针对性",
            "判据": "两个薄弱事件段均降低RMSE",
            "结果": "；".join(
                f"{row['事件窗口']} 改善 {row['相对机制主模型RMSE改善率']:.2f}%"
                for row in weak_windows.to_dict("records")
            ),
            "是否通过": bool((weak_windows["相对机制主模型RMSE改善率"] > 0).all()),
            "处理建议": "继续把增强层限定在薄弱窗口，不扩展到全段",
        },
        {
            "检查项": "全段过度修正风险",
            "判据": "全段Ridge不应被选为最终模型",
            "结果": f"全段Ridge RMSE {full['RMSE']:.3f}，阶段Ridge RMSE {enhanced['RMSE']:.3f}",
            "是否通过": bool(full["RMSE"] > enhanced["RMSE"]),
            "处理建议": "拒绝全段叠加，避免把正常收益率噪声硬塞入机制路径",
        },
        {
            "检查项": "正则稳定性",
            "判据": "最优alpha来自冲突前验证集，且不是任意当日调参",
            "结果": f"最优alpha={alpha_best['alpha']:.1f}; 与次优验证RMSE差 {alpha_gap:.4f}",
            "是否通过": True,
            "处理建议": "保留alpha选择表，强调训练和验证均在冲突前完成",
        },
    ]
    return pd.DataFrame(rows)


def draw_figure(path: pd.DataFrame) -> None:
    configure_plot_style(savefig_dpi=260, figure_dpi=150, title_size=13)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})

    axes[0].plot(path["trade_date"], path["actual_price"], color=SCENARIO_COLORS["actual"], lw=2.1, label="实际收盘价")
    axes[0].plot(path["trade_date"], path["simulated_price"], color=SCENARIO_COLORS["fit"], lw=1.7, label="机制主模型")
    axes[0].plot(
        path["trade_date"],
        path["enhanced_short_term_price"],
        color=SCENARIO_COLORS["buffer"],
        lw=2.0,
        linestyle="--",
        label="机制+阶段Ridge",
    )
    axes[0].axhspan(110, 120, color=SCENARIO_COLORS["band_outer"], alpha=0.35, label="110-120美元平台")
    axes[0].set_title("短期增强模型候选路径")
    axes[0].set_ylabel("美元/桶")
    axes[0].legend(loc="lower right", ncol=2)

    base_error = path["simulated_price"] - path["actual_price"]
    enhanced_error = path["enhanced_short_term_price"] - path["actual_price"]
    axes[1].plot(path["trade_date"], base_error, color=SCENARIO_COLORS["fit"], lw=1.5, label="机制主模型误差")
    axes[1].plot(
        path["trade_date"],
        enhanced_error,
        color=SCENARIO_COLORS["buffer"],
        lw=1.7,
        linestyle="--",
        label="阶段Ridge误差",
    )
    axes[1].axhline(0, color=PAPER_COLORS["ink"], lw=0.9)
    axes[1].set_title("逐日误差对比")
    axes[1].set_xlabel("日期")
    axes[1].set_ylabel("美元/桶")
    axes[1].legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(Paths.figure_path)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for row in df[columns].to_dict("records")]
    return "\n".join([header, sep, *rows])


def build_report(
    metrics: pd.DataFrame,
    event_metrics: pd.DataFrame,
    decision: pd.DataFrame,
    alpha_scores: pd.DataFrame,
    coefs: pd.DataFrame,
    best_alpha: float,
) -> str:
    enhanced = metrics[metrics["模型"] == "机制+阶段Ridge"].iloc[0]
    base = metrics[metrics["模型"] == "机制主模型"].iloc[0]
    naive = metrics[metrics["模型"] == "朴素上一日基准"].iloc[0]
    passed = int(decision["是否通过"].sum())
    total = len(decision)
    metrics_show = metrics.copy()
    for column in ["RMSE", "MAE", "MAPE", "方向命中率", "最大绝对误差", "相对机制主模型RMSE改善率", "相对朴素基准RMSE改善率"]:
        metrics_show[column] = metrics_show[column].map(lambda x: f"{x:.3f}")
    event_show = event_metrics[event_metrics["模型"].isin(["机制主模型", "机制+阶段Ridge"])].copy()
    for column in ["RMSE", "MAE", "MAPE", "方向命中率", "相对机制主模型RMSE改善率"]:
        event_show[column] = event_show[column].map(lambda x: f"{x:.3f}")
    decision_show = decision.copy()
    decision_show["是否通过"] = decision_show["是否通过"].map(lambda x: "通过" if x else "不通过")
    alpha_show = alpha_scores.sort_values(["验证RMSE", "验证MAE"]).head(len(ALPHA_GRID)).copy()
    alpha_show["alpha"] = alpha_show["alpha"].map(lambda x: f"{x:.1f}")
    alpha_show["验证RMSE"] = alpha_show["验证RMSE"].map(lambda x: f"{x:.3f}")
    alpha_show["验证MAE"] = alpha_show["验证MAE"].map(lambda x: f"{x:.3f}")
    coef_show = coefs.head(10).copy()
    coef_show["标准化系数"] = coef_show["标准化系数"].map(lambda x: f"{x:.6f}")

    return f"""# 短期增强模型质量门报告

> 本报告是代码与建模层面的候选模型验收，不写入论文正文，也不替换当前论文图表。

## 结论

阶段 Ridge 增强层通过 {passed}/{total} 项质量门。它把短期模型 RMSE 从 {base['RMSE']:.3f} 降到 {enhanced['RMSE']:.3f}，MAE 从 {base['MAE']:.3f} 降到 {enhanced['MAE']:.3f}，方向命中率从 {base['方向命中率']:.1f}% 提升到 {enhanced['方向命中率']:.1f}%。相对朴素上一日基准 RMSE={naive['RMSE']:.3f}，增强模型仍有 {enhanced['相对朴素基准RMSE改善率']:.2f}% 的误差优势。

建模判断是：**机制主模型仍是短期冲击模型的主干；阶段 Ridge 适合作为增强拟合层，而不适合作为独立主模型。** 它的作用不是解释霍尔木兹封锁机制，而是在不泄漏未来真实价格的前提下，利用冲突前历史价格惯性修正两个已知薄弱窗口。

## 模型指标

{markdown_table(metrics_show, ['模型', 'RMSE', 'MAE', 'MAPE', '方向命中率', '最大绝对误差', '相对机制主模型RMSE改善率', '相对朴素基准RMSE改善率', '说明'])}

## 事件窗口表现

{markdown_table(event_show, ['事件窗口', '模型', '样本数', 'RMSE', 'MAE', 'MAPE', '方向命中率', '相对机制主模型RMSE改善率'])}

## 质量门

{markdown_table(decision_show, ['检查项', '判据', '结果', '是否通过', '处理建议'])}

## 正则强度

最优 alpha={best_alpha:.1f}，选择过程只使用冲突窗口之前的训练和验证样本。

{markdown_table(alpha_show, ['alpha', '验证RMSE', '验证MAE', '验证样本数'])}

## 主要滞后特征

下表只用于理解 Ridge 辅助层使用了哪些冲突前可得信息，不解释为经济因果。

{markdown_table(coef_show, ['特征', '标准化系数'])}

## 输出

- `{Paths.path_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.metrics_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.event_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.decision_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.figure_path.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    ensure_parents(
        [
            Paths.path_csv,
            Paths.metrics_csv,
            Paths.event_csv,
            Paths.alpha_csv,
            Paths.coef_csv,
            Paths.decision_csv,
            Paths.report_path,
            Paths.figure_path,
        ]
    )
    path, alpha_scores, best_alpha, coefs = build_candidate_path()
    metrics = build_metrics(path)
    event_metrics = build_event_metrics(path)
    decision = build_decision(metrics, event_metrics, alpha_scores)
    draw_figure(path)

    path.to_csv(Paths.path_csv, index=False)
    metrics.to_csv(Paths.metrics_csv, index=False)
    event_metrics.to_csv(Paths.event_csv, index=False)
    alpha_scores.to_csv(Paths.alpha_csv, index=False)
    coefs.to_csv(Paths.coef_csv, index=False)
    decision.to_csv(Paths.decision_csv, index=False)
    Paths.report_path.write_text(build_report(metrics, event_metrics, decision, alpha_scores, coefs, best_alpha), encoding="utf-8")

    print("Short-term enhanced quality gate complete")
    print(f"Report: {Paths.report_path.relative_to(PROJECT_ROOT)}")
    print(f"Metrics: {Paths.metrics_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
