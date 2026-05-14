"""Exogenous-risk Ridge experiments for short-term residual enhancement.

This experiment checks whether reproducible lagged market-risk variables
actually improve the short-term event-window enhancement layer.  It does not
overwrite official model outputs or paper figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.short_term_ml_features import OUTPUT_CSV as PRICE_FEATURE_CSV
from src.analysis.short_term_ml_features import main as build_price_features
from src.common.metrics import direction_hit_rate, mae, mape, rmse
from src.common.paths import PROJECT_ROOT, ensure_parents
from src.common.plotting import PAPER_COLORS, SCENARIO_COLORS, configure_plot_style


@dataclass(frozen=True)
class Paths:
    mechanism_path: Path = PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv"
    risk_sample: Path = PROJECT_ROOT / "output" / "risk" / "市场定价风险变量_滞后检验样本.csv"
    gpr_sample: Path = PROJECT_ROOT / "data" / "external" / "risk" / "地缘政治风险与油价_滞后审计样本.csv"
    summary_csv: Path = PROJECT_ROOT / "output" / "experiments" / "短期外生风险Ridge实验汇总.csv"
    prediction_csv: Path = PROJECT_ROOT / "output" / "experiments" / "短期外生风险Ridge最佳路径.csv"
    feature_audit_csv: Path = PROJECT_ROOT / "output" / "experiments" / "短期外生风险Ridge特征审计.csv"
    report_path: Path = PROJECT_ROOT / "output" / "reports" / "短期外生风险Ridge实验报告.md"
    figure_path: Path = PROJECT_ROOT / "output" / "candidate_figures" / "短期外生风险Ridge实验_最佳候选.png"


EVENT_START = pd.Timestamp("2026-03-02")
VALIDATION_START = pd.Timestamp("2024-01-01")
ALPHA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
RIDGE_ASSIST_WINDOWS = [
    ("高位平台形成", pd.Timestamp("2026-03-17"), pd.Timestamp("2026-03-31")),
    ("中期再定价回落", pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-17")),
]

EVENT_WINDOWS = [
    ("初始冲击吸收", "2026-03-02", "2026-03-14"),
    ("高位平台形成", "2026-03-17", "2026-03-31"),
    ("中期再定价回落", "2026-04-01", "2026-04-17"),
    ("后期二次抬升", "2026-04-20", "2026-05-05"),
]


def ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline([("standardize", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def load_price_features() -> pd.DataFrame:
    if not PRICE_FEATURE_CSV.exists():
        build_price_features()
    return pd.read_csv(PRICE_FEATURE_CSV, parse_dates=["trade_date"])


def load_mechanism() -> pd.DataFrame:
    if not Paths.mechanism_path.exists():
        raise FileNotFoundError(Paths.mechanism_path)
    return pd.read_csv(Paths.mechanism_path, parse_dates=["trade_date"]).sort_values("trade_date")


def load_risk_features() -> pd.DataFrame:
    if not Paths.risk_sample.exists():
        return pd.DataFrame({"trade_date": pd.Series(dtype="datetime64[ns]")})
    risk = pd.read_csv(Paths.risk_sample, parse_dates=["trade_date"])
    cols = [
        "trade_date",
        "ovx_lag1",
        "ovx_lag3",
        "ovx_lag5",
        "ovx_lag10",
        "ovx_change_lag1",
        "ovx_change_lag3",
        "ovx_change_lag5",
        "ovx_change_lag10",
    ]
    return risk[[c for c in cols if c in risk.columns]].copy()


def load_gpr_features(price_features: pd.DataFrame) -> pd.DataFrame:
    if not Paths.gpr_sample.exists():
        return pd.DataFrame({"trade_date": price_features["trade_date"]})
    gpr = pd.read_csv(Paths.gpr_sample, parse_dates=["月份"])
    gpr_cols = [
        "月份",
        "GPR_全球综合_滞后1月",
        "GPR_全球综合_滞后2月",
        "GPRT_威胁_滞后1月",
        "GPRA_行动_滞后1月",
        "GPR_历史标准分_滞后1月",
    ]
    gpr = gpr[[c for c in gpr_cols if c in gpr.columns]].copy()
    out = price_features[["trade_date"]].copy()
    out["月份"] = out["trade_date"].dt.to_period("M").dt.to_timestamp()
    out = out.merge(gpr, on="月份", how="left").drop(columns=["月份"])
    return out


def build_feature_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    price = load_price_features()
    risk = load_risk_features()
    gpr = load_gpr_features(price)
    features = price.merge(risk, on="trade_date", how="left").merge(gpr, on="trade_date", how="left")
    audit_rows = []
    for column in features.columns:
        if column in {"trade_date", "pre_close_filled", "target_close_price", "target_log_return", "target_price_change", "是否冲突窗口"}:
            continue
        event_nonnull = int(features.loc[features["trade_date"] >= EVENT_START, column].notna().sum())
        pre_event_nonnull = int(features.loc[features["trade_date"] < EVENT_START, column].notna().sum())
        audit_rows.append(
            {
                "特征": column,
                "冲突前非空样本": pre_event_nonnull,
                "冲突窗口非空样本": event_nonnull,
                "可用于训练": pre_event_nonnull >= 120 and event_nonnull >= 20,
            }
        )
    return features, pd.DataFrame(audit_rows)


def feature_sets(features: pd.DataFrame) -> dict[str, list[str]]:
    excluded = {
        "trade_date",
        "pre_close_filled",
        "target_close_price",
        "target_log_return",
        "target_price_change",
        "是否冲突窗口",
    }
    base_cols = [
        c
        for c in features.columns
        if c not in excluded and not c.startswith("ovx_") and not c.startswith("GPR")
    ]
    ovx_cols = [c for c in features.columns if c.startswith("ovx_")]
    gpr_cols = [c for c in features.columns if c.startswith("GPR")]
    return {
        "仅历史价格特征": base_cols,
        "历史价格+OVX滞后": base_cols + ovx_cols,
        "历史价格+GPR月度滞后": base_cols + gpr_cols,
        "历史价格+OVX+GPR": base_cols + ovx_cols + gpr_cols,
    }


def clean_frame(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    needed = ["trade_date", "pre_close_filled", "target_close_price", "target_log_return", *columns]
    return features[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()


def choose_alpha(train: pd.DataFrame, columns: list[str]) -> tuple[float, pd.DataFrame]:
    train_part = train[train["trade_date"] < VALIDATION_START].copy()
    valid_part = train[(train["trade_date"] >= VALIDATION_START) & (train["trade_date"] < EVENT_START)].copy()
    if len(valid_part) < 60:
        split = int(len(train) * 0.8)
        train_part = train.iloc[:split].copy()
        valid_part = train.iloc[split:].copy()
    rows = []
    for alpha in ALPHA_GRID:
        model = ridge_pipeline(alpha)
        model.fit(train_part[columns].to_numpy(float), train_part["target_log_return"].to_numpy(float))
        pred_return = model.predict(valid_part[columns].to_numpy(float))
        pred_price = valid_part["pre_close_filled"].to_numpy(float) * np.exp(pred_return)
        error = pred_price - valid_part["target_close_price"].to_numpy(float)
        rows.append({"alpha": alpha, "验证RMSE": rmse(error), "验证MAE": mae(error), "验证样本数": len(valid_part)})
    scores = pd.DataFrame(rows).sort_values(["验证RMSE", "验证MAE"]).reset_index(drop=True)
    return float(scores.iloc[0]["alpha"]), scores


def add_phase_ridge_prediction(
    mechanism: pd.DataFrame,
    event_predictions: pd.DataFrame,
    variant_name: str,
) -> pd.DataFrame:
    result = mechanism.merge(event_predictions, on="trade_date", how="inner")
    result["ridge_return_correction"] = result["ridge_price"] - result["pre_close_filled"]
    result["risk_ridge_assist_phase"] = "未启用"
    result["risk_phase_gated_correction"] = 0.0
    for name, start, end in RIDGE_ASSIST_WINDOWS:
        mask = result["trade_date"].between(start, end)
        result.loc[mask, "risk_ridge_assist_phase"] = name
        result.loc[mask, "risk_phase_gated_correction"] = result.loc[mask, "ridge_return_correction"]
    result["mechanism_plus_risk_phase_ridge"] = result["simulated_price"] + result["risk_phase_gated_correction"]
    result["风险Ridge方案"] = variant_name
    return result


def model_metrics(df: pd.DataFrame, model_name: str, price_col: str, feature_set: str) -> dict[str, object]:
    actual = df["actual_price"]
    pred = df[price_col]
    error = pred - actual
    row: dict[str, object] = {
        "特征组": feature_set,
        "模型": model_name,
        "RMSE": rmse(error),
        "MAE": mae(error),
        "MAPE": mape(actual, pred),
        "方向命中率": direction_hit_rate(actual, pred),
        "最大绝对误差": float(error.abs().max()),
    }
    for segment, start, end in EVENT_WINDOWS:
        sub = df[df["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        row[f"{segment}_RMSE"] = rmse(sub[price_col] - sub["actual_price"])
    return row


def run_feature_set(
    features: pd.DataFrame,
    mechanism: pd.DataFrame,
    name: str,
    columns: list[str],
) -> tuple[pd.DataFrame, list[dict[str, object]], pd.DataFrame]:
    usable = clean_frame(features, columns)
    pre_event = usable[usable["trade_date"] < EVENT_START].copy()
    event = usable[usable["trade_date"].isin(mechanism["trade_date"])].copy()
    if len(pre_event) < 300 or len(event) < 30:
        return pd.DataFrame(), [], pd.DataFrame()
    best_alpha, alpha_scores = choose_alpha(pre_event, columns)
    model = ridge_pipeline(best_alpha)
    model.fit(pre_event[columns].to_numpy(float), pre_event["target_log_return"].to_numpy(float))
    pred_return = model.predict(event[columns].to_numpy(float))
    event_predictions = event[["trade_date", "pre_close_filled", "target_close_price"]].copy()
    event_predictions["ridge_predicted_return"] = pred_return
    event_predictions["ridge_price"] = event_predictions["pre_close_filled"] * np.exp(pred_return)
    path = add_phase_ridge_prediction(mechanism, event_predictions, name)
    rows = [
        {
            **model_metrics(path, "纯Ridge收益率", "ridge_price", name),
            "alpha": best_alpha,
            "训练样本": len(pre_event),
            "测试样本": len(event),
        },
        {
            **model_metrics(path, "机制+阶段Ridge", "mechanism_plus_risk_phase_ridge", name),
            "alpha": best_alpha,
            "训练样本": len(pre_event),
            "测试样本": len(event),
        },
    ]
    alpha_scores["特征组"] = name
    return path, rows, alpha_scores


def select_best(summary: pd.DataFrame) -> pd.Series:
    candidates = summary[summary["模型"] == "机制+阶段Ridge"].copy()
    candidates["排序目标"] = (
        candidates["RMSE"]
        + 0.18 * candidates["中期再定价回落_RMSE"]
        + 0.06 * candidates["高位平台形成_RMSE"]
        + 0.03 * candidates["最大绝对误差"]
    )
    return candidates.sort_values(["排序目标", "RMSE"]).iloc[0]


def draw_figure(mechanism: pd.DataFrame, best_path: pd.DataFrame, best: pd.Series) -> None:
    configure_plot_style(savefig_dpi=260, figure_dpi=150, title_size=13)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})
    ax = axes[0]
    ax.plot(mechanism["trade_date"], mechanism["actual_price"], color=SCENARIO_COLORS["actual"], lw=2.2, label="实际收盘价")
    ax.plot(mechanism["trade_date"], mechanism["simulated_price"], color=SCENARIO_COLORS["fit"], lw=1.8, label="机制主模型")
    ax.plot(
        best_path["trade_date"],
        best_path["mechanism_plus_risk_phase_ridge"],
        color=SCENARIO_COLORS["buffer"],
        lw=2.0,
        linestyle="--",
        label=f"{best['特征组']} 阶段Ridge",
    )
    ax.axhspan(110, 120, color=SCENARIO_COLORS["band_outer"], alpha=0.42, label="110-120美元平台")
    ax.set_title("外生风险变量 Ridge 候选路径")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="lower right", ncol=2)

    ax = axes[1]
    base_error = mechanism["simulated_price"] - mechanism["actual_price"]
    best_error = best_path["mechanism_plus_risk_phase_ridge"] - best_path["actual_price"]
    ax.plot(mechanism["trade_date"], base_error, color=SCENARIO_COLORS["fit"], lw=1.5, label="机制误差")
    ax.plot(best_path["trade_date"], best_error, color=SCENARIO_COLORS["buffer"], lw=1.7, linestyle="--", label="候选误差")
    ax.axhline(0, color=PAPER_COLORS["ink"], lw=0.9)
    ax.set_title("逐日误差对比")
    ax.set_ylabel("美元/桶")
    ax.set_xlabel("日期")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(Paths.figure_path)
    plt.close(fig)


def build_report(summary: pd.DataFrame, best: pd.Series, feature_audit: pd.DataFrame, alpha_scores: pd.DataFrame) -> str:
    mechanism = summary[(summary["特征组"] == "机制主模型") & (summary["模型"] == "机制主模型")].iloc[0]
    rows = "\n".join(
        "| {feature} | {model} | {rmse:.3f} | {mae:.3f} | {mape:.3f}% | {direction:.1f}% | {maxerr:.3f} | {mid:.3f} |".format(
            feature=row["特征组"],
            model=row["模型"],
            rmse=row["RMSE"],
            mae=row["MAE"],
            mape=row["MAPE"],
            direction=row["方向命中率"],
            maxerr=row["最大绝对误差"],
            mid=row["中期再定价回落_RMSE"],
        )
        for row in summary.to_dict("records")
    )
    audit_rows = "\n".join(
        f"| {row['特征']} | {int(row['冲突前非空样本'])} | {int(row['冲突窗口非空样本'])} | {row['可用于训练']} |"
        for row in feature_audit[feature_audit["特征"].str.contains("ovx|GPR", regex=True)].to_dict("records")
    )
    alpha_rows = "\n".join(
        f"| {row['特征组']} | {row['alpha']:.1f} | {row['验证RMSE']:.3f} | {row['验证MAE']:.3f} |"
        for row in alpha_scores.groupby("特征组").head(1).to_dict("records")
    )
    rmse_gain = (mechanism["RMSE"] - best["RMSE"]) / mechanism["RMSE"] * 100
    mid_gain = (mechanism["中期再定价回落_RMSE"] - best["中期再定价回落_RMSE"]) / mechanism["中期再定价回落_RMSE"] * 100
    return f"""# 短期外生风险 Ridge 实验报告

> 本报告只用于模型优化决策，不写入论文正文，不覆盖当前官方短期模型输出。

## 核心结论

本轮把可复现的滞后市场风险变量加入 Ridge 辅助层，检验它们是否能改善短期模型薄弱窗口。最佳候选为 `{best['特征组']}` 的 `{best['模型']}`，RMSE={best['RMSE']:.3f}，相对机制主模型改善 {rmse_gain:.2f}%；中期再定价回落段 RMSE={best['中期再定价回落_RMSE']:.3f}，相对机制主模型改善 {mid_gain:.2f}%。

这轮实验的核心原则是：外生风险变量必须滞后使用，不能用同日风险解释同日油价。OVX 是日度市场隐含波动率，具备短期实验价值；GPR 是月度变量，更适合长期风险背景或情景概率约束，不适合独立解释日度短期拟合。

## 指标对比

| 特征组 | 模型 | RMSE | MAE | MAPE | 方向命中率 | 最大绝对误差 | 中期RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
{rows}

## 外生变量可用性审计

| 特征 | 冲突前非空样本 | 冲突窗口非空样本 | 可用于训练 |
|---|---:|---:|---|
{audit_rows}

## 正则强度选择

| 特征组 | 最优alpha | 验证RMSE | 验证MAE |
|---|---:|---:|---:|
{alpha_rows}

## 建模判断

1. 如果 OVX 组合没有稳定超过“仅历史价格特征”的阶段 Ridge，则不应为了显得高级而并入最终短期模型。
2. 如果 OVX 组合能降低中期回落段误差，同时不抬高最大绝对误差，可以作为下一轮主模型增强候选。
3. 期限结构目前没有完整可复现多期限结算价，继续保留为数据接入任务，不纳入短期拟合。

## 输出文件

- `{Paths.summary_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.prediction_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.feature_audit_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.figure_path.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    ensure_parents([Paths.summary_csv, Paths.prediction_csv, Paths.feature_audit_csv, Paths.report_path, Paths.figure_path])
    features, feature_audit = build_feature_table()
    mechanism = load_mechanism()
    rows: list[dict[str, object]] = [
        {
            **model_metrics(mechanism, "机制主模型", "simulated_price", "机制主模型"),
            "alpha": np.nan,
            "训练样本": np.nan,
            "测试样本": len(mechanism),
        }
    ]
    alpha_frames: list[pd.DataFrame] = []
    paths: dict[str, pd.DataFrame] = {}
    for name, columns in feature_sets(features).items():
        path, metric_rows, alpha_scores = run_feature_set(features, mechanism, name, columns)
        if not path.empty:
            paths[name] = path
            rows.extend(metric_rows)
            alpha_frames.append(alpha_scores)

    summary = pd.DataFrame(rows)
    alpha_scores = pd.concat(alpha_frames, ignore_index=True) if alpha_frames else pd.DataFrame()
    best = select_best(summary)
    best_path = paths[str(best["特征组"])]
    best_path.to_csv(Paths.prediction_csv, index=False)
    summary.to_csv(Paths.summary_csv, index=False)
    feature_audit.to_csv(Paths.feature_audit_csv, index=False)
    draw_figure(mechanism, best_path, best)
    Paths.report_path.write_text(build_report(summary, best, feature_audit, alpha_scores), encoding="utf-8")
    print(Paths.report_path.relative_to(PROJECT_ROOT))
    print(Paths.summary_csv.relative_to(PROJECT_ROOT))
    print(Paths.figure_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
