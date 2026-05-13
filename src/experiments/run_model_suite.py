"""Aggregate model evidence into one experiment-management layer.

The suite does not recalibrate models. It reads existing reproducible artifacts
and creates comparable tables for model selection, paper writing, and defense.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.common.paths import PROJECT_ROOT, ensure_parents


CONFIG_PATH = PROJECT_ROOT / "config" / "experiments.yml"


@dataclass(frozen=True)
class ExperimentPaths:
    summary_csv: Path
    short_term_ranking_csv: Path
    long_term_summary_csv: Path
    report_path: Path


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(path: str) -> Path:
    return PROJECT_ROOT / path


def resolve_outputs(config: dict[str, Any]) -> ExperimentPaths:
    suite = config["experiment_suite"]
    return ExperimentPaths(
        summary_csv=resolve_project_path(suite["summary_csv"]),
        short_term_ranking_csv=resolve_project_path(suite["short_term_ranking_csv"]),
        long_term_summary_csv=resolve_project_path(suite["long_term_summary_csv"]),
        report_path=resolve_project_path(suite["report_path"]),
    )


def read_required_csv(config: dict[str, Any], key: str) -> pd.DataFrame:
    path = resolve_project_path(config["inputs"][key])
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment input: {path.relative_to(PROJECT_ROOT)}")
    return pd.read_csv(path)


def clean_metric(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def model_row(
    *,
    category: str,
    model: str,
    source: str,
    role: str,
    note: str,
    **metrics: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "模型类别": category,
        "模型": model,
        "证据来源": source,
        "角色定位": role,
        "说明": note,
        "RMSE": None,
        "MAE": None,
        "MAPE": None,
        "方向命中率": None,
        "最大绝对误差": None,
        "第60天价格": None,
        "第90天价格": None,
        "第120天价格": None,
        "第180天价格": None,
        "第180天价格_P05": None,
        "第180天价格_P50": None,
        "第180天价格_P95": None,
        "外推期最高价": None,
        "外推期均价": None,
        "突破120概率": None,
        "突破130概率": None,
        "二次跳涨风险": None,
        "质量评分": None,
    }
    for key, value in metrics.items():
        if key in row:
            row[key] = value
    return row


def collect_short_term_baselines(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = config["inputs"]["short_term_baselines"]
    df = read_required_csv(config, "short_term_baselines")
    rows: list[dict[str, Any]] = []
    for record in df.to_dict("records"):
        rows.append(
            model_row(
                category="短期拟合/预测",
                model=str(record["模型"]),
                source=source,
                role="短期误差对比",
                note=str(record.get("说明", "")),
                RMSE=clean_metric(record.get("RMSE")),
                MAE=clean_metric(record.get("MAE")),
                MAPE=clean_metric(record.get("MAPE")),
                方向命中率=clean_metric(record.get("方向命中率")),
                最大绝对误差=clean_metric(record.get("最大绝对误差")),
            )
        )
    return rows


def collect_short_term_ml(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = config["inputs"]["short_term_ml"]
    df = read_required_csv(config, "short_term_ml")
    rows: list[dict[str, Any]] = []
    for record in df.to_dict("records"):
        rows.append(
            model_row(
                category="短期机器学习辅助",
                model=str(record["模型"]),
                source=source,
                role="机器学习残差/收益率辅助",
                note=str(record.get("说明", "")),
                RMSE=clean_metric(record.get("RMSE")),
                MAE=clean_metric(record.get("MAE")),
                MAPE=clean_metric(record.get("MAPE")),
                方向命中率=clean_metric(record.get("方向命中率")),
                最大绝对误差=clean_metric(record.get("最大绝对误差")),
            )
        )
    return rows


def collect_quality_gate(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = config["inputs"]["short_term_quality_gate"]
    df = read_required_csv(config, "short_term_quality_gate")
    rows: list[dict[str, Any]] = []
    for record in df.to_dict("records"):
        rows.append(
            model_row(
                category="质量闸门",
                model=str(record["维度"]),
                source=source,
                role=str(record.get("结论", "")),
                note=f"{record.get('核心指标', '')}；{record.get('风险提示', '')}",
                质量评分=clean_metric(record.get("评分")),
            )
        )
    return rows


def collect_traditional_counterfactual(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = config["inputs"]["traditional_counterfactual"]
    df = read_required_csv(config, "traditional_counterfactual")
    rows: list[dict[str, Any]] = []
    for record in df.to_dict("records"):
        note = (
            f"线性反事实价格={record.get('线性需求反事实价格_美元每桶'):.3f}，"
            f"常弹性机械上界={record.get('常弹性机械上界价格_美元每桶'):.3f}"
        )
        rows.append(
            model_row(
                category="传统供需反事实",
                model=f"传统供需基准-{record['情景']}",
                source=source,
                role="反事实压力上界",
                note=note,
                外推期最高价=clean_metric(record.get("线性需求反事实价格_美元每桶")),
            )
        )
    return rows


def collect_long_term_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = config["inputs"]["long_term_scenarios"]
    df = read_required_csv(config, "long_term_scenarios")
    rows: list[dict[str, Any]] = []
    for record in df.to_dict("records"):
        rows.append(
            model_row(
                category="长期确定性情景",
                model=f"长期三情景-{record['情景']}",
                source=source,
                role="60-180天条件中心路径",
                note="长期无真实标签，作为条件情景外推，不按短期拟合 RMSE 排名。",
                第60天价格=clean_metric(record.get("第60天价格")),
                第90天价格=clean_metric(record.get("第90天价格")),
                第120天价格=clean_metric(record.get("第120天价格")),
                第180天价格=clean_metric(record.get("第180天价格")),
                外推期最高价=clean_metric(record.get("外推期最高价")),
                外推期均价=clean_metric(record.get("外推期均价")),
                二次跳涨风险=record.get("二次跳涨风险"),
            )
        )
    return rows


def probability(value: pd.Series) -> float:
    if value.empty:
        return float("nan")
    return float(value.astype(bool).mean() * 100)


def quantile(series: pd.Series, q: float) -> float:
    return float(series.astype(float).quantile(q))


def collect_monte_carlo(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = config["inputs"]["monte_carlo_samples"]
    df = read_required_csv(config, "monte_carlo_samples")
    return [
        model_row(
            category="长期概率模型",
            model="蒙特卡洛情景树",
            source=source,
            role="尾部风险与分位区间",
            note=f"样本数={len(df)}，用于补足长期中心线过平的问题。",
            第90天价格=clean_metric(df["第90天价格"].mean()),
            第120天价格=clean_metric(df["第120天价格"].mean()),
            第180天价格=clean_metric(df["第180天价格"].mean()),
            第180天价格_P05=quantile(df["第180天价格"], 0.05),
            第180天价格_P50=quantile(df["第180天价格"], 0.50),
            第180天价格_P95=quantile(df["第180天价格"], 0.95),
            外推期最高价=clean_metric(df["外推期最高价"].mean()),
            外推期均价=clean_metric(df["外推期均价"].mean()),
            突破120概率=probability(df["是否突破120"]),
            突破130概率=probability(df["是否突破130"]),
            二次跳涨风险=f"高缺口概率 {probability(df['是否高缺口']):.1f}%",
        )
    ]


def collect_state_transition(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = config["inputs"]["state_transition_samples"]
    df = read_required_csv(config, "state_transition_samples")
    final_state_share = df["最终状态"].value_counts(normalize=True).mul(100).round(1).to_dict()
    return [
        model_row(
            category="长期概率模型",
            model="马尔可夫状态转移情景树",
            source=source,
            role="状态不确定性与概率区间",
            note=f"样本数={len(df)}，最终状态占比={final_state_share}",
            第90天价格=clean_metric(df["第90天价格"].mean()),
            第120天价格=clean_metric(df["第120天价格"].mean()),
            第180天价格=clean_metric(df["第180天价格"].mean()),
            第180天价格_P05=quantile(df["第180天价格"], 0.05),
            第180天价格_P50=quantile(df["第180天价格"], 0.50),
            第180天价格_P95=quantile(df["第180天价格"], 0.95),
            外推期最高价=clean_metric(df["外推期最高价"].mean()),
            外推期均价=clean_metric(df["外推期均价"].mean()),
            突破120概率=probability(df["突破120"]),
            突破130概率=probability(df["突破130"]),
            二次跳涨风险="由状态转移路径和突破概率表达",
        )
    ]


def build_summary(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(collect_short_term_baselines(config))
    rows.extend(collect_short_term_ml(config))
    rows.extend(collect_quality_gate(config))
    rows.extend(collect_traditional_counterfactual(config))
    rows.extend(collect_long_term_scenarios(config))
    rows.extend(collect_monte_carlo(config))
    rows.extend(collect_state_transition(config))
    return pd.DataFrame(rows)


def build_short_term_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    ranking = summary[summary["RMSE"].notna()].copy()
    reference_mask = (ranking["模型类别"] == "短期机器学习辅助") & ranking["模型"].isin(["机制递推主模型", "朴素上一日基准"])
    ranking = ranking.loc[~reference_mask].copy()
    ranking = ranking.sort_values(["RMSE", "MAE"], ascending=True).reset_index(drop=True)
    ranking.insert(0, "排名", np.arange(1, len(ranking) + 1))
    best_rmse = float(ranking["RMSE"].iloc[0]) if not ranking.empty else float("nan")
    ranking["相对最优RMSE差值"] = ranking["RMSE"] - best_rmse
    return ranking


def build_long_term_summary(summary: pd.DataFrame) -> pd.DataFrame:
    mask = summary["模型类别"].isin(["长期确定性情景", "长期概率模型"])
    return summary.loc[mask].copy().reset_index(drop=True)


def fmt_float(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return str(value)
    return f"{float(value):.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    data = df.loc[:, columns].copy()
    if max_rows is not None:
        data = data.head(max_rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in data.to_dict("records"):
        lines.append("| " + " | ".join(fmt_float(record[col]) for col in columns) + " |")
    return "\n".join(lines)


def build_report(
    config: dict[str, Any],
    summary: pd.DataFrame,
    short_ranking: pd.DataFrame,
    long_summary: pd.DataFrame,
) -> str:
    suite = config["experiment_suite"]
    rules = config["rules"]
    best_short = short_ranking.iloc[0] if not short_ranking.empty else None
    best_short_text = (
        f"当前短期误差最优模型为 **{best_short['模型']}**，RMSE={best_short['RMSE']:.3f}。"
        if best_short is not None
        else "当前没有可排序的短期误差指标。"
    )
    quality_rows = summary[summary["模型类别"] == "质量闸门"].copy()
    avg_quality = quality_rows["质量评分"].dropna().mean()

    return f"""# 模型实验管理报告

## 1. 运行结论

本报告由 `src.experiments.run_model_suite` 自动汇总生成，目标是把短期拟合、机器学习辅助、长期情景、传统供需反事实和质量闸门放到同一张实验台账中。

{best_short_text}

短期模型可以用 RMSE、MAE、MAPE、方向命中率等指标横向比较；长期模型没有 60--180 天真实标签，不能把情景中心线包装成拟合精度，因此这里只汇总中心路径、分位区间和尾部概率。

## 2. 短期模型排序

{markdown_table(short_ranking, ["排名", "模型类别", "模型", "RMSE", "MAE", "MAPE", "方向命中率", "相对最优RMSE差值"], max_rows=10)}

## 3. 长期情景与概率摘要

{markdown_table(long_summary, ["模型类别", "模型", "第90天价格", "第120天价格", "第180天价格", "第180天价格_P05", "第180天价格_P50", "第180天价格_P95", "突破120概率", "突破130概率"])}

## 4. 质量闸门

质量闸门平均评分为 {avg_quality:.2f}。这些评分不是新的拟合指标，而是对拟合精度、基准优势、机器学习辅助、稳健性、可解释性等维度的提交前检查。

{markdown_table(quality_rows, ["模型", "角色定位", "质量评分", "说明"])}

## 5. 输出文件

- `{suite["summary_csv"]}`
- `{suite["short_term_ranking_csv"]}`
- `{suite["long_term_summary_csv"]}`
- `{suite["report_path"]}`

## 6. 使用边界

- 短期窗口有真实价格，因此允许进行误差排序。
- 长期窗口是条件情景和概率外推，{rules["long_term_note"]}
- 该实验层只汇总现有产物，不重新校准参数，不改变论文结论。
"""


def write_outputs(
    paths: ExperimentPaths,
    summary: pd.DataFrame,
    short_ranking: pd.DataFrame,
    long_summary: pd.DataFrame,
    report: str,
) -> None:
    ensure_parents([paths.summary_csv, paths.short_term_ranking_csv, paths.long_term_summary_csv, paths.report_path])
    summary.to_csv(paths.summary_csv, index=False)
    short_ranking.to_csv(paths.short_term_ranking_csv, index=False)
    long_summary.to_csv(paths.long_term_summary_csv, index=False)
    paths.report_path.write_text(report, encoding="utf-8")


def main() -> None:
    config = load_config()
    paths = resolve_outputs(config)
    summary = build_summary(config)
    short_ranking = build_short_term_ranking(summary)
    long_summary = build_long_term_summary(summary)
    report = build_report(config, summary, short_ranking, long_summary)
    write_outputs(paths, summary, short_ranking, long_summary, report)

    print("Model experiment suite complete")
    print(f"Summary: {paths.summary_csv.relative_to(PROJECT_ROOT)}")
    print(f"Short-term ranking: {paths.short_term_ranking_csv.relative_to(PROJECT_ROOT)}")
    print(f"Long-term summary: {paths.long_term_summary_csv.relative_to(PROJECT_ROOT)}")
    print(f"Report: {paths.report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
