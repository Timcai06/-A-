"""Literature-inspired challenger for the short-term mechanism model.

The current mechanism model is treated as the champion.  This module tests a
separate challenger with three literature-backed structural refinements:

1. inventory/buffer relief depends on buffer coverage and remaining inventory;
2. SPR affects price through both physical barrels and a credibility discount;
3. price adjustment is asymmetric between upward shock repricing and downward
   expectation repair.

The challenger is regularized against the champion so it is not selected for a
tiny in-sample RMSE gain that comes from excessive structural freedom.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.short_term_mechanism_optimization_audit import (
    EVENT_WINDOWS,
    load_best_objects,
    load_current_path,
    load_event_window,
    model_metrics,
)
from src.common.metrics import mae, rmse
from src.common.paths import PROJECT_ROOT, ensure_parents
from src.common.plotting import PAPER_COLORS, SCENARIO_COLORS, configure_plot_style
from src.models import dynamic_short_term as dynamic


@dataclass(frozen=True)
class Paths:
    challenger_path_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期文献增强Challenger路径.csv"
    candidate_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期文献增强Challenger候选.csv"
    metrics_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期文献增强Challenger指标.csv"
    event_csv: Path = PROJECT_ROOT / "output" / "calibration" / "短期文献增强Challenger事件窗口.csv"
    report_path: Path = PROJECT_ROOT / "output" / "reports" / "短期文献增强Challenger报告.md"
    figure_path: Path = PROJECT_ROOT / "output" / "candidate_figures" / "短期文献增强Challenger对比.png"


@dataclass(frozen=True)
class ChallengerConfig:
    upward_speed_multiplier: float = 1.0
    downward_speed_multiplier: float = 1.0
    inventory_relief_strength: float = 0.0
    spr_credibility_strength: float = 0.0
    coverage_center: float = 0.55
    credibility_ramp_days: int = 10


def activation(value: float, center: float, slope: float) -> float:
    bounded = float(np.clip(value, 0.0, 1.5))
    return float(1.0 / (1.0 + np.exp(-slope * (bounded - center))))


def simulate_challenger(
    event_df: pd.DataFrame,
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
    config: ChallengerConfig,
    time_axis: str = "calendar",
) -> pd.DataFrame:
    first_date = event_df["trade_date"].min()
    base_price = float(event_df.iloc[0]["pre_close"])
    previous_price = base_price
    previous_fear_excess = assumptions.fear_initial
    previous_buffer_coverage_ratio = 0.0
    inventory_remaining = assumptions.commercial_inventory
    gap_closure_day: int | None = None
    rows: list[dict[str, Any]] = []

    for step_index, actual_row in enumerate(event_df.itertuples(index=False)):
        trade_date = actual_row.trade_date
        if time_axis == "calendar":
            day_index = int((trade_date - first_date).days)
        elif time_axis == "trading_day":
            day_index = step_index
        else:
            raise ValueError(f"Unsupported time_axis: {time_axis}")

        elasticity = dynamic.interpolate_elasticity(day_index, assumptions)
        price_ratio = max(previous_price / base_price, 0.1)
        price_adjusted_demand = assumptions.base_demand * (price_ratio**elasticity)
        demand_decline = dynamic.ramp(day_index, 0, assumptions.demand_decline_ramp_days, assumptions.observed_demand_decline)
        effective_demand = max(price_adjusted_demand - demand_decline, assumptions.base_demand * 0.70)

        spr_release = dynamic.ramp(day_index, assumptions.spr_delay_days, assumptions.spr_ramp_days, assumptions.spr_max_release)
        route_supply = dynamic.ramp(day_index, assumptions.route_start_day, assumptions.route_ramp_days, assumptions.route_max_capacity)
        gross_shortage = max(effective_demand - (assumptions.base_supply - assumptions.supply_interruption), 0.0)
        supply_without_inventory = assumptions.base_supply - assumptions.supply_interruption + spr_release + route_supply
        raw_gap = max(effective_demand - supply_without_inventory, 0.0)

        inventory_buffer = min(raw_gap * behavior.inventory_response, assumptions.inventory_daily_cap, inventory_remaining)
        inventory_remaining -= inventory_buffer
        effective_supply = supply_without_inventory + inventory_buffer
        residual_gap = max(effective_demand - effective_supply, 0.0)
        total_buffer_supply = spr_release + route_supply + inventory_buffer
        buffer_coverage_ratio = float(np.clip(total_buffer_supply / max(gross_shortage, 1.0), 0.0, 1.5))
        coverage_momentum = buffer_coverage_ratio - previous_buffer_coverage_ratio
        if gap_closure_day is None and residual_gap <= assumptions.base_demand * 0.005:
            gap_closure_day = day_index

        fear_excess = assumptions.fear_initial * np.exp(-assumptions.fear_decay * day_index)
        shortage_pressure = (
            base_price
            * behavior.pressure_scale
            * (residual_gap / assumptions.base_demand)
            / max(abs(elasticity), 0.01)
        )
        blockade_risk_premium = (
            base_price
            * behavior.risk_weight
            * (assumptions.supply_interruption / assumptions.base_demand)
            * (1 - np.exp(-day_index / 7))
            * np.exp(-dynamic.BLOCKADE_RISK_DECAY * day_index)
        )
        uncertainty_premium = base_price * behavior.uncertainty_floor * (1 - np.exp(-day_index / 18))
        panic_premium = base_price * 0.45 * fear_excess

        base_buffer_discount = dynamic.buffer_confirmation_discount(
            day_index,
            gap_closure_day,
            base_price,
            behavior,
            buffer_coverage_ratio,
        )
        base_relief_discount = dynamic.expectation_relief_discount(
            day_index,
            base_price,
            behavior,
            buffer_coverage_ratio,
            coverage_momentum,
        )

        inventory_remaining_ratio = float(np.clip(inventory_remaining / max(assumptions.commercial_inventory, 1.0), 0.0, 1.0))
        buffer_confidence = activation(buffer_coverage_ratio, center=config.coverage_center, slope=8.0)
        inventory_relief_discount = (
            base_price
            * config.inventory_relief_strength
            * buffer_confidence
            * inventory_remaining_ratio
            * (1 - np.exp(-day_index / 10))
        )
        spr_credibility = (
            min(spr_release / max(assumptions.spr_max_release, 1.0), 1.0)
            * (1 - np.exp(-max(day_index - assumptions.spr_delay_days + 1, 0) / max(config.credibility_ramp_days, 1)))
        )
        spr_credibility_discount = (
            base_price
            * config.spr_credibility_strength
            * spr_credibility
            * (assumptions.supply_interruption / assumptions.base_demand)
        )

        target_price = (
            base_price
            + shortage_pressure
            + blockade_risk_premium
            + uncertainty_premium
            + panic_premium
            - base_buffer_discount
            - base_relief_discount
            - inventory_relief_discount
            - spr_credibility_discount
        )

        target_gap = target_price - previous_price
        speed = behavior.adjustment_speed
        if target_gap >= 0:
            speed = min(0.60, speed * config.upward_speed_multiplier)
        else:
            speed = max(0.12, speed * config.downward_speed_multiplier)
        simulated_price = previous_price + speed * target_gap
        simulated_price += 2.5 * (fear_excess - previous_fear_excess)
        simulated_price = float(np.clip(simulated_price, base_price * 0.75, 180.0))

        rows.append(
            {
                "day_index": day_index,
                "step_index": step_index,
                "time_axis": time_axis,
                "trade_date": trade_date,
                "actual_price": float(actual_row.close_price),
                "simulated_price": simulated_price,
                "effective_supply": effective_supply,
                "effective_demand": effective_demand,
                "gross_shortage": gross_shortage,
                "supply_gap": residual_gap,
                "spr_release": spr_release,
                "route_supply": route_supply,
                "inventory_buffer": inventory_buffer,
                "total_buffer_supply": total_buffer_supply,
                "buffer_coverage_ratio": buffer_coverage_ratio,
                "buffer_coverage_momentum": coverage_momentum,
                "inventory_remaining": inventory_remaining,
                "inventory_remaining_ratio": inventory_remaining_ratio,
                "demand_decline": demand_decline,
                "demand_elasticity": elasticity,
                "fear_factor": 1 + fear_excess,
                "shortage_pressure": shortage_pressure,
                "blockade_risk_premium": blockade_risk_premium,
                "uncertainty_premium": uncertainty_premium,
                "panic_premium": panic_premium,
                "buffer_confirmation_discount": base_buffer_discount,
                "expectation_relief_discount": base_relief_discount,
                "inventory_relief_discount": inventory_relief_discount,
                "spr_credibility_discount": spr_credibility_discount,
                "spr_credibility": spr_credibility,
                "asymmetric_adjustment_speed": speed,
                "literature_challenger_note": "库存覆盖/SPR可信度/非对称调价 Challenger",
            }
        )
        previous_price = simulated_price
        previous_fear_excess = fear_excess
        previous_buffer_coverage_ratio = buffer_coverage_ratio

    return pd.DataFrame(rows)


def regularization_penalty(config: ChallengerConfig) -> float:
    return (
        0.35 * abs(config.upward_speed_multiplier - 1.0)
        + 0.50 * abs(config.downward_speed_multiplier - 1.0)
        + 3.0 * config.inventory_relief_strength
        + 4.0 * config.spr_credibility_strength
        + 0.25 * abs(config.coverage_center - 0.55)
    )


def candidate_score(metrics: dict[str, float], config: ChallengerConfig) -> float:
    return (
        metrics["RMSE"]
        + 0.14 * metrics["高位平台形成_RMSE"]
        + 0.18 * metrics["中期再定价回落_RMSE"]
        + 0.10 * metrics["后期二次抬升_RMSE"]
        + 0.12 * abs(metrics["峰值误差"])
        + 0.18 * abs(metrics["末日误差"])
        + regularization_penalty(config)
    )


def event_metrics(path: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, start, end in EVENT_WINDOWS:
        sub = path[path["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
        error = sub["simulated_price"] - sub["actual_price"]
        rows.append(
            {
                "模型": model_name,
                "事件窗口": name,
                "样本数": len(sub),
                "RMSE": rmse(error),
                "MAE": mae(error),
                "平均误差": float(error.mean()),
                "最大绝对误差": float(error.abs().max()),
            }
        )
    return pd.DataFrame(rows)


def run_challenger_search(
    event_df: pd.DataFrame,
    assumptions: dynamic.PhysicalAssumptions,
    behavior: dynamic.BehavioralParameters,
) -> tuple[pd.DataFrame, pd.DataFrame, ChallengerConfig]:
    rows: list[dict[str, Any]] = []
    paths: dict[int, pd.DataFrame] = {}
    candidate_id = 0
    for upward in [0.95, 1.00, 1.08]:
        for downward in [0.70, 0.82, 0.92, 1.00]:
            for inventory_strength in [0.00, 0.015, 0.030, 0.045]:
                for spr_strength in [0.00, 0.010, 0.020, 0.030]:
                    for center in [0.48, 0.55, 0.62]:
                        config = ChallengerConfig(
                            upward_speed_multiplier=upward,
                            downward_speed_multiplier=downward,
                            inventory_relief_strength=inventory_strength,
                            spr_credibility_strength=spr_strength,
                            coverage_center=center,
                        )
                        simulation = simulate_challenger(event_df, assumptions, behavior, config)
                        metrics = model_metrics(simulation)
                        score = candidate_score(metrics, config)
                        rows.append(
                            {
                                "candidate_id": candidate_id,
                                **asdict(config),
                                "regularization_penalty": regularization_penalty(config),
                                "regularized_score": score,
                                "RMSE": metrics["RMSE"],
                                "MAE": metrics["MAE"],
                                "MAPE": metrics["MAPE"],
                                "方向命中率": metrics["方向命中率"],
                                "最大绝对误差": metrics["最大绝对误差"],
                                "峰值误差": metrics["峰值误差"],
                                "末日误差": metrics["末日误差"],
                                "高位平台形成_RMSE": metrics["高位平台形成_RMSE"],
                                "中期再定价回落_RMSE": metrics["中期再定价回落_RMSE"],
                                "后期二次抬升_RMSE": metrics["后期二次抬升_RMSE"],
                            }
                        )
                        paths[candidate_id] = simulation
                        candidate_id += 1

    candidates = pd.DataFrame(rows).sort_values(["regularized_score", "RMSE"]).reset_index(drop=True)
    best_id = int(candidates.iloc[0]["candidate_id"])
    best_config = ChallengerConfig(
        upward_speed_multiplier=float(candidates.iloc[0]["upward_speed_multiplier"]),
        downward_speed_multiplier=float(candidates.iloc[0]["downward_speed_multiplier"]),
        inventory_relief_strength=float(candidates.iloc[0]["inventory_relief_strength"]),
        spr_credibility_strength=float(candidates.iloc[0]["spr_credibility_strength"]),
        coverage_center=float(candidates.iloc[0]["coverage_center"]),
        credibility_ramp_days=int(candidates.iloc[0]["credibility_ramp_days"]),
    )
    return candidates, paths[best_id], best_config


def metric_table(champion: pd.DataFrame, challenger: pd.DataFrame, challenger_config: ChallengerConfig, candidates: pd.DataFrame) -> pd.DataFrame:
    champion_metrics = model_metrics(champion)
    challenger_metrics = model_metrics(challenger)
    best_unregularized = candidates.sort_values(["RMSE", "regularized_score"]).iloc[0]
    rows = [
        {
            "模型": "Champion主机制模型",
            "RMSE": champion_metrics["RMSE"],
            "MAE": champion_metrics["MAE"],
            "MAPE": champion_metrics["MAPE"],
            "方向命中率": champion_metrics["方向命中率"],
            "最大绝对误差": champion_metrics["最大绝对误差"],
            "regularized_score": np.nan,
            "说明": "当前官方主机制路径",
        },
        {
            "模型": "文献增强Challenger",
            "RMSE": challenger_metrics["RMSE"],
            "MAE": challenger_metrics["MAE"],
            "MAPE": challenger_metrics["MAPE"],
            "方向命中率": challenger_metrics["方向命中率"],
            "最大绝对误差": challenger_metrics["最大绝对误差"],
            "regularized_score": candidate_score(challenger_metrics, challenger_config),
            "说明": "正则化选择的库存/SPR/非对称调价候选",
        },
        {
            "模型": "RMSE最小Challenger",
            "RMSE": float(best_unregularized["RMSE"]),
            "MAE": float(best_unregularized["MAE"]),
            "MAPE": float(best_unregularized["MAPE"]),
            "方向命中率": float(best_unregularized["方向命中率"]),
            "最大绝对误差": float(best_unregularized["最大绝对误差"]),
            "regularized_score": float(best_unregularized["regularized_score"]),
            "说明": "仅供观察，不作为最终选择标准",
        },
    ]
    metrics = pd.DataFrame(rows)
    champion_rmse = float(metrics.loc[metrics["模型"] == "Champion主机制模型", "RMSE"].iloc[0])
    metrics["相对Champion_RMSE改善率"] = (champion_rmse - metrics["RMSE"]) / champion_rmse * 100
    return metrics


def draw_figure(champion: pd.DataFrame, challenger: pd.DataFrame, candidates: pd.DataFrame) -> None:
    configure_plot_style(savefig_dpi=260, figure_dpi=150, title_size=13)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))

    ax = axes[0, 0]
    ax.plot(champion["trade_date"], champion["actual_price"], color=SCENARIO_COLORS["actual"], lw=2.0, label="实际价格")
    ax.plot(champion["trade_date"], champion["simulated_price"], color=SCENARIO_COLORS["fit"], lw=1.7, label="Champion")
    ax.plot(challenger["trade_date"], challenger["simulated_price"], color=SCENARIO_COLORS["buffer"], lw=1.8, linestyle="--", label="Challenger")
    ax.axhspan(110, 120, color=SCENARIO_COLORS["band_outer"], alpha=0.35, label="110-120区间")
    ax.set_title("Champion 与文献增强 Challenger 路径")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="lower right", ncol=2)

    ax = axes[0, 1]
    champion_error = champion["simulated_price"] - champion["actual_price"]
    challenger_error = challenger["simulated_price"] - challenger["actual_price"]
    ax.plot(champion["trade_date"], champion_error, color=SCENARIO_COLORS["fit"], lw=1.5, label="Champion误差")
    ax.plot(challenger["trade_date"], challenger_error, color=SCENARIO_COLORS["buffer"], lw=1.5, linestyle="--", label="Challenger误差")
    ax.axhline(0, color=PAPER_COLORS["ink"], lw=0.9)
    ax.set_title("逐日误差对比")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper left")

    ax = axes[1, 0]
    top = candidates.sort_values("regularized_score").head(25)
    scatter = ax.scatter(
        top["regularization_penalty"],
        top["RMSE"],
        c=top["regularized_score"],
        cmap="viridis",
        s=44,
        alpha=0.85,
    )
    ax.set_title("候选复杂度与 RMSE")
    ax.set_xlabel("结构惩罚")
    ax.set_ylabel("RMSE")
    fig.colorbar(scatter, ax=ax, label="正则化得分")

    ax = axes[1, 1]
    components = challenger[
        [
            "trade_date",
            "buffer_confirmation_discount",
            "expectation_relief_discount",
            "inventory_relief_discount",
            "spr_credibility_discount",
        ]
    ].copy()
    ax.stackplot(
        components["trade_date"],
        components["buffer_confirmation_discount"],
        components["expectation_relief_discount"],
        components["inventory_relief_discount"],
        components["spr_credibility_discount"],
        labels=["缓冲确认", "预期修复", "库存覆盖", "SPR可信度"],
        colors=[SCENARIO_COLORS["fit"], SCENARIO_COLORS["buffer"], SCENARIO_COLORS["neutral"], SCENARIO_COLORS["optimistic"]],
        alpha=0.78,
    )
    ax.set_title("Challenger 下行机制分解")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper left", fontsize=8)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(Paths.figure_path)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for row in df[columns].to_dict("records")]
    return "\n".join([header, sep, *rows])


def build_report(metrics: pd.DataFrame, events: pd.DataFrame, candidates: pd.DataFrame, config: ChallengerConfig) -> str:
    champion = metrics[metrics["模型"] == "Champion主机制模型"].iloc[0]
    challenger = metrics[metrics["模型"] == "文献增强Challenger"].iloc[0]
    best_rmse = metrics[metrics["模型"] == "RMSE最小Challenger"].iloc[0]
    metric_show = metrics.copy()
    for column in ["RMSE", "MAE", "MAPE", "方向命中率", "最大绝对误差", "regularized_score", "相对Champion_RMSE改善率"]:
        metric_show[column] = metric_show[column].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    event_show = events.copy()
    for column in ["RMSE", "MAE", "平均误差", "最大绝对误差"]:
        event_show[column] = event_show[column].map(lambda x: f"{float(x):.3f}")
    top_show = candidates.head(8).copy()
    for column in ["upward_speed_multiplier", "downward_speed_multiplier", "inventory_relief_strength", "spr_credibility_strength", "coverage_center", "regularized_score", "RMSE", "中期再定价回落_RMSE"]:
        top_show[column] = top_show[column].map(lambda x: f"{float(x):.3f}")

    challenger_gain = float(challenger["相对Champion_RMSE改善率"])
    best_gain = float(best_rmse["相对Champion_RMSE改善率"])
    if abs(challenger_gain) < 0.005:
        challenger_gain = 0.0
    if abs(best_gain) < 0.005:
        best_gain = 0.0

    if challenger_gain >= 1.0:
        decision = "Challenger 有可观察收益，但仍需要进一步做删块和扰动稳定性验证，暂不直接替换 Champion。"
    elif challenger_gain > 0:
        decision = "Challenger 只有小幅收益，说明结构方向有价值，但不足以替换当前主机制模型。"
    else:
        decision = "Challenger 未优于 Champion，当前主机制模型应继续保留。"

    return f"""# 短期文献增强 Challenger 报告

> 本报告只用于主机制模型优化研究，不替换当前官方短期路径，不写入论文正文。

## 结构来源

本轮 Challenger 参考油价冲击分解、库存便利收益、SPR政策可信度和金融市场非对称调价思想，只在当前主机制框架内新增三个低自由度结构：

1. 库存/缓冲折价由缓冲覆盖率和库存剩余比例共同决定。
2. SPR 既提供实际释放量，也提供有上限的政策可信度折价。
3. 价格向目标价调整时区分上行冲击速度和下行修复速度。

## 结论

{decision}

正则化选择的 Challenger 相对 Champion 的 RMSE 改善为 {challenger_gain:.2f}%。仅按 RMSE 选择的 Challenger 改善为 {best_gain:.2f}%，但本文不把 RMSE 最小作为唯一选择标准，因为这会鼓励结构自由度过高。

## 总体指标

{markdown_table(metric_show, ['模型', 'RMSE', 'MAE', 'MAPE', '方向命中率', '最大绝对误差', 'regularized_score', '相对Champion_RMSE改善率', '说明'])}

## 事件窗口

{markdown_table(event_show, ['模型', '事件窗口', '样本数', 'RMSE', 'MAE', '平均误差', '最大绝对误差'])}

## 正则化最优候选

{markdown_table(top_show, ['candidate_id', 'upward_speed_multiplier', 'downward_speed_multiplier', 'inventory_relief_strength', 'spr_credibility_strength', 'coverage_center', 'regularized_score', 'RMSE', '中期再定价回落_RMSE'])}

## 被选中配置

| 参数 | 数值 |
|---|---:|
| upward_speed_multiplier | {config.upward_speed_multiplier:.3f} |
| downward_speed_multiplier | {config.downward_speed_multiplier:.3f} |
| inventory_relief_strength | {config.inventory_relief_strength:.3f} |
| spr_credibility_strength | {config.spr_credibility_strength:.3f} |
| coverage_center | {config.coverage_center:.3f} |

## 建模判断

- 如果 Challenger 明显优于 Champion，应继续做删块检验、局部扰动和贡献项量级审计。
- 如果 Challenger 只小幅优于 Champion，则保留为模型优化证据，不替换主模型。
- 如果 Challenger 不优于 Champion，则说明当前主机制模型已经足够稳，论文中继续使用 Champion。

## 文献依据与吸收方式

- Kilian 的油价冲击分解思想适合支撑“供应冲击、需求调整、油市特定风险”三类机制分解，但不要求在短期窗口继续增加自由参数。
- Kilian 与 Murphy 关于库存和投机交易的研究说明库存状态会影响油价风险溢价；本轮以“缓冲覆盖率 × 库存剩余比例”检验该思想，未发现足够替换收益。
- Convenience yield 与期货期限结构文献说明库存紧张会抬高近月价格，因此更适合在拿到完整多期限 Brent 结算价后进入长期/风险约束层。
- SPR 研究说明政策释放影响价格既有物理供给渠道，也有预期渠道；本轮加入有上限的 SPR 可信度折价后，短期拟合收益不足，暂不进入主模型。
- 金融市场非对称调价思想支持“冲击上涨快、修复回落慢”的设定；本轮检验显示当前 Champion 已经足够稳定，非对称调价不应为了微小 RMSE 收益被强行采用。

## 输出

- `{Paths.challenger_path_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.candidate_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.metrics_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.event_csv.relative_to(PROJECT_ROOT)}`
- `{Paths.figure_path.relative_to(PROJECT_ROOT)}`
"""


def main() -> None:
    ensure_parents([Paths.challenger_path_csv, Paths.candidate_csv, Paths.metrics_csv, Paths.event_csv, Paths.report_path, Paths.figure_path])
    event_df = load_event_window()
    assumptions, behavior, _ = load_best_objects()
    champion = load_current_path()
    candidates, challenger, config = run_challenger_search(event_df, assumptions, behavior)
    metrics = metric_table(champion, challenger, config, candidates)
    events = pd.concat(
        [
            event_metrics(champion, "Champion主机制模型"),
            event_metrics(challenger, "文献增强Challenger"),
        ],
        ignore_index=True,
    )
    draw_figure(champion, challenger, candidates)

    challenger.to_csv(Paths.challenger_path_csv, index=False)
    candidates.to_csv(Paths.candidate_csv, index=False)
    metrics.to_csv(Paths.metrics_csv, index=False)
    events.to_csv(Paths.event_csv, index=False)
    Paths.report_path.write_text(build_report(metrics, events, candidates, config), encoding="utf-8")

    print("Short-term literature challenger complete")
    print(f"Report: {Paths.report_path.relative_to(PROJECT_ROOT)}")
    print(f"Metrics: {Paths.metrics_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
