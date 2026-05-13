"""Traditional supply-demand counterfactual baseline model.

This baseline deliberately keeps only the direct supply-gap effect and a low
short-term demand-price elasticity. Its job is not to forecast the realized
event-window oil price, but to show how much a static low-elasticity supply-
demand view can overestimate prices before buffer and expectation mechanisms
are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.common.paths import PROJECT_ROOT, ensure_parent
from src.common.plotting import configure_plot_style as apply_plot_style

CONFIG_PATH = PROJECT_ROOT / "config" / "base.yml"
PROBLEM_PARAMETERS_PATH = PROJECT_ROOT / "data" / "metadata" / "题面参数表.csv"


@dataclass(frozen=True)
class BaselineConfig:
    event_csv: Path
    output_csv: Path
    figure_path: Path
    report_path: Path
    pre_war_supply: float
    pre_war_demand: float
    elasticity: float
    interruptions: tuple[float, float, float]


def load_yaml_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config(config: dict[str, Any]) -> BaselineConfig:
    paths = config["paths"]
    assumptions = config["baseline_assumptions"]
    return BaselineConfig(
        event_csv=PROJECT_ROOT / paths["processed_event_window_csv"],
        output_csv=PROJECT_ROOT / "output" / "baseline" / "传统供需基准模型结果.csv",
        figure_path=PROJECT_ROOT / "figures" / "baseline_vs_actual.png",
        report_path=PROJECT_ROOT / "output" / "reports" / "传统供需基准模型报告.md",
        pre_war_supply=float(assumptions["pre_war_supply"]),
        pre_war_demand=float(assumptions["pre_war_demand"]),
        elasticity=float(assumptions["short_term_price_elasticity"]),
        interruptions=(
            float(assumptions["supply_interruption_low"]),
            float(assumptions["supply_interruption_mid"]),
            float(assumptions["supply_interruption_high"]),
        ),
    )


def load_event_window(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Event-window CSV not found: {path}")
    return pd.read_csv(path, parse_dates=["trade_date"])


def baseline_price_linearized(base_price: float, shortage_ratio: float, elasticity: float) -> float:
    """Linear-demand counterfactual estimate.

    Under a local linear inverse-demand curve, the point elasticity at P0 gives

        price_change_pct = shortage_ratio / abs(elasticity)

    exactly for the counterfactual line used here. This formula should not be
    interpreted as a first-order approximation to the constant-elasticity
    demand curve when the shortage ratio is large.
    """

    if elasticity >= 0:
        raise ValueError("Demand price elasticity must be negative for this baseline.")
    return base_price * (1 + shortage_ratio / abs(elasticity))


def baseline_price_constant_elasticity(base_price: float, supply_ratio: float, elasticity: float) -> float:
    """Mechanical constant-elasticity equilibrium estimate.

    This is kept as a theoretical upper-bound reference. With very low short-run
    oil demand elasticity, the value can become unrealistically large.
    """

    if elasticity >= 0:
        raise ValueError("Demand price elasticity must be negative for this baseline.")
    return base_price * (supply_ratio ** (1 / elasticity))


def run_baseline_model(event_df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    base_row = event_df.iloc[0]
    base_price = float(base_row["pre_close"])
    actual_peak_close = float(event_df["close_price"].max())
    actual_peak_high = float(event_df["high_price"].max())
    actual_mean_close = float(event_df["close_price"].mean())
    actual_final_close = float(event_df["close_price"].iloc[-1])

    rows: list[dict[str, Any]] = []
    for scenario, interruption in zip(["低中断", "中中断", "高中断"], cfg.interruptions):
        effective_supply = cfg.pre_war_supply - interruption
        supply_ratio = effective_supply / cfg.pre_war_demand
        shortage_ratio = interruption / cfg.pre_war_demand
        linear_price = baseline_price_linearized(base_price, shortage_ratio, cfg.elasticity)
        constant_elasticity_price = baseline_price_constant_elasticity(base_price, supply_ratio, cfg.elasticity)
        rows.append(
            {
                "情景": scenario,
                "供应中断量_万桶每日": interruption,
                "有效供给_万桶每日": effective_supply,
                "供给缺口比例": shortage_ratio,
                "基准价格_美元每桶": base_price,
                "需求价格弹性": cfg.elasticity,
                "线性需求反事实价格_美元每桶": linear_price,
                "常弹性机械上界价格_美元每桶": constant_elasticity_price,
                "实际窗口最高收盘价_美元每桶": actual_peak_close,
                "实际窗口最高盘中价_美元每桶": actual_peak_high,
                "实际窗口平均收盘价_美元每桶": actual_mean_close,
                "实际窗口末日收盘价_美元每桶": actual_final_close,
                "线性模型相对最高收盘价倍数": linear_price / actual_peak_close,
                "常弹性上界相对最高收盘价倍数": constant_elasticity_price / actual_peak_close,
                "线性模型相对窗口均价倍数": linear_price / actual_mean_close,
            }
        )
    return pd.DataFrame(rows)


def save_results(results: pd.DataFrame, cfg: BaselineConfig) -> None:
    ensure_parent(cfg.output_csv)
    results.to_csv(cfg.output_csv, index=False)


def configure_plot_style() -> None:
    apply_plot_style(savefig_dpi=180, figure_dpi=150, title_size=13)


def save_figure(event_df: pd.DataFrame, results: pd.DataFrame, cfg: BaselineConfig) -> None:
    ensure_parent(cfg.figure_path)
    configure_plot_style()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        event_df["trade_date"],
        event_df["close_price"],
        color="#2563eb",
        linewidth=1.8,
        marker="o",
        markersize=2.6,
        label="附件CSV实际收盘价",
    )

    colors = {
        "低中断": "#f59e0b",
        "中中断": "#ef4444",
        "高中断": "#7c3aed",
    }
    for _, row in results.iterrows():
        ax.axhline(
            row["线性需求反事实价格_美元每桶"],
            color=colors[row["情景"]],
            linestyle="--",
            linewidth=1.6,
            label=f"{row['情景']}线性需求基准: {row['线性需求反事实价格_美元每桶']:.1f}",
        )

    ax.axhspan(110, 120, color="#10b981", alpha=0.10, label="题面叙述110-120区间")
    ax.set_title("线性需求反事实基准与实际价格对比")
    ax.set_xlabel("日期")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(cfg.figure_path, dpi=180)
    plt.close(fig)


def build_report(event_df: pd.DataFrame, results: pd.DataFrame, cfg: BaselineConfig) -> str:
    base_price = float(event_df.iloc[0]["pre_close"])
    event_start = event_df["trade_date"].min().date()
    event_end = event_df["trade_date"].max().date()
    rows = "\n".join(
        "| {情景} | {供应中断量_万桶每日:.0f} | {线性需求反事实价格_美元每桶:.2f} | {常弹性机械上界价格_美元每桶:.2f} | {线性模型相对最高收盘价倍数:.2f} | {常弹性上界相对最高收盘价倍数:.2f} |".format(
            **row
        )
        for row in results.to_dict("records")
    )

    return f"""# 传统供需基准模型报告

## 运行结论

已建立传统供需反事实基准模型。在线性需求曲线口径下，供应中断 1400-1800 万桶/日会给出约 278-337 美元/桶的理论价格，明显高于附件 CSV 中冲突窗口最高收盘价 114.06 美元/桶。

这说明只看供应缺口和低需求弹性会显著高估油价，后续必须引入战略储备、商业库存、绕道运输、需求收缩、价格传导折价和预期反转等缓冲机制。

## 输入数据

- 实际价格窗口：`{cfg.event_csv.relative_to(PROJECT_ROOT)}`
- 题面参数表：`{PROBLEM_PARAMETERS_PATH.relative_to(PROJECT_ROOT)}`
- 冲突窗口实际交易日：{event_start} 至 {event_end}
- 基准价格：{base_price:.2f} USD/barrel，即冲突窗口首日 `pre_close`
- 短期需求价格弹性：{cfg.elasticity}

## 模型口径与需求函数假设

主口径采用线性需求曲线假设。设局部反需求函数为 \\(P=a-bQ\\)，在战前均衡点附近用题面短期点弹性 \\(\\varepsilon\\) 标定斜率，则供应缺口对应的价格压力为：

```text
price = base_price * (1 + supply_shortage_ratio / abs(elasticity))
```

这里的公式不是把常弹性需求函数做一阶近似，而是线性需求反事实基准的主口径。选择线性需求，是因为它隐含消费者边际支付意愿随消费量下降；若采用常弹性需求 \\(Q=A P^\\varepsilon\\)，在 \\(\\varepsilon=-0.05\\) 且缺口较大时会给出极高的机械上界。本文同时保留常弹性口径，用来说明“传统低弹性供需模型高估现实价格”的结论不依赖单一公式。

## 结果表

| 情景 | 供应中断量(万桶/日) | 线性需求反事实价格 | 常弹性机械上界价格 | 线性口径相对最高收盘倍数 | 常弹性口径相对最高收盘倍数 |
|---|---:|---:|---:|---:|---:|
{rows}

实际窗口末日收盘价为 {float(results["实际窗口末日收盘价_美元每桶"].iloc[0]):.2f} 美元/桶。基准模型的职责不是精确预测末日价格，而是证明静态低弹性供需框架无法解释真实价格平台。

## 输出产物

- `{cfg.output_csv.relative_to(PROJECT_ROOT)}`
- `{cfg.figure_path.relative_to(PROJECT_ROOT)}`

## 后续作用

- 综合机制递推模型在此基础上引入库存、SPR、绕道运输和需求弹性变化。
- 参数校准环节使用实际价格路径约束动态模型，而不是继续依赖该静态基准。
"""


def write_report(event_df: pd.DataFrame, results: pd.DataFrame, cfg: BaselineConfig) -> None:
    ensure_parent(cfg.report_path)
    cfg.report_path.write_text(build_report(event_df, results, cfg), encoding="utf-8")


def main() -> None:
    config = load_yaml_config()
    cfg = resolve_config(config)
    event_df = load_event_window(cfg.event_csv)
    results = run_baseline_model(event_df, cfg)
    save_results(results, cfg)
    save_figure(event_df, results, cfg)
    write_report(event_df, results, cfg)

    print("Baseline model complete")
    print(f"Results: {cfg.output_csv.relative_to(PROJECT_ROOT)}")
    print(f"Figure: {cfg.figure_path.relative_to(PROJECT_ROOT)}")
    print(f"Report: {cfg.report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
