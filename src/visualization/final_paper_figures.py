"""Generate figures used by the final paper."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

from src.common.paths import PROJECT_ROOT
from src.common.plotting import SCENARIO_COLORS, configure_plot_style as apply_plot_style, direct_label

PAPER_FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"


def configure_plot_style() -> None:
    apply_plot_style(savefig_dpi=280, figure_dpi=150, title_size=14)
    plt.rcParams.update({"axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9})


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    short_term = pd.read_csv(
        PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv",
        parse_dates=["trade_date"],
    )
    scenarios = pd.read_csv(
        PROJECT_ROOT / "output" / "scenarios" / "三情景预测结果.csv",
        parse_dates=["trade_date"],
    )
    return short_term, scenarios


def save_integrated_forecast_figure(short_term: pd.DataFrame, scenarios: pd.DataFrame) -> Path:
    path = PAPER_FIGURES_DIR / "短期拟合与长期预测总览.png"
    cutoff_date = short_term["trade_date"].max()
    cutoff_price = float(short_term.loc[short_term["trade_date"].idxmax(), "simulated_price"])

    fig, ax = plt.subplots(figsize=(12, 6.6))

    observed_start = short_term["trade_date"].min()
    forecast_end = scenarios["trade_date"].max()
    ax.axvspan(observed_start, cutoff_date, color="#E8EEF6", alpha=0.36, lw=0)
    ax.axvspan(cutoff_date, forecast_end, color="#F2E8D5", alpha=0.30, lw=0)

    ax.plot(
        short_term["trade_date"],
        short_term["actual_price"],
        color=SCENARIO_COLORS["actual"],
        linewidth=2.3,
        marker="o",
        markersize=3.2,
        label="附件真实收盘价",
        zorder=4,
    )
    ax.plot(
        short_term["trade_date"],
        short_term["simulated_price"],
        color=SCENARIO_COLORS["fit"],
        linewidth=2.2,
        label="短期机制拟合",
        zorder=3,
    )

    color_map = {
        "optimistic": ("乐观情景预测", SCENARIO_COLORS["optimistic"]),
        "neutral": ("中性情景预测", SCENARIO_COLORS["neutral"]),
        "pessimistic": ("悲观情景预测", SCENARIO_COLORS["pessimistic"]),
    }
    for scenario, (label, color) in color_map.items():
        forecast = scenarios[(scenarios["scenario"] == scenario) & (~scenarios["is_observed_price"].astype(bool))]
        if forecast.empty:
            continue
        bridge = pd.DataFrame(
            {
                "trade_date": [cutoff_date],
                "forecast_price": [cutoff_price],
            }
        )
        line = pd.concat([bridge, forecast[["trade_date", "forecast_price"]]], ignore_index=True)
        ax.plot(
            line["trade_date"],
            line["forecast_price"],
            color=color,
            linewidth=2.2,
            linestyle="--" if scenario != "neutral" else "-",
            label=label,
            zorder=2,
        )

    ax.axvline(cutoff_date, color=SCENARIO_COLORS["actual"], linewidth=1.2, linestyle=":")
    ax.text(
        cutoff_date,
        ax.get_ylim()[1] - 2,
        "真实数据截止\n2026-05-05",
        ha="right",
        va="top",
        fontsize=9,
        color=SCENARIO_COLORS["actual"],
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#9ca3af", "alpha": 0.9},
    )
    ax.axhspan(110, 120, color=SCENARIO_COLORS["optimistic"], alpha=0.08, label="题面 110-120 美元平台")
    ax.text(observed_start, 121.0, "短期拟合区", color=SCENARIO_COLORS["neutral"], fontsize=10, weight="bold")
    ax.text(cutoff_date + pd.Timedelta(days=9), 121.0, "60-180 天情景外推区", color="#8C510A", fontsize=10, weight="bold")

    ax.set_title("短期机制拟合与 60-180 天三情景预测总览")
    ax.set_xlabel("日期")
    ax.set_ylabel("美元/桶")
    ax.set_ylim(72, 134)
    ax.set_xlim(observed_start, forecast_end + pd.Timedelta(days=18))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for scenario, (_, color) in color_map.items():
        forecast = scenarios[(scenarios["scenario"] == scenario) & (~scenarios["is_observed_price"].astype(bool))]
        if forecast.empty:
            continue
        label = {"optimistic": "乐观情景", "neutral": "中性情景", "pessimistic": "悲观情景"}[scenario]
        direct_label(
            ax,
            forecast["trade_date"].iloc[-1],
            forecast["forecast_price"].iloc[-1],
            label,
            color,
            dx=8,
            dy=0,
            size=9.5,
        )
    direct_label(
        ax,
        short_term["trade_date"].iloc[-1],
        short_term["actual_price"].iloc[-1],
        "真实价格",
        SCENARIO_COLORS["actual"],
        dx=-50,
        dy=12,
        size=9.2,
    )
    ax.legend(loc="lower left", ncol=2, frameon=True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_method_route_figure() -> Path:
    path = PAPER_FIGURES_DIR / "论文总体技术路线图.png"
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    nodes = [
        ("附件数据清洗", "统一日期、价格口径\n截取冲突窗口", 0.65, 4.65, "#eff6ff", "#2563eb"),
        ("传统供需基准", "低短期弹性\n估计反事实上界", 3.10, 4.65, "#f8fafc", "#475569"),
        ("短期动态递推", "SPR、库存、绕道\n恐慌与预期修复", 5.55, 4.65, "#ecfdf5", "#059669"),
        ("模型检验防御", "基准对比、滞后检验\n消融与压力测试", 8.00, 4.65, "#fff7ed", "#ea580c"),
        ("长期情景预测", "60-180 天\n乐观/中性/悲观", 3.10, 1.55, "#fef2f2", "#dc2626"),
        ("敏感性分析", "识别关键参数\n解释政策含义", 5.55, 1.55, "#f5f3ff", "#7c3aed"),
        ("最终结论", "短期平台机制\n长期尾部风险", 8.00, 1.55, "#fefce8", "#ca8a04"),
    ]

    for title, subtitle, x, y, fill, edge in nodes:
        box = FancyBboxPatch(
            (x, y),
            1.9,
            1.05,
            boxstyle="round,pad=0.12,rounding_size=0.08",
            linewidth=1.6,
            edgecolor=edge,
            facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(x + 0.95, y + 0.70, title, ha="center", va="center", fontsize=13, weight="bold", color="#111827")
        ax.text(x + 0.95, y + 0.34, subtitle, ha="center", va="center", fontsize=10.5, color="#374151", linespacing=1.35)

    arrows = [
        ((2.55, 5.18), (3.05, 5.18)),
        ((5.00, 5.18), (5.50, 5.18)),
        ((7.45, 5.18), (7.95, 5.18)),
        ((6.50, 4.60), (4.35, 2.68)),
        ((7.00, 4.60), (6.55, 2.68)),
        ((9.00, 4.60), (9.00, 2.68)),
        ((5.00, 2.08), (5.50, 2.08)),
        ((7.45, 2.08), (7.95, 2.08)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.5,
                color="#6b7280",
                connectionstyle="arc3,rad=0.0",
            )
        )

    ax.text(
        6,
        6.35,
        "从附件真实数据出发，先解释短期价格平台，再外推长期情景风险",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
        color="#111827",
    )
    ax.text(
        6,
        0.72,
        "检验逻辑贯穿全流程：不编造数据、不把未来当拟合、不把价格平台写成硬编码上限",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#4b5563",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    short_term, scenarios = load_frames()
    outputs = [
        save_method_route_figure(),
        save_integrated_forecast_figure(short_term, scenarios),
    ]
    print("Generated final paper figures:")
    for path in outputs:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
