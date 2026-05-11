"""Generate figures used by the final paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "Heiti TC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 280,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


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
    ax.axvspan(observed_start, cutoff_date, color="#dbeafe", alpha=0.22, lw=0)
    ax.axvspan(cutoff_date, forecast_end, color="#fef3c7", alpha=0.22, lw=0)

    ax.plot(
        short_term["trade_date"],
        short_term["actual_price"],
        color="#111827",
        linewidth=2.3,
        marker="o",
        markersize=3.2,
        label="附件真实收盘价",
        zorder=4,
    )
    ax.plot(
        short_term["trade_date"],
        short_term["simulated_price"],
        color="#2563eb",
        linewidth=2.2,
        label="短期机制拟合",
        zorder=3,
    )

    color_map = {
        "optimistic": ("乐观情景预测", "#059669"),
        "neutral": ("中性情景预测", "#2563eb"),
        "pessimistic": ("悲观情景预测", "#dc2626"),
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

    ax.axvline(cutoff_date, color="#111827", linewidth=1.2, linestyle=":")
    ax.text(
        cutoff_date,
        ax.get_ylim()[1] - 2,
        "真实数据截止\n2026-05-05",
        ha="right",
        va="top",
        fontsize=9,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#9ca3af", "alpha": 0.9},
    )
    ax.axhspan(110, 120, color="#10b981", alpha=0.10, label="题面 110-120 美元平台")
    ax.text(observed_start, 121.0, "短期拟合区", color="#1d4ed8", fontsize=10, weight="bold")
    ax.text(cutoff_date + pd.Timedelta(days=9), 121.0, "60-180 天情景外推区", color="#92400e", fontsize=10, weight="bold")

    ax.set_title("短期机制拟合与 60-180 天三情景预测总览")
    ax.set_xlabel("日期")
    ax.set_ylabel("美元/桶")
    ax.set_ylim(72, 134)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.legend(loc="upper left", ncol=2, frameon=True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    short_term, scenarios = load_frames()
    outputs = [save_integrated_forecast_figure(short_term, scenarios)]
    print("Generated final paper figures:")
    for path in outputs:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
