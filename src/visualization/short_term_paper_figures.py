"""Generate polished static figures for the short-term model paper."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT
from src.common.plotting import configure_plot_style as apply_plot_style

PAPER_FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"


def configure_plot_style() -> None:
    apply_plot_style(savefig_dpi=260, figure_dpi=150, title_size=14)
    plt.rcParams.update({"axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9})


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = pd.read_csv(
        PROJECT_ROOT / "output" / "calibration" / "动态模型校准后路径.csv",
        parse_dates=["trade_date"],
    )
    model["error"] = model["simulated_price"] - model["actual_price"]
    model["abs_error"] = model["error"].abs()
    segment = pd.read_csv(PROJECT_ROOT / "output" / "calibration" / "动态模型分段误差.csv")
    candidates = pd.read_csv(PROJECT_ROOT / "output" / "calibration" / "动态模型候选参数前10.csv")
    return model, segment, candidates


def add_phase_background(ax: plt.Axes, model: pd.DataFrame) -> None:
    phase_ranges = [
        ("前期冲击", 0, 14, "#fee2e2"),
        ("中期平台", 15, 35, "#fef3c7"),
        ("后期再定价", 36, int(model["day_index"].max()), "#dcfce7"),
    ]
    for label, start_day, end_day, color in phase_ranges:
        sub = model[(model["day_index"] >= start_day) & (model["day_index"] <= end_day)]
        if sub.empty:
            continue
        ax.axvspan(sub["trade_date"].min(), sub["trade_date"].max(), color=color, alpha=0.28, lw=0)
        ax.text(
            sub["trade_date"].min(),
            ax.get_ylim()[1],
            label,
            va="top",
            ha="left",
            fontsize=9,
            color="#374151",
        )


def save_fit_figure(model: pd.DataFrame) -> Path:
    path = PAPER_FIGURES_DIR / "短期模型拟合效果.png"
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(
        model["trade_date"],
        model["actual_price"],
        marker="o",
        markersize=3.8,
        linewidth=2.1,
        color="#111827",
        label="实际收盘价",
    )
    ax.plot(
        model["trade_date"],
        model["simulated_price"],
        marker="s",
        markersize=3.2,
        linewidth=2.0,
        color="#2563eb",
        label="短期动态模型",
    )
    ax.axhspan(110, 120, color="#10b981", alpha=0.12, label="题面 110-120 美元区间")
    ax.set_ylim(min(model["actual_price"].min(), model["simulated_price"].min()) - 5, 123)
    add_phase_background(ax, model)
    ax.set_title("短期动态模型对冲突窗口油价的拟合效果")
    ax.set_xlabel("日期")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper left", ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_error_figure(model: pd.DataFrame, segment: pd.DataFrame) -> Path:
    path = PAPER_FIGURES_DIR / "短期模型误差诊断.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), gridspec_kw={"height_ratios": [1.0, 1.05]})
    colors = np.where(model["error"] >= 0, "#ef4444", "#0f766e")
    axes[0].bar(model["trade_date"], model["error"], color=colors, width=1.2)
    axes[0].axhline(0, color="#111827", lw=0.9)
    axes[0].set_title("逐日拟合误差：模拟价格 - 实际价格")
    axes[0].set_ylabel("美元/桶")

    segment_plot = segment[["分段", "RMSE", "MAE"]].copy()
    x = np.arange(len(segment_plot))
    width = 0.35
    axes[1].bar(x - width / 2, segment_plot["RMSE"], width=width, label="RMSE", color="#2563eb")
    axes[1].bar(x + width / 2, segment_plot["MAE"], width=width, label="MAE", color="#0f766e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(segment_plot["分段"], rotation=18, ha="right")
    axes[1].set_ylabel("美元/桶")
    axes[1].set_title("分段误差对比")
    axes[1].legend()
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_mechanism_figure(model: pd.DataFrame) -> Path:
    path = PAPER_FIGURES_DIR / "短期模型机制贡献.png"
    fig, ax = plt.subplots(figsize=(11, 6.2))
    mechanism_map = {
        "shortage_pressure": ("供需缺口压力", "#b91c1c", "-"),
        "blockade_risk_premium": ("封锁风险溢价", "#f97316", "-"),
        "uncertainty_premium": ("不确定性溢价", "#7c3aed", "-"),
        "panic_premium": ("恐慌溢价", "#64748b", "-"),
        "buffer_confirmation_discount": ("缓冲确认折价", "#059669", "--"),
        "expectation_relief_discount": ("预期修复折价", "#0f766e", "--"),
    }
    for column, (label, color, style) in mechanism_map.items():
        if column not in model.columns:
            continue
        values = -model[column] if "discount" in column else model[column]
        ax.plot(model["trade_date"], values, label=label, color=color, linestyle=style, linewidth=2.0)

    ax.axhline(0, color="#111827", lw=0.9)
    ax.set_title("短期动态模型中的价格推升项与压低项")
    ax.set_xlabel("日期")
    ax.set_ylabel("价格贡献：正值推升，负值压低")
    ax.legend(loc="upper left", ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_candidate_figure(candidates: pd.DataFrame) -> Path:
    path = PAPER_FIGURES_DIR / "候选模型误差对比.png"
    selected = candidates.head(6).copy()
    selected["候选标签"] = selected["candidate_id"].astype(str) + "\n" + selected["candidate_source"].replace(
        {"continuous_refinement": "精修", "seeded_random": "随机"}
    )
    metrics = ["RMSE", "高价平台RMSE", "低价回落RMSE", "后期RMSE"]
    x = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    width = 0.18
    palette = ["#2563eb", "#7c3aed", "#0f766e", "#f97316"]
    for idx, metric in enumerate(metrics):
        ax.bar(x + (idx - 1.5) * width, selected[metric], width=width, label=metric, color=palette[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(selected["候选标签"])
    ax.set_title("代表候选模型的误差结构对比")
    ax.set_ylabel("美元/桶")
    ax.legend(ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    model, segment, candidates = load_frames()
    outputs = [
        save_fit_figure(model),
        save_error_figure(model, segment),
        save_mechanism_figure(model),
        save_candidate_figure(candidates),
    ]
    print("Generated short-term paper figures:")
    for path in outputs:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
