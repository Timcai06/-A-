"""Generate preview figures for proposed paper palettes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "candidate_figures"

PALETTES = {
    "warm": {
        "prefix": "配色预览",
        "deep": "#1e0941",
        "wine": "#6d1754",
        "rose": "#b03b57",
        "coral": "#e17151",
        "gold": "#fab251",
        "yellow": "#f9f871",
        "muted": "#5b5263",
        "grid": "#eadfe5",
        "edge": "#d7c9d2",
    },
    "dark_triad": {
        "prefix": "三色预览",
        "deep": "#41091a",
        "wine": "#00400c",
        "rose": "#00336d",
        "coral": "#7a2840",
        "gold": "#2b6f37",
        "yellow": "#e9edf5",
        "muted": "#4b4650",
        "grid": "#e8e3e5",
        "edge": "#d4c8cc",
    },
    "muted_oil": {
        "prefix": "降饱和预览",
        "deep": "#1A1A1A",
        "wine": "#254F7A",
        "rose": "#8A4B3A",
        "coral": "#A6654F",
        "gold": "#C7A76C",
        "yellow": "#EEF3F6",
        "muted": "#4F5963",
        "grid": "#E7E2DF",
        "edge": "#CFC7C3",
        "green": "#4F8A5B",
        "teal": "#3E8C87",
        "bluegray": "#3F7897",
        "band_mid": "#C9DCE5",
    },
    "tol_paper": {
        "prefix": "Tol论文预览",
        "deep": "#1A1A1A",
        "wine": "#4477AA",
        "rose": "#CC6677",
        "coral": "#882255",
        "gold": "#DDCC77",
        "yellow": "#EEF3F6",
        "muted": "#4B5563",
        "grid": "#E7E2DF",
        "edge": "#CFC7C3",
        "green": "#117733",
        "teal": "#44AA99",
        "bluegray": "#3F7897",
        "band_mid": "#C9DCE5",
    },
}


def setup_style(palette: dict[str, str]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": palette["edge"],
            "axes.labelcolor": palette["muted"],
            "axes.titlecolor": palette["deep"],
            "axes.titleweight": "bold",
            "xtick.color": palette["muted"],
            "ytick.color": palette["muted"],
            "grid.color": palette["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.85,
            "legend.frameon": False,
            "legend.labelcolor": palette["muted"],
            "savefig.dpi": 260,
        }
    )


def save(fig: plt.Figure, name: str, palette: dict[str, str]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{palette['prefix']}_{name}"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def preview_short_fit(palette: dict[str, str]) -> Path:
    df = pd.read_csv(ROOT / "output/calibration/动态模型校准后路径.csv", parse_dates=["trade_date"])
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.axhspan(110, 120, color=palette["yellow"], alpha=0.5, label="110-120 美元平台")
    ax.plot(df["trade_date"], df["actual_price"], color=palette["deep"], lw=2.4, marker="o", ms=3.2, label="实际收盘价")
    ax.plot(df["trade_date"], df["simulated_price"], color=palette["wine"], lw=2.2, marker="s", ms=2.8, label="短期冲击模型")
    ax.set_title("配色预览：短期冲击模型拟合效果")
    ax.set_xlabel("日期")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="lower left", ncol=3)
    fig.autofmt_xdate()
    return save(fig, "短期拟合.png", palette)


def preview_mechanism(palette: dict[str, str]) -> Path:
    df = pd.read_csv(ROOT / "output/calibration/动态模型校准后路径.csv", parse_dates=["trade_date"])
    mechanism = [
        ("shortage_pressure", "供需缺口压力", palette["rose"], "-"),
        ("blockade_risk_premium", "封锁风险溢价", palette["coral"], "-"),
        ("uncertainty_premium", "不确定性溢价", palette["gold"], "-"),
        ("panic_premium", "恐慌溢价", palette["muted"], "-"),
        ("buffer_confirmation_discount", "缓冲确认折价", palette["wine"], "--"),
        ("expectation_relief_discount", "预期修复折价", palette["deep"], "--"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for col, label, color, style in mechanism:
        if col not in df:
            continue
        values = -df[col] if "discount" in col else df[col]
        ax.plot(df["trade_date"], values, color=color, linestyle=style, lw=2.0, label=label)
    ax.axhline(0, color=palette["deep"], lw=0.9, alpha=0.65)
    ax.set_title("配色预览：短期机制贡献分解")
    ax.set_xlabel("日期")
    ax.set_ylabel("价格贡献")
    ax.legend(loc="upper right", ncol=2)
    fig.autofmt_xdate()
    return save(fig, "机制贡献.png", palette)


def preview_monte_carlo_fan(palette: dict[str, str]) -> Path:
    q = pd.read_csv(ROOT / "output/monte_carlo/蒙特卡洛路径分位数.csv", parse_dates=["trade_date"])
    fig, ax = plt.subplots(figsize=(11, 5.6))
    dates = q["trade_date"]
    ax.fill_between(dates, q["p05"], q["p95"], color=palette["yellow"], alpha=0.62, label="P05-P95")
    ax.fill_between(dates, q["p25"], q["p75"], color=palette["gold"], alpha=0.28, label="P25-P75")
    ax.plot(dates, q["p50"], color=palette["wine"], lw=2.4, label="中位数路径")
    ax.axhline(120, color=palette["rose"], lw=1.2, linestyle="--", label="120 美元风险线")
    ax.set_title("配色预览：蒙特卡洛情景树扇形区间")
    ax.set_xlabel("日期")
    ax.set_ylabel("美元/桶")
    ax.legend(loc="upper right", ncol=4)
    fig.autofmt_xdate()
    return save(fig, "蒙特卡洛扇形.png", palette)


def preview_sensitivity(palette: dict[str, str]) -> Path:
    df = pd.read_csv(ROOT / "output/sensitivity/参数重要性排序.csv").head(8)
    df = df.sort_values("综合敏感度得分")
    colors = [palette["deep"], palette["wine"], palette["rose"], palette["coral"], palette["gold"], palette["rose"], palette["wine"], palette["deep"]]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    bars = ax.barh(df["参数"], df["综合敏感度得分"], color=colors[: len(df)], alpha=0.92)
    for bar, value in zip(bars, df["综合敏感度得分"], strict=False):
        ax.text(value + 0.25, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", color=palette["deep"], fontsize=9)
    ax.set_title("配色预览：关键参数敏感度排序")
    ax.set_xlabel("综合敏感度")
    ax.set_ylabel("")
    ax.set_xlim(0, max(df["综合敏感度得分"]) * 1.18)
    return save(fig, "敏感性排序.png", palette)


def preview_parameter_profile(palette: dict[str, str]) -> Path:
    df = pd.read_csv(ROOT / "output/calibration/短期模型参数剖面.csv")
    params = df["参数中文名"].drop_duplicates().head(5).tolist()
    colors = [palette["deep"], palette["wine"], palette["rose"], palette["coral"], palette["gold"]]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for name, color in zip(params, colors, strict=False):
        sub = df[df["参数中文名"] == name].sort_values("相对最优比例")
        ax.plot(sub["相对最优比例"], sub["RMSE"], color=color, lw=2.1, label=name)
    ax.axvline(1.0, color=palette["deep"], lw=1.0, linestyle="--", alpha=0.75)
    ax.set_title("配色预览：短期模型参数剖面")
    ax.set_xlabel("相对当前参数比例")
    ax.set_ylabel("RMSE")
    ax.legend(loc="upper right")
    return save(fig, "参数剖面.png", palette)


def preview_contact_sheet(paths: list[Path], palette: dict[str, str]) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    axes = axes.ravel()
    for ax, path in zip(axes, paths, strict=False):
        ax.imshow(mpimg.imread(path))
        ax.set_title(path.stem.replace(f"{palette['prefix']}_", ""), color=palette["deep"], fontsize=12, fontweight="bold")
        ax.axis("off")
    for ax in axes[len(paths) :]:
        ax.axis("off")
    return save(fig, "总览.png", palette)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--palette", choices=PALETTES.keys(), default="warm")
    args = parser.parse_args()
    palette = PALETTES[args.palette]
    setup_style(palette)
    paths = [
        preview_short_fit(palette),
        preview_mechanism(palette),
        preview_monte_carlo_fan(palette),
        preview_sensitivity(palette),
        preview_parameter_profile(palette),
    ]
    paths.append(preview_contact_sheet(paths, palette))
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
