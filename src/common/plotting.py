"""Shared Matplotlib style helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt


DEFAULT_CHINESE_FONTS = [
    "Arial Unicode MS",
    "Hiragino Sans GB",
    "Songti SC",
    "Heiti TC",
    "DejaVu Sans",
]

PAPER_COLORS = {
    "ink": "#1A1A1A",
    "muted": "#4b5563",
    "blue": "#2B6A99",
    "navy": "#1F4E79",
    "teal": "#1B9E77",
    "green": "#1B9E77",
    "red": "#B2182B",
    "brick": "#B2182B",
    "amber": "#b45309",
    "band": "#8DA0CB",
    "band_light": "#D8E2F0",
    "grid": "#e5e7eb",
    "border": "#cbd5e1",
}

SCENARIO_COLORS = {
    "actual": PAPER_COLORS["ink"],
    "fit": PAPER_COLORS["blue"],
    "neutral": PAPER_COLORS["blue"],
    "optimistic": PAPER_COLORS["teal"],
    "pessimistic": PAPER_COLORS["brick"],
    "risk": PAPER_COLORS["brick"],
    "muted": PAPER_COLORS["muted"],
    "band_outer": PAPER_COLORS["band_light"],
    "band_inner": PAPER_COLORS["band"],
}


def configure_plot_style(savefig_dpi: int = 180, figure_dpi: int = 150, title_size: int = 13) -> None:
    """Apply a consistent Chinese-compatible Matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": DEFAULT_CHINESE_FONTS,
            "axes.unicode_minus": False,
            "figure.dpi": figure_dpi,
            "savefig.dpi": savefig_dpi,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.titlesize": title_size,
            "axes.titleweight": "bold",
            "axes.titlecolor": PAPER_COLORS["ink"],
            "axes.labelsize": 10,
            "axes.labelcolor": PAPER_COLORS["muted"],
            "axes.edgecolor": PAPER_COLORS["border"],
            "axes.linewidth": 0.8,
            "xtick.color": PAPER_COLORS["muted"],
            "ytick.color": PAPER_COLORS["muted"],
            "grid.color": PAPER_COLORS["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.85,
            "lines.linewidth": 2.0,
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.framealpha": 0.88,
            "legend.edgecolor": PAPER_COLORS["border"],
            "legend.facecolor": "white",
            "patch.edgecolor": "white",
        }
    )


def direct_label(ax, x, y, text: str, color: str, *, dx: float = 0.0, dy: float = 0.0, size: float = 9.2) -> None:
    """Place a small same-color label near a plotted line."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=size,
        color=color,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
    )
