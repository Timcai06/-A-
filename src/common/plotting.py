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
    "ink": "#111827",
    "muted": "#4b5563",
    "blue": "#1f4e79",
    "teal": "#0f766e",
    "red": "#b91c1c",
    "amber": "#b45309",
    "grid": "#e5e7eb",
    "border": "#cbd5e1",
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
            "lines.linewidth": 1.8,
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": PAPER_COLORS["border"],
            "legend.facecolor": "white",
            "patch.edgecolor": "white",
        }
    )
