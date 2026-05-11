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


def configure_plot_style(savefig_dpi: int = 180, figure_dpi: int = 150, title_size: int = 13) -> None:
    """Apply a consistent Chinese-compatible Matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": DEFAULT_CHINESE_FONTS,
            "axes.unicode_minus": False,
            "figure.dpi": figure_dpi,
            "savefig.dpi": savefig_dpi,
            "axes.titlesize": title_size,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

