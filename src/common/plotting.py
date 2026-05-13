"""Shared Matplotlib style helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

from src.common.paths import PROJECT_ROOT


DEFAULT_CHINESE_FONTS = [
    "Arial Unicode MS",
    "Hiragino Sans GB",
    "Songti SC",
    "Heiti TC",
    "DejaVu Sans",
]

STYLE_CONFIG_PATH = PROJECT_ROOT / "config" / "figure_style.yml"


def _load_style_config(path: Path = STYLE_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_palette(config: dict[str, Any]) -> dict[str, str]:
    palette = config.get("palette", {})
    neutrals = config.get("neutrals", {})
    colors = {
        "lime": "#55ff37",
        "emerald": "#00e978",
        "teal": "#00cfae",
        "cyan": "#00b3d9",
        "sky": "#0095f2",
        "blue": "#0074f2",
        "ink": "#1A1A1A",
        "muted": "#4b5563",
        "grid": "#e5e7eb",
        "border": "#cbd5e1",
        "band_light": "#DDF8FF",
        "band_mid": "#99E7F4",
    }
    colors.update({key: str(value) for key, value in palette.items()})
    colors.update({key: str(value) for key, value in neutrals.items()})
    return colors


_STYLE_CONFIG = _load_style_config()
_COLORS = _resolve_palette(_STYLE_CONFIG)
_ROLES = _STYLE_CONFIG.get("roles", {})

PAPER_COLORS = {
    "ink": _COLORS["ink"],
    "muted": _COLORS["muted"],
    "blue": _COLORS["blue"],
    "navy": _COLORS["blue"],
    "teal": _COLORS["teal"],
    "green": _COLORS["emerald"],
    "emerald": _COLORS["emerald"],
    "cyan": _COLORS["cyan"],
    "sky": _COLORS["sky"],
    "lime": _COLORS["lime"],
    "red": _COLORS["sky"],
    "brick": _COLORS["blue"],
    "amber": _COLORS["teal"],
    "band": _COLORS["band_mid"],
    "band_light": _COLORS["band_light"],
    "grid": _COLORS["grid"],
    "border": _COLORS["border"],
}


def role_color(role: str, fallback: str | None = None) -> str:
    """Return a configured semantic color role."""
    color_key = _ROLES.get(role)
    if color_key in _COLORS:
        return _COLORS[color_key]
    if fallback is not None:
        return fallback
    return PAPER_COLORS["muted"]

SCENARIO_COLORS = {
    "actual": role_color("actual", PAPER_COLORS["ink"]),
    "fit": role_color("fit", PAPER_COLORS["blue"]),
    "machine_learning": role_color("machine_learning", PAPER_COLORS["sky"]),
    "neutral": role_color("neutral", PAPER_COLORS["cyan"]),
    "optimistic": role_color("optimistic", PAPER_COLORS["green"]),
    "pessimistic": role_color("pessimistic", PAPER_COLORS["blue"]),
    "risk": role_color("risk", PAPER_COLORS["sky"]),
    "buffer": role_color("buffer", PAPER_COLORS["teal"]),
    "highlight": role_color("highlight", PAPER_COLORS["lime"]),
    "muted": PAPER_COLORS["muted"],
    "band_outer": role_color("band_outer", PAPER_COLORS["band_light"]),
    "band_inner": role_color("band_inner", PAPER_COLORS["band"]),
}

SEABORN_PALETTE = [
    PAPER_COLORS["green"],
    PAPER_COLORS["teal"],
    PAPER_COLORS["cyan"],
    PAPER_COLORS["sky"],
    PAPER_COLORS["blue"],
    PAPER_COLORS["lime"],
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
