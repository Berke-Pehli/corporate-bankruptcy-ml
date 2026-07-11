"""Shared visual styling helpers for project figures.

The project figures are generated from several modules. These small helpers keep
fonts, colors, grids, and saving behavior consistent without changing the
underlying calculations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter

MODEL_COLORS = {
    "Majority-class baseline": "#8c8c8c",
    "Logistic Regression": "#4c78a8",
    "Random Forest": "#59a14f",
    "Gradient Boosting": "#f28e2b",
}

OUTCOME_COLORS = {
    "detected": "#59a14f",
    "missed": "#e15759",
    "false_alarm": "#f28e2b",
}

DIRECTION_COLORS = {
    "positive": "#c44e52",
    "negative": "#4c78a8",
}

METRIC_COLORS = {
    "Precision": "#4c78a8",
    "Recall": "#e15759",
    "F1": "#59a14f",
    "ROC-AUC": "#4c78a8",
    "PR-AUC": "#f28e2b",
    "Failed F1": "#59a14f",
}

BASELINE_COLOR = "#8c8c8c"
GRID_ALPHA = 0.25
FIGURE_DPI = 300


def apply_project_style() -> None:
    """Apply consistent Matplotlib defaults for project-generated figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
        }
    )


def style_axis(
    ax: Axes,
    *,
    grid_axis: str = "y",
    percent_y: bool = False,
    integer_x: bool = False,
) -> None:
    """Apply common axis styling."""
    ax.grid(axis=grid_axis, linestyle="--", alpha=GRID_ALPHA)
    if percent_y:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def save_figure(fig: Figure, output_path: Path) -> None:
    """Save a figure with consistent DPI and tight bounding box."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
