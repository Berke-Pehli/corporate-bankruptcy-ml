"""Create visual outputs for the corporate bankruptcy ML project.

This module contains plotting functions used across the project. The first
visualization documents the class imbalance in the raw bankruptcy dataset. Later
phases will add model evaluation plots such as ROC curves, precision-recall
curves, and confusion matrices.

Inputs:
    - raw or processed pandas DataFrames
    - model evaluation tables

Outputs:
    - PNG figures saved under outputs/figures/

Current figures:
    - class_balance.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bankruptcy_ml.config import RAW_TARGET_COLUMN


def plot_class_balance(data: pd.DataFrame, output_path: Path) -> None:
    """Plot the raw target class distribution.

    Bankruptcy prediction is an imbalanced classification problem. This figure
    shows the number of alive and failed company-year observations and makes it
    clear why accuracy alone is not a sufficient evaluation metric.

    Args:
        data: Raw bankruptcy dataset.
        output_path: Path where the class-balance figure should be saved.
    """
    class_counts = data[RAW_TARGET_COLUMN].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    class_counts.plot(kind="bar", ax=ax)

    ax.set_title("Class Balance in the Raw Bankruptcy Dataset")
    ax.set_xlabel("Status label")
    ax.set_ylabel("Number of company-year observations")
    ax.tick_params(axis="x", rotation=0)

    for index, value in enumerate(class_counts):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)