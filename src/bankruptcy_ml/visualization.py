"""Create visual outputs for the corporate bankruptcy ML project.

This module contains plotting functions used across the project. The figures are
designed for the README, final paper, and oral exam discussion.

Inputs:
    - raw or processed pandas DataFrames
    - model validation prediction tables
    - model validation metric tables

Outputs:
    - PNG figures saved under outputs/figures/

Current figures:
    - class_balance.png
    - validation_roc_curves.png
    - validation_precision_recall_curves.png
    - validation_confusion_matrices.png
    - validation_metric_comparison.png

Important interpretation note:
    Bankruptcy prediction is an imbalanced classification problem. Accuracy can
    be misleading, so the project visualizes ROC-AUC, PR-AUC, recall, precision,
    F1-score, and confusion matrices.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

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


def plot_validation_roc_curves(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot validation ROC curves for all fitted models.

    Args:
        predictions: Validation prediction table containing actual labels,
            predicted classes, probabilities, and model names.
        output_path: Path where the ROC curve figure should be saved.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, model_predictions in predictions.groupby("model"):
        RocCurveDisplay.from_predictions(
            y_true=model_predictions["actual_failed"],
            y_score=model_predictions["probability_failed"],
            name=model_name,
            ax=ax,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title("Validation ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_validation_precision_recall_curves(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot validation precision-recall curves for all fitted models.

    Precision-recall curves are especially useful for imbalanced classification
    because they focus directly on the minority bankruptcy class.

    Args:
        predictions: Validation prediction table containing actual labels,
            predicted classes, probabilities, and model names.
        output_path: Path where the precision-recall curve figure should be saved.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, model_predictions in predictions.groupby("model"):
        PrecisionRecallDisplay.from_predictions(
            y_true=model_predictions["actual_failed"],
            y_score=model_predictions["probability_failed"],
            name=model_name,
            ax=ax,
        )

    baseline_rate = predictions["actual_failed"].mean()
    ax.axhline(
        baseline_rate,
        linestyle="--",
        linewidth=1,
        label=f"Failure rate baseline ({baseline_rate:.1%})",
    )

    ax.set_title("Validation Precision-Recall Curves")
    ax.set_xlabel("Recall for failed firms")
    ax.set_ylabel("Precision for failed firms")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_validation_confusion_matrices(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot validation confusion matrices for all fitted models.

    Args:
        predictions: Validation prediction table containing actual labels,
            predicted classes, probabilities, and model names.
        output_path: Path where the confusion matrix figure should be saved.
    """
    model_names = list(predictions["model"].unique())
    n_models = len(model_names)

    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))
    axes = axes.flatten()

    for ax, model_name in zip(axes, model_names, strict=False):
        model_predictions = predictions[predictions["model"] == model_name]

        matrix = confusion_matrix(
            model_predictions["actual_failed"],
            model_predictions["predicted_failed"],
            labels=[0, 1],
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["alive", "failed"],
        )
        display.plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(model_name)

    for unused_ax in axes[n_models:]:
        unused_ax.axis("off")

    fig.suptitle("Validation Confusion Matrices", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_validation_metric_comparison(
    model_comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot key validation metrics across models.

    The figure focuses on metrics that are meaningful for imbalanced bankruptcy
    prediction. Accuracy is intentionally excluded because the majority-class
    baseline already demonstrates why it is misleading.

    Args:
        model_comparison: Table with validation metrics for each model.
        output_path: Path where the metric comparison figure should be saved.
    """
    metrics_to_plot = [
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "precision_failed",
        "recall_failed",
        "f1_failed",
    ]

    plot_data = model_comparison.set_index("model")[metrics_to_plot]

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_data.plot(kind="bar", ax=ax)

    ax.set_title("Validation Metric Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)