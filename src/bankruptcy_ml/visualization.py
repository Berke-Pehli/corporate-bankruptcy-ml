"""Create visual outputs for the corporate bankruptcy ML project.

This module contains plotting functions used across the project. The figures are
designed for the README, final paper, and oral exam discussion.

Inputs:
    - raw or processed pandas DataFrames
    - model validation prediction tables
    - model validation metric tables

Outputs:
    - PNG figures saved under outputs/figures/

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


KEY_MODELS = [
    "Majority-class baseline",
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
]

KEY_CURVE_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
]

CORE_METRICS = [
    "roc_auc",
    "pr_auc",
    "recall_failed",
    "precision_failed",
    "f1_failed",
]

READABLE_METRIC_NAMES = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "recall_failed": "Recall failed",
    "precision_failed": "Precision failed",
    "f1_failed": "F1 failed",
    "balanced_accuracy": "Balanced accuracy",
}


def _filter_models_in_order(
    data: pd.DataFrame,
    models: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Filter a model table and preserve a clean plotting order.

    Args:
        data: DataFrame containing a 'model' column.
        models: Optional list of model names to keep and order.

    Returns:
        A tuple containing the filtered DataFrame and the ordered model names
        that are present in the data.
    """
    if models is None:
        ordered_models = list(data["model"].unique())
        return data.copy(), ordered_models

    available_models = set(data["model"])
    ordered_models = [model for model in models if model in available_models]
    filtered = data[data["model"].isin(ordered_models)].copy()

    return filtered, ordered_models


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
    total = class_counts.sum()
    class_shares = class_counts / total

    fig, ax = plt.subplots(figsize=(8, 5.5))
    class_counts.plot(kind="bar", ax=ax)

    ax.set_title("Class Balance in the Raw Bankruptcy Dataset")
    ax.set_xlabel("Status label")
    ax.set_ylabel("Number of company-year observations")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, class_counts.max() * 1.15)

    for index, (count, share) in enumerate(
        zip(class_counts, class_shares, strict=False),
    ):
        ax.text(
            index,
            count + class_counts.max() * 0.015,
            f"{count:,}\n({share:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_roc_curves(
    predictions: pd.DataFrame,
    output_path: Path,
    title: str,
    models: list[str] | None = None,
) -> None:
    """Plot ROC curves for selected models.

    Args:
        predictions: Prediction table with actual labels and predicted
            probabilities.
        output_path: Path where the figure should be saved.
        title: Figure title.
        models: Optional list of models to include.
    """
    plot_data, ordered_models = _filter_models_in_order(predictions, models=models)

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name in ordered_models:
        model_predictions = plot_data[plot_data["model"] == model_name]
        RocCurveDisplay.from_predictions(
            y_true=model_predictions["actual_failed"],
            y_score=model_predictions["probability_failed"],
            name=model_name,
            ax=ax,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="No-skill line")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_precision_recall_curves(
    predictions: pd.DataFrame,
    output_path: Path,
    title: str,
    models: list[str] | None = None,
) -> None:
    """Plot precision-recall curves for selected models.

    Args:
        predictions: Prediction table with actual labels and predicted
            probabilities.
        output_path: Path where the figure should be saved.
        title: Figure title.
        models: Optional list of models to include.
    """
    plot_data, ordered_models = _filter_models_in_order(predictions, models=models)

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name in ordered_models:
        model_predictions = plot_data[plot_data["model"] == model_name]
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

    ax.set_title(title)
    ax.set_xlabel("Recall for failed firms")
    ax.set_ylabel("Precision for failed firms")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_confusion_matrices(
    predictions: pd.DataFrame,
    output_path: Path,
    title: str,
    models: list[str] | None = None,
) -> None:
    """Plot confusion matrices for selected models.

    Args:
        predictions: Prediction table with actual and predicted labels.
        output_path: Path where the figure should be saved.
        title: Figure title.
        models: Optional list of models to include.
    """
    plot_data, ordered_models = _filter_models_in_order(predictions, models=models)

    n_models = len(ordered_models)
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))
    axes = axes.flatten()

    for ax, model_name in zip(axes, ordered_models, strict=False):
        model_predictions = plot_data[plot_data["model"] == model_name]

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

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_metric_comparison(
    model_comparison: pd.DataFrame,
    output_path: Path,
    title: str,
    metrics_to_plot: list[str],
    models: list[str] | None = None,
) -> None:
    """Plot selected model metrics.

    Args:
        model_comparison: Model metric table.
        output_path: Path where the figure should be saved.
        title: Figure title.
        metrics_to_plot: Metrics to include.
        models: Optional list of models to include.
    """
    plot_data, ordered_models = _filter_models_in_order(
        model_comparison,
        models=models,
    )

    plot_data = plot_data.set_index("model").reindex(ordered_models)[metrics_to_plot]
    plot_data = plot_data.rename(columns=READABLE_METRIC_NAMES)

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_data.plot(kind="bar", ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_validation_roc_curves(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot validation ROC curves for all fitted models."""
    _plot_roc_curves(
        predictions=predictions,
        output_path=output_path,
        title="Validation ROC Curves",
    )


def plot_validation_precision_recall_curves(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot validation precision-recall curves for all fitted models."""
    _plot_precision_recall_curves(
        predictions=predictions,
        output_path=output_path,
        title="Validation Precision-Recall Curves",
    )


def plot_validation_confusion_matrices(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot validation confusion matrices for all fitted models."""
    _plot_confusion_matrices(
        predictions=predictions,
        output_path=output_path,
        title="Validation Confusion Matrices",
    )


def plot_validation_metric_comparison(
    model_comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot key validation metrics across models."""
    metrics_to_plot = [
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "precision_failed",
        "recall_failed",
        "f1_failed",
    ]

    _plot_metric_comparison(
        model_comparison=model_comparison,
        output_path=output_path,
        title="Validation Metric Comparison",
        metrics_to_plot=metrics_to_plot,
    )


def plot_final_test_roc_curves(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot final test ROC curves for all fitted models."""
    _plot_roc_curves(
        predictions=predictions,
        output_path=output_path,
        title="Final Test ROC Curves",
    )


def plot_final_test_precision_recall_curves(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot final test precision-recall curves for all fitted models."""
    _plot_precision_recall_curves(
        predictions=predictions,
        output_path=output_path,
        title="Final Test Precision-Recall Curves",
    )


def plot_final_test_confusion_matrices(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot final test confusion matrices for all fitted models."""
    _plot_confusion_matrices(
        predictions=predictions,
        output_path=output_path,
        title="Final Test Confusion Matrices",
    )


def plot_final_test_metric_comparison(
    model_comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot key final test metrics across models."""
    metrics_to_plot = [
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "precision_failed",
        "recall_failed",
        "f1_failed",
    ]

    _plot_metric_comparison(
        model_comparison=model_comparison,
        output_path=output_path,
        title="Final Test Metric Comparison",
        metrics_to_plot=metrics_to_plot,
    )


def plot_final_test_roc_curves_key_models(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot final test ROC curves for the key non-baseline models.

    The majority-class baseline is excluded from this simplified curve because
    the no-skill diagonal already provides the relevant baseline reference.
    """
    _plot_roc_curves(
        predictions=predictions,
        output_path=output_path,
        title="Final Test ROC Curves (Key Models)",
        models=KEY_CURVE_MODELS,
    )


def plot_final_test_precision_recall_curves_key_models(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot final test precision-recall curves for the key non-baseline models.

    The majority-class baseline curve is excluded because it creates a visually
    distracting vertical line. The dashed failure-rate line remains as the
    no-skill baseline for precision-recall analysis.
    """
    _plot_precision_recall_curves(
        predictions=predictions,
        output_path=output_path,
        title="Final Test Precision-Recall Curves (Key Models)",
        models=KEY_CURVE_MODELS,
    )


def plot_final_test_confusion_matrices_key_models(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot final test confusion matrices only for the key models.

    The majority-class baseline is kept here because the confusion matrix makes
    its weakness very clear: it predicts all observations as alive and misses all
    failed firms.
    """
    _plot_confusion_matrices(
        predictions=predictions,
        output_path=output_path,
        title="Final Test Confusion Matrices (Key Models)",
        models=KEY_MODELS,
    )


def plot_final_test_core_metric_summary(
    model_comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot a cleaner final-test metric summary for the key models.

    This figure is intended for the README, report, and oral exam. It excludes
    balanced accuracy and keeps only the most interpretable metrics for the
    minority bankruptcy class.
    """
    _plot_metric_comparison(
        model_comparison=model_comparison,
        output_path=output_path,
        title="Final Test Core Metric Summary",
        metrics_to_plot=CORE_METRICS,
        models=KEY_MODELS,
    )