"""pytask tasks for validation evaluation figures.

This task module turns model validation outputs into figures for the README,
final paper, and oral exam discussion.

Run:
    pixi run build

Inputs:
    - outputs/tables/validation_predictions.csv
    - outputs/tables/validation_model_comparison.csv

Outputs:
    - outputs/figures/validation_roc_curves.png
    - outputs/figures/validation_precision_recall_curves.png
    - outputs/figures/validation_confusion_matrices.png
    - outputs/figures/validation_metric_comparison.png
"""

from pathlib import Path

import pandas as pd

from bankruptcy_ml.config import (
    VALIDATION_CONFUSION_MATRICES_PATH,
    VALIDATION_METRIC_COMPARISON_PATH,
    VALIDATION_MODEL_COMPARISON_PATH,
    VALIDATION_PRECISION_RECALL_CURVES_PATH,
    VALIDATION_PREDICTIONS_PATH,
    VALIDATION_ROC_CURVES_PATH,
)
from bankruptcy_ml.visualization import (
    plot_validation_confusion_matrices,
    plot_validation_metric_comparison,
    plot_validation_precision_recall_curves,
    plot_validation_roc_curves,
)


def task_plot_validation_roc_curves(
    depends_on: Path = VALIDATION_PREDICTIONS_PATH,
    produces: Path = VALIDATION_ROC_CURVES_PATH,
) -> None:
    """Create validation ROC curves for all models."""
    predictions = pd.read_csv(depends_on)
    plot_validation_roc_curves(predictions=predictions, output_path=produces)


def task_plot_validation_precision_recall_curves(
    depends_on: Path = VALIDATION_PREDICTIONS_PATH,
    produces: Path = VALIDATION_PRECISION_RECALL_CURVES_PATH,
) -> None:
    """Create validation precision-recall curves for all models."""
    predictions = pd.read_csv(depends_on)
    plot_validation_precision_recall_curves(
        predictions=predictions,
        output_path=produces,
    )


def task_plot_validation_confusion_matrices(
    depends_on: Path = VALIDATION_PREDICTIONS_PATH,
    produces: Path = VALIDATION_CONFUSION_MATRICES_PATH,
) -> None:
    """Create validation confusion matrices for all models."""
    predictions = pd.read_csv(depends_on)
    plot_validation_confusion_matrices(predictions=predictions, output_path=produces)


def task_plot_validation_metric_comparison(
    depends_on: Path = VALIDATION_MODEL_COMPARISON_PATH,
    produces: Path = VALIDATION_METRIC_COMPARISON_PATH,
) -> None:
    """Create validation metric comparison figure."""
    model_comparison = pd.read_csv(depends_on)
    plot_validation_metric_comparison(
        model_comparison=model_comparison,
        output_path=produces,
    )