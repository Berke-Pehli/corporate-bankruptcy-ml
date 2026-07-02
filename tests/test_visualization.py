"""Tests for validation visualization utilities."""

import pandas as pd

from bankruptcy_ml.visualization import (
    plot_validation_metric_comparison,
    plot_validation_precision_recall_curves,
)


def test_plot_validation_precision_recall_curves_creates_file(tmp_path) -> None:
    """Check that the precision-recall plotting function writes a file."""
    predictions = pd.DataFrame(
        {
            "model": ["Model A", "Model A", "Model B", "Model B"],
            "actual_failed": [0, 1, 0, 1],
            "predicted_failed": [0, 1, 0, 0],
            "probability_failed": [0.1, 0.8, 0.2, 0.4],
        }
    )
    output_path = tmp_path / "precision_recall.png"

    plot_validation_precision_recall_curves(
        predictions=predictions,
        output_path=output_path,
    )

    assert output_path.exists()


def test_plot_validation_metric_comparison_creates_file(tmp_path) -> None:
    """Check that the metric comparison plotting function writes a file."""
    model_comparison = pd.DataFrame(
        {
            "model": ["Model A", "Model B"],
            "balanced_accuracy": [0.6, 0.7],
            "roc_auc": [0.65, 0.75],
            "pr_auc": [0.2, 0.3],
            "precision_failed": [0.1, 0.2],
            "recall_failed": [0.8, 0.5],
            "f1_failed": [0.18, 0.28],
        }
    )
    output_path = tmp_path / "metrics.png"

    plot_validation_metric_comparison(
        model_comparison=model_comparison,
        output_path=output_path,
    )

    assert output_path.exists()