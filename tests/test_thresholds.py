"""Tests for threshold-analysis utilities."""

import pandas as pd

from bankruptcy_ml.thresholds import create_threshold_analysis, select_thresholds


def test_create_threshold_analysis_returns_metrics_for_each_threshold() -> None:
    """Check that threshold analysis computes precision, recall, and F1."""
    predictions = pd.DataFrame(
        {
            "model": ["Model A"] * 4,
            "actual_failed": [0, 1, 0, 1],
            "probability_failed": [0.1, 0.8, 0.3, 0.7],
        }
    )

    result = create_threshold_analysis(
        predictions=predictions,
        thresholds=[0.2, 0.5],
        model_names=["Model A"],
    )

    assert len(result) == 2
    assert set(result["threshold"]) == {0.2, 0.5}
    assert "precision_failed" in result.columns
    assert "recall_failed" in result.columns
    assert "f1_failed" in result.columns


def test_select_thresholds_returns_max_f1_rule() -> None:
    """Check that selected thresholds include the max-F1 rule."""
    threshold_analysis = pd.DataFrame(
        {
            "model": ["Model A", "Model A", "Model A"],
            "threshold": [0.2, 0.5, 0.8],
            "precision_failed": [0.4, 0.6, 1.0],
            "recall_failed": [1.0, 0.75, 0.25],
            "f1_failed": [0.57, 0.67, 0.40],
            "predicted_failed_share": [0.8, 0.5, 0.1],
            "n_predicted_failed": [8, 5, 1],
        }
    )

    result = select_thresholds(threshold_analysis)

    max_f1 = result[result["selection_rule"] == "maximize_f1"].iloc[0]

    assert max_f1["threshold"] == 0.5