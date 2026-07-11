"""Tests for bankruptcy model evaluation utilities."""

import numpy as np
import pandas as pd

from bankruptcy_ml.evaluation import (
    PREDICTION_TABLE_COLUMNS,
    create_prediction_table,
    evaluate_binary_classifier,
    save_prediction_table,
)


def test_evaluate_binary_classifier_returns_expected_metrics() -> None:
    """Check that evaluation returns key metrics and confusion components."""
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0])
    probability_failed = np.array([0.1, 0.2, 0.8, 0.4])

    metrics = evaluate_binary_classifier(
        model_name="test_model",
        y_true=y_true,
        y_pred=y_pred,
        probability_failed=probability_failed,
    )

    assert metrics["model"] == "test_model"
    assert metrics["true_negative"] == 2
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 1
    assert metrics["true_positive"] == 1
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics


def test_create_prediction_table_contains_identifiers_and_predictions() -> None:
    """Check that prediction table keeps identifiers and model outputs."""
    validation_data = pd.DataFrame(
        {
            "company_name": ["A", "B"],
            "year": [2000, 2001],
            "failed": [0, 1],
        }
    )
    y_pred = np.array([0, 1])
    probability_failed = np.array([0.2, 0.9])

    result = create_prediction_table(
        validation_data=validation_data,
        model_name="test_model",
        y_pred=y_pred,
        probability_failed=probability_failed,
    )

    assert list(result["company_name"]) == ["A", "B"]
    assert list(result["predicted_failed"]) == [0, 1]
    assert list(result["probability_failed"]) == [0.2, 0.9]
    assert list(result.columns) == PREDICTION_TABLE_COLUMNS


def test_save_prediction_table_is_byte_stable_and_preserves_labels(tmp_path) -> None:
    """Check deterministic serialization without changing labels or metrics input."""
    predictions = pd.DataFrame(
        {
            "model": ["Random Forest", "Random Forest"],
            "company_name": ["A", "B"],
            "year": [2000, 2001],
            "actual_failed": [0, 1],
            "predicted_failed": [0, 1],
            "probability_failed": [
                0.5510659311188297,
                0.5510659311188298,
            ],
        }
    )

    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"

    save_prediction_table(predictions, first_path)
    save_prediction_table(predictions, second_path)

    first_content = first_path.read_bytes()
    second_content = second_path.read_bytes()
    reloaded = pd.read_csv(first_path)

    assert first_content == second_content
    assert list(reloaded.columns) == PREDICTION_TABLE_COLUMNS
    assert reloaded["predicted_failed"].tolist() == predictions[
        "predicted_failed"
    ].tolist()
    assert reloaded["probability_failed"].between(0, 1).all()
    assert reloaded["probability_failed"].tolist() == [
        0.55106593112,
        0.55106593112,
    ]
