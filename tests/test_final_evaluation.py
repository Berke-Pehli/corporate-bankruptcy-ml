"""Tests for final test evaluation utilities."""

import numpy as np
import pandas as pd

from bankruptcy_ml.final_evaluation import evaluate_models_on_dataset


class SimpleProbabilityModel:
    """Small test classifier with predict and predict_proba methods."""

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Return binary predictions for the test fixture."""
        return np.array([0, 1, 0, 1])

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for the test fixture."""
        probabilities_failed = np.array([0.1, 0.8, 0.2, 0.9])
        probabilities_alive = 1 - probabilities_failed
        return np.column_stack([probabilities_alive, probabilities_failed])


def test_evaluate_models_on_dataset_returns_three_tables() -> None:
    """Check that final evaluation creates metrics, reports, and predictions."""
    data = pd.DataFrame(
        {
            "company_name": ["A", "B", "C", "D"],
            "year": [2000, 2000, 2001, 2001],
            "failed": [0, 1, 0, 1],
            "X1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    models = {"simple_model": SimpleProbabilityModel()}

    metrics, reports, predictions = evaluate_models_on_dataset(
        models=models,
        data=data,
    )

    assert len(metrics) == 1
    assert not reports.empty
    assert len(predictions) == len(data)
    assert "probability_failed" in predictions.columns