"""Tests for final test evaluation utilities."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from bankruptcy_ml.final_evaluation import save_final_test_evaluation


class DummyProbabilityModel:
    """Minimal probability model used to test final evaluation output creation."""

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Return deterministic binary predictions."""
        return np.array([0, 1, 1, 0])

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Return deterministic class probabilities."""
        return np.array(
            [
                [0.90, 0.10],
                [0.20, 0.80],
                [0.30, 0.70],
                [0.60, 0.40],
            ]
        )


def test_save_final_test_evaluation_creates_expected_outputs(tmp_path: Path) -> None:
    """Check that final test evaluation writes metrics, reports, and predictions."""
    test_data = pd.DataFrame(
        {
            "company_name": ["A", "B", "C", "D"],
            "year": [2018, 2018, 2018, 2018],
            "failed": [0, 1, 0, 1],
            "X1": [1.0, 2.0, 3.0, 4.0],
            "X2": [4.0, 3.0, 2.0, 1.0],
        }
    )

    test_data_path = tmp_path / "test.csv"
    model_path = tmp_path / "dummy_model.joblib"
    final_test_metrics_path = tmp_path / "final_test_metrics.csv"
    final_test_classification_reports_path = (
        tmp_path / "final_test_classification_reports.csv"
    )
    final_test_predictions_path = tmp_path / "final_test_predictions.csv"

    test_data.to_csv(test_data_path, index=False)
    joblib.dump(DummyProbabilityModel(), model_path)

    save_final_test_evaluation(
        model_paths={"Dummy Model": model_path},
        test_data_path=test_data_path,
        final_test_metrics_path=final_test_metrics_path,
        final_test_classification_reports_path=final_test_classification_reports_path,
        final_test_predictions_path=final_test_predictions_path,
    )

    metrics = pd.read_csv(final_test_metrics_path)
    classification_reports = pd.read_csv(final_test_classification_reports_path)
    predictions = pd.read_csv(final_test_predictions_path)

    assert final_test_metrics_path.exists()
    assert final_test_classification_reports_path.exists()
    assert final_test_predictions_path.exists()

    assert metrics["model"].tolist() == ["Dummy Model"]
    assert "roc_auc" in metrics.columns
    assert "pr_auc" in metrics.columns
    assert "f1_failed" in metrics.columns

    assert "model" in classification_reports.columns
    assert "class_or_average" in classification_reports.columns

    expected_prediction_columns = {
        "model",
        "company_name",
        "year",
        "actual_failed",
        "predicted_failed",
        "probability_failed",
    }
    assert expected_prediction_columns.issubset(predictions.columns)
    assert predictions["model"].unique().tolist() == ["Dummy Model"]
    assert predictions["actual_failed"].tolist() == [0, 1, 0, 1]
    assert predictions["predicted_failed"].tolist() == [0, 1, 1, 0]
    assert predictions["probability_failed"].tolist() == [0.10, 0.80, 0.70, 0.40]