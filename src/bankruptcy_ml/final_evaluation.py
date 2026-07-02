"""Evaluate fitted bankruptcy prediction models on the final test set.

This module evaluates the selected models on the untouched final test set. The
final test set was created through a company-level split and was not used during
model selection or validation.

Inputs:
    - data/processed/test.csv
    - fitted model objects saved under outputs/models/

Outputs:
    - outputs/tables/final_test_metrics.csv
    - outputs/tables/final_test_classification_reports.csv
    - outputs/tables/final_test_predictions.csv

Methodological note:
    Validation metrics were used for model comparison and parameter selection.
    Final test metrics are used only after the modeling choices are fixed. This
    avoids evaluating models on the same data used for selection.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from bankruptcy_ml.evaluation import (
    create_classification_report_table,
    create_prediction_table,
    evaluate_binary_classifier,
    get_probability_failed,
)
from bankruptcy_ml.features import split_features_target


def load_models(model_paths: dict[str, Path]) -> dict[str, object]:
    """Load fitted model objects.

    Args:
        model_paths: Dictionary mapping model names to model file paths.

    Returns:
        Dictionary mapping model names to fitted model objects.

    Raises:
        FileNotFoundError: If one or more model files do not exist.
    """
    models = {}

    for model_name, model_path in model_paths.items():
        if not model_path.exists():
            msg = f"Model file not found for {model_name}: {model_path}"
            raise FileNotFoundError(msg)

        models[model_name] = joblib.load(model_path)

    return models


def evaluate_models_on_dataset(
    models: dict[str, object],
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate fitted models on a supplied dataset.

    Args:
        models: Dictionary mapping model names to fitted model objects.
        data: Dataset containing identifiers, target, and feature columns.

    Returns:
        A tuple containing:
            - model metrics table
            - classification report table
            - prediction table
    """
    x_test, y_test = split_features_target(data)

    metric_rows = []
    report_tables = []
    prediction_tables = []

    for model_name, model in models.items():
        y_pred = model.predict(x_test)
        probability_failed = get_probability_failed(model, x_test)

        metric_rows.append(
            evaluate_binary_classifier(
                model_name=model_name,
                y_true=y_test,
                y_pred=y_pred,
                probability_failed=probability_failed,
            )
        )

        report_tables.append(
            create_classification_report_table(
                model_name=model_name,
                y_true=y_test,
                y_pred=y_pred,
            )
        )

        prediction_tables.append(
            create_prediction_table(
                validation_data=data,
                model_name=model_name,
                y_pred=y_pred,
                probability_failed=probability_failed,
            )
        )

    metrics = pd.DataFrame(metric_rows)
    classification_reports = pd.concat(report_tables, ignore_index=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)

    return metrics, classification_reports, predictions


def save_final_test_evaluation(
    model_paths: dict[str, Path],
    test_data_path: Path,
    final_test_metrics_path: Path,
    final_test_classification_reports_path: Path,
    final_test_predictions_path: Path,
) -> None:
    """Evaluate fitted models on the final test set and save result tables.

    Args:
        model_paths: Dictionary mapping model names to fitted model paths.
        test_data_path: Path to final test data.
        final_test_metrics_path: Output path for final test metrics.
        final_test_classification_reports_path: Output path for final test
            classification reports.
        final_test_predictions_path: Output path for final test predictions.
    """
    test_data = pd.read_csv(test_data_path)
    models = load_models(model_paths)

    metrics, classification_reports, predictions = evaluate_models_on_dataset(
        models=models,
        data=test_data,
    )

    final_test_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    final_test_classification_reports_path.parent.mkdir(parents=True, exist_ok=True)
    final_test_predictions_path.parent.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(final_test_metrics_path, index=False)
    classification_reports.to_csv(final_test_classification_reports_path, index=False)
    predictions.to_csv(final_test_predictions_path, index=False)