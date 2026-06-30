"""Evaluation utilities for bankruptcy classification models.

This module computes classification metrics and prediction tables for the
bankruptcy prediction project. Because the bankruptcy class is rare, the project
does not rely on accuracy alone.

Inputs:
    - fitted classifiers
    - validation or test feature matrices
    - validation or test target vectors

Outputs:
    - model comparison tables
    - classification report tables
    - prediction tables

Evaluation metrics:
    - Accuracy
    - Balanced accuracy
    - ROC-AUC
    - PR-AUC
    - Precision for the bankruptcy class
    - Recall for the bankruptcy class
    - F1-score for the bankruptcy class
    - Confusion matrix components
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from bankruptcy_ml.config import COMPANY_COLUMN, TARGET_COLUMN, YEAR_COLUMN


def get_probability_failed(model, x: pd.DataFrame) -> np.ndarray:
    """Return predicted probabilities for the bankruptcy class.

    Args:
        model: Fitted classifier with ``predict_proba``.
        x: Feature matrix.

    Returns:
        A NumPy array with predicted probabilities for class 1.
    """
    probabilities = model.predict_proba(x)
    return probabilities[:, 1]


def evaluate_binary_classifier(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    probability_failed: np.ndarray,
) -> dict[str, float | int | str]:
    """Compute binary classification metrics for one model.

    Args:
        model_name: Human-readable model name.
        y_true: True binary target values.
        y_pred: Predicted binary class labels.
        probability_failed: Predicted probabilities for class 1.

    Returns:
        A dictionary containing classification metrics and confusion matrix
        components.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, probability_failed),
        "pr_auc": average_precision_score(y_true, probability_failed),
        "precision_failed": precision_score(y_true, y_pred, zero_division=0),
        "recall_failed": recall_score(y_true, y_pred, zero_division=0),
        "f1_failed": f1_score(y_true, y_pred, zero_division=0),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def create_classification_report_table(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Create a flat classification report table for one model.

    Args:
        model_name: Human-readable model name.
        y_true: True binary target values.
        y_pred: Predicted binary class labels.

    Returns:
        A DataFrame version of scikit-learn's classification report.
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["alive", "failed"],
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            rows.append(
                {
                    "model": model_name,
                    "class_or_average": label,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def create_prediction_table(
    validation_data: pd.DataFrame,
    model_name: str,
    y_pred: np.ndarray,
    probability_failed: np.ndarray,
) -> pd.DataFrame:
    """Create a validation prediction table for one model.

    Args:
        validation_data: Validation DataFrame containing identifiers and target.
        model_name: Human-readable model name.
        y_pred: Predicted binary class labels.
        probability_failed: Predicted probabilities for class 1.

    Returns:
        A DataFrame containing identifiers, actual labels, predictions, and
        predicted bankruptcy probabilities.
    """
    return pd.DataFrame(
        {
            "model": model_name,
            COMPANY_COLUMN: validation_data[COMPANY_COLUMN].to_numpy(),
            YEAR_COLUMN: validation_data[YEAR_COLUMN].to_numpy(),
            "actual_failed": validation_data[TARGET_COLUMN].to_numpy(),
            "predicted_failed": y_pred,
            "probability_failed": probability_failed,
        }
    )