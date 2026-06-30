"""Baseline models for corporate bankruptcy prediction.

This module defines simple benchmark models used to evaluate whether machine
learning models add predictive value. The most important benchmark is the
majority-class baseline, which always predicts the most common class.

Inputs:
    - data/processed/train.csv

Outputs:
    - fitted baseline model objects saved under outputs/models/

Purpose:
    Bankruptcy prediction is an imbalanced classification problem. A model can
    achieve high accuracy by predicting that every company remains alive. This
    baseline makes that issue visible and motivates the use of metrics such as
    balanced accuracy, recall, F1-score, ROC-AUC, and PR-AUC.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier


def build_majority_class_baseline() -> DummyClassifier:
    """Create a majority-class baseline classifier.

    The model always predicts the most frequent class observed in the training
    data. For this dataset, that class is expected to be ``alive`` because failed
    firms are much less frequent.

    Returns:
        An unfitted scikit-learn DummyClassifier.
    """
    return DummyClassifier(strategy="most_frequent")