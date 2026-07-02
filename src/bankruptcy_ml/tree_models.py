"""Tree-based models for corporate bankruptcy prediction.

This module builds and selects tree-based classification models for the
corporate bankruptcy prediction project. Tree-based methods are useful because
they can capture nonlinear relationships and interactions between financial
statement variables.

Inputs:
    - data/processed/train.csv

Outputs:
    - fitted tree-based models saved under outputs/models/

Models:
    - Decision Tree
    - Random Forest
    - Gradient Boosting

Course connection:
    The Decision Tree represents a simple and interpretable tree-based model,
    but individual trees can have high variance. Random Forest reduces this
    variance by averaging many decorrelated trees. Gradient Boosting builds trees
    sequentially and tries to correct earlier prediction errors.

Important interpretation note:
    Tree feature importances are predictive importance measures. They should not
    be interpreted as causal effects of financial variables on bankruptcy.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from bankruptcy_ml.config import RANDOM_STATE


def build_decision_tree(
    max_depth: int | None = 3,
    min_samples_leaf: int = 100,
) -> DecisionTreeClassifier:
    """Build a Decision Tree classifier.

    The tree depth and minimum leaf size control the bias-variance trade-off.
    Smaller trees are easier to interpret and less likely to overfit.

    Args:
        max_depth: Maximum depth of the tree.
        min_samples_leaf: Minimum number of observations required in each leaf.

    Returns:
        An unfitted DecisionTreeClassifier.
    """
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def build_random_forest(
    n_estimators: int = 300,
    max_depth: int | None = 5,
    min_samples_leaf: int = 50,
    max_features: str | None = "sqrt",
) -> RandomForestClassifier:
    """Build a Random Forest classifier.

    Random Forest averages many decision trees and reduces variance by using
    bootstrapped samples and random subsets of predictors at each split.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each tree.
        min_samples_leaf: Minimum number of observations required in each leaf.
        max_features: Number of features considered at each split.

    Returns:
        An unfitted RandomForestClassifier.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_gradient_boosting(
    n_estimators: int = 150,
    learning_rate: float = 0.05,
    max_depth: int = 2,
    min_samples_leaf: int = 100,
) -> GradientBoostingClassifier:
    """Build a Gradient Boosting classifier.

    Gradient Boosting builds trees sequentially. Each new tree focuses on
    correcting errors from the previous ensemble. The learning rate and tree
    depth control model flexibility and overfitting risk.

    Args:
        n_estimators: Number of boosting stages.
        learning_rate: Shrinkage parameter applied to each tree contribution.
        max_depth: Maximum depth of each weak learner.
        min_samples_leaf: Minimum number of observations required in each leaf.

    Returns:
        An unfitted GradientBoostingClassifier.
    """
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE,
    )


def select_decision_tree(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[DecisionTreeClassifier, dict[str, Any], float]:
    """Select a Decision Tree using validation PR-AUC.

    Args:
        x_train: Training feature matrix.
        y_train: Training target vector.
        x_valid: Validation feature matrix.
        y_valid: Validation target vector.

    Returns:
        A tuple containing the best fitted model, selected parameters, and
        validation PR-AUC.
    """
    parameter_grid = {
        "max_depth": [2, 3, 4, 5],
        "min_samples_leaf": [50, 100, 200],
    }

    best_model = None
    best_params = None
    best_score = -1.0

    for max_depth, min_samples_leaf in product(
        parameter_grid["max_depth"],
        parameter_grid["min_samples_leaf"],
    ):
        candidate = build_decision_tree(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        candidate.fit(x_train, y_train)

        probability_failed = candidate.predict_proba(x_valid)[:, 1]
        score = average_precision_score(y_valid, probability_failed)

        if score > best_score:
            best_model = candidate
            best_params = {
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
            }
            best_score = score

    if best_model is None or best_params is None:
        msg = "No Decision Tree model was selected."
        raise RuntimeError(msg)

    return best_model, best_params, best_score


def select_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[RandomForestClassifier, dict[str, Any], float]:
    """Select a Random Forest using validation PR-AUC.

    Args:
        x_train: Training feature matrix.
        y_train: Training target vector.
        x_valid: Validation feature matrix.
        y_valid: Validation target vector.

    Returns:
        A tuple containing the best fitted model, selected parameters, and
        validation PR-AUC.
    """
    parameter_grid = {
        "n_estimators": [300],
        "max_depth": [4, 6, None],
        "min_samples_leaf": [50, 100],
        "max_features": ["sqrt"],
    }

    best_model = None
    best_params = None
    best_score = -1.0

    for n_estimators, max_depth, min_samples_leaf, max_features in product(
        parameter_grid["n_estimators"],
        parameter_grid["max_depth"],
        parameter_grid["min_samples_leaf"],
        parameter_grid["max_features"],
    ):
        candidate = build_random_forest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )
        candidate.fit(x_train, y_train)

        probability_failed = candidate.predict_proba(x_valid)[:, 1]
        score = average_precision_score(y_valid, probability_failed)

        if score > best_score:
            best_model = candidate
            best_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
            }
            best_score = score

    if best_model is None or best_params is None:
        msg = "No Random Forest model was selected."
        raise RuntimeError(msg)

    return best_model, best_params, best_score


def select_gradient_boosting(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[GradientBoostingClassifier, dict[str, Any], float]:
    """Select a Gradient Boosting model using validation PR-AUC.

    GradientBoostingClassifier does not have a ``class_weight`` parameter.
    Therefore, this function uses sample weights to give more importance to the
    minority bankruptcy class during fitting.

    Args:
        x_train: Training feature matrix.
        y_train: Training target vector.
        x_valid: Validation feature matrix.
        y_valid: Validation target vector.

    Returns:
        A tuple containing the best fitted model, selected parameters, and
        validation PR-AUC.
    """
    parameter_grid = {
        "n_estimators": [100, 150],
        "learning_rate": [0.03, 0.05],
        "max_depth": [2, 3],
        "min_samples_leaf": [100],
    }

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    best_model = None
    best_params = None
    best_score = -1.0

    for n_estimators, learning_rate, max_depth, min_samples_leaf in product(
        parameter_grid["n_estimators"],
        parameter_grid["learning_rate"],
        parameter_grid["max_depth"],
        parameter_grid["min_samples_leaf"],
    ):
        candidate = build_gradient_boosting(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        candidate.fit(x_train, y_train, sample_weight=sample_weight)

        probability_failed = candidate.predict_proba(x_valid)[:, 1]
        score = average_precision_score(y_valid, probability_failed)

        if score > best_score:
            best_model = candidate
            best_params = {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
            }
            best_score = score

    if best_model is None or best_params is None:
        msg = "No Gradient Boosting model was selected."
        raise RuntimeError(msg)

    return best_model, best_params, best_score