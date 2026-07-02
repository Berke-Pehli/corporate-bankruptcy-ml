"""Tests for tree-based bankruptcy prediction models."""

import pandas as pd

from bankruptcy_ml.tree_models import (
    build_decision_tree,
    build_gradient_boosting,
    build_random_forest,
)


def test_build_decision_tree_can_fit_small_dataset() -> None:
    """Check that the Decision Tree model can fit a small classification dataset."""
    x = pd.DataFrame({"X1": [1.0, 2.0, 3.0, 4.0], "X2": [4.0, 3.0, 2.0, 1.0]})
    y = pd.Series([0, 0, 1, 1])

    model = build_decision_tree(max_depth=2, min_samples_leaf=1)
    model.fit(x, y)

    predictions = model.predict(x)

    assert len(predictions) == len(y)


def test_build_random_forest_can_fit_small_dataset() -> None:
    """Check that the Random Forest model can fit a small classification dataset."""
    x = pd.DataFrame({"X1": [1.0, 2.0, 3.0, 4.0], "X2": [4.0, 3.0, 2.0, 1.0]})
    y = pd.Series([0, 0, 1, 1])

    model = build_random_forest(
        n_estimators=10,
        max_depth=2,
        min_samples_leaf=1,
        max_features="sqrt",
    )
    model.fit(x, y)

    predictions = model.predict(x)

    assert len(predictions) == len(y)


def test_build_gradient_boosting_can_fit_small_dataset() -> None:
    """Check that the Gradient Boosting model can fit a small classification dataset."""
    x = pd.DataFrame({"X1": [1.0, 2.0, 3.0, 4.0], "X2": [4.0, 3.0, 2.0, 1.0]})
    y = pd.Series([0, 0, 1, 1])

    model = build_gradient_boosting(
        n_estimators=5,
        learning_rate=0.1,
        max_depth=1,
        min_samples_leaf=1,
    )
    model.fit(x, y)

    predictions = model.predict(x)

    assert len(predictions) == len(y)