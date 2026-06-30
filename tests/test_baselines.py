"""Tests for baseline bankruptcy prediction models."""

import pandas as pd

from bankruptcy_ml.baselines import build_majority_class_baseline


def test_majority_class_baseline_predicts_most_frequent_class() -> None:
    """Check that the majority baseline predicts the most common class."""
    x = pd.DataFrame({"X1": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, 0, 0, 1])

    model = build_majority_class_baseline()
    model.fit(x, y)

    predictions = model.predict(x)

    assert set(predictions) == {0}