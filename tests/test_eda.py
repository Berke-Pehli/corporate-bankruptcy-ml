"""Tests for exploratory data analysis utilities."""

import pandas as pd

from bankruptcy_ml.eda import (
    create_annual_failure_rate,
    create_class_feature_summary,
    create_train_test_year_distribution,
)


def test_create_annual_failure_rate_returns_expected_rates() -> None:
    """Check that annual failure rates are computed correctly."""
    data = pd.DataFrame(
        {
            "year": [2000, 2000, 2001, 2001],
            "failed": [0, 1, 0, 0],
            "company_name": ["A", "B", "A", "B"],
            "X1": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = create_annual_failure_rate(data)

    rate_2000 = result.loc[result["year"] == 2000, "failure_rate"].iloc[0]
    rate_2001 = result.loc[result["year"] == 2001, "failure_rate"].iloc[0]

    assert rate_2000 == 0.5
    assert rate_2001 == 0.0


def test_create_train_test_year_distribution_contains_both_splits() -> None:
    """Check that train/test year distribution contains both split labels."""
    train_data = pd.DataFrame(
        {
            "year": [2000, 2000, 2001],
            "company_name": ["A", "B", "C"],
            "failed": [0, 1, 0],
        }
    )
    test_data = pd.DataFrame(
        {
            "year": [2000, 2001],
            "company_name": ["D", "E"],
            "failed": [0, 1],
        }
    )

    result = create_train_test_year_distribution(train_data, test_data)

    assert set(result["split"]) == {"train", "test"}
    assert "share_within_split" in result.columns


def test_create_class_feature_summary_returns_alive_and_failed_rows() -> None:
    """Check that feature summaries are created for both classes."""
    data = pd.DataFrame(
        {
            "failed": [0, 0, 1, 1],
            "X1": [1.0, 2.0, 3.0, 4.0],
            "X8": [10.0, 20.0, 30.0, 40.0],
        }
    )

    result = create_class_feature_summary(data, key_features=["X1", "X8"])

    assert set(result["status"]) == {"alive", "failed"}
    assert set(result["feature"]) == {"X1", "X8"}