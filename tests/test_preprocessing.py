"""Tests for preprocessing the raw bankruptcy dataset.

These tests check that the binary target is created correctly and that the
modeling dataset keeps only the columns needed for splitting and modeling.
"""

import pandas as pd

from bankruptcy_ml.config import COMPANY_COLUMN, TARGET_COLUMN, YEAR_COLUMN
from bankruptcy_ml.preprocessing import create_model_dataset, encode_target


def test_encode_target_maps_failed_to_one_and_alive_to_zero() -> None:
    """Check that raw status labels are encoded as expected."""
    data = pd.DataFrame(
        {
            "company_name": ["A", "B"],
            "status_label": ["alive", "failed"],
            "year": [2000, 2000],
            "X1": [1.0, 2.0],
        }
    )

    result = encode_target(data)

    assert result.loc[0, TARGET_COLUMN] == 0
    assert result.loc[1, TARGET_COLUMN] == 1


def test_create_model_dataset_keeps_identifiers_target_and_features() -> None:
    """Check that model dataset contains identifiers, target, and predictors."""
    data = pd.DataFrame(
        {
            "company_name": ["A", "B", "C"],
            "status_label": ["alive", "failed", "alive"],
            "year": [2000, 2001, 2002],
            "X1": [1.0, 2.0, 3.0],
            "X2": [5.0, 5.0, 5.0],
        }
    )

    model_dataset, removed_columns = create_model_dataset(data)

    assert COMPANY_COLUMN in model_dataset.columns
    assert YEAR_COLUMN in model_dataset.columns
    assert TARGET_COLUMN in model_dataset.columns
    assert "X1" in model_dataset.columns
    assert "X2" not in model_dataset.columns
    assert removed_columns == ["X2"]