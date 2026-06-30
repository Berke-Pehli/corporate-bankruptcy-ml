"""Tests for leakage-safe company-level splitting.

The project uses repeated company-year observations, so the same company must
never appear in both the training and test sets.
"""

import pandas as pd

from bankruptcy_ml.config import COMPANY_COLUMN
from bankruptcy_ml.splitting import (
    check_no_company_overlap,
    create_company_level_split,
    create_split_summary,
)


def test_company_level_split_has_no_company_overlap() -> None:
    """Check that no company appears in both train and test sets."""
    data = pd.DataFrame(
        {
            "company_name": [f"C{i}" for i in range(20) for _ in range(2)],
            "year": [2000, 2001] * 20,
            "failed": [0, 1] * 20,
            "X1": range(40),
        }
    )

    train_data, test_data = create_company_level_split(
        data,
        test_size=0.25,
        random_state=42,
    )

    assert check_no_company_overlap(train_data, test_data)


def test_check_no_company_overlap_detects_leakage() -> None:
    """Check that company overlap is detected when it exists."""
    train_data = pd.DataFrame({COMPANY_COLUMN: ["A", "B", "C"]})
    test_data = pd.DataFrame({COMPANY_COLUMN: ["C", "D"]})

    assert not check_no_company_overlap(train_data, test_data)


def test_create_split_summary_returns_full_train_and_test_rows() -> None:
    """Check that the split summary contains full, train, and test rows."""
    full_data = pd.DataFrame(
        {
            "company_name": ["A", "A", "B", "B"],
            "failed": [0, 0, 1, 1],
            "X1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    train_data = full_data.iloc[:2]
    test_data = full_data.iloc[2:]

    summary = create_split_summary(full_data, train_data, test_data)

    assert set(summary["split"]) == {"full", "train", "test"}