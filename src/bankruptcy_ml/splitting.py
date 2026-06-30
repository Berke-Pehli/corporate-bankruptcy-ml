"""Create leakage-safe train-test splits for company-year bankruptcy data.

The American bankruptcy dataset contains repeated observations for the same
company across multiple years. A random row-level split could place the same
company in both training and test sets, causing data leakage and overstating
out-of-sample performance.

This module therefore splits the data at the company level.

Inputs:
    - data/processed/model_dataset.csv

Outputs:
    - data/processed/train.csv
    - data/processed/test.csv
    - outputs/tables/split_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from bankruptcy_ml.config import COMPANY_COLUMN, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def create_company_level_split(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset so companies cannot appear in both train and test.

    Args:
        data: Model-ready company-year dataset.
        test_size: Share of companies assigned to the test set.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple containing the training and test DataFrames.

    Raises:
        ValueError: If the company identifier or target column is missing.
    """
    required_columns = {COMPANY_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        msg = f"Missing required split columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, test_idx = next(
        splitter.split(
            data,
            y=data[TARGET_COLUMN],
            groups=data[COMPANY_COLUMN],
        )
    )

    train_data = data.iloc[train_idx].copy()
    test_data = data.iloc[test_idx].copy()

    return train_data, test_data


def check_no_company_overlap(train_data: pd.DataFrame, test_data: pd.DataFrame) -> bool:
    """Check whether train and test sets share any company identifiers.

    Args:
        train_data: Training dataset.
        test_data: Test dataset.

    Returns:
        True if no company appears in both sets, otherwise False.
    """
    train_companies = set(train_data[COMPANY_COLUMN].unique())
    test_companies = set(test_data[COMPANY_COLUMN].unique())

    return train_companies.isdisjoint(test_companies)


def create_split_summary(
    full_data: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create a summary table for the company-level train-test split.

    Args:
        full_data: Full model-ready dataset.
        train_data: Training dataset.
        test_data: Test dataset.

    Returns:
        A DataFrame summarizing row counts, company counts, and failure rates.
    """

    def _summarize_split(name: str, split_data: pd.DataFrame) -> dict[str, float | int | str]:
        return {
            "split": name,
            "n_rows": int(len(split_data)),
            "n_companies": int(split_data[COMPANY_COLUMN].nunique()),
            "n_failed_rows": int(split_data[TARGET_COLUMN].sum()),
            "n_alive_rows": int(len(split_data) - split_data[TARGET_COLUMN].sum()),
            "failure_rate": float(split_data[TARGET_COLUMN].mean()),
        }

    summary = pd.DataFrame(
        [
            _summarize_split("full", full_data),
            _summarize_split("train", train_data),
            _summarize_split("test", test_data),
        ]
    )

    summary["company_share"] = summary["n_companies"] / int(
        full_data[COMPANY_COLUMN].nunique()
    )
    summary["row_share"] = summary["n_rows"] / int(len(full_data))

    return summary


def save_company_level_split(
    model_dataset_path: Path,
    train_path: Path,
    test_path: Path,
    split_summary_path: Path,
) -> None:
    """Create and save the leakage-safe company-level train-test split.

    Args:
        model_dataset_path: Path to the cleaned model-ready dataset.
        train_path: Output path for the training data.
        test_path: Output path for the test data.
        split_summary_path: Output path for the split summary table.

    Raises:
        ValueError: If company leakage is detected after splitting.
    """
    data = pd.read_csv(model_dataset_path)
    train_data, test_data = create_company_level_split(data)

    if not check_no_company_overlap(train_data, test_data):
        msg = "Company overlap detected between train and test sets."
        raise ValueError(msg)

    split_summary = create_split_summary(data, train_data, test_data)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    split_summary_path.parent.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    split_summary.to_csv(split_summary_path, index=False)