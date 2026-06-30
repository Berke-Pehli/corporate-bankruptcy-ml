"""Validate the raw American bankruptcy dataset.

This module checks the structure and quality of the raw company-year dataset
before any modeling decisions are made. The goal is to document the raw data
clearly and to detect issues that would affect the classification pipeline.

Inputs:
    - data/raw/american_bankruptcy.csv

Outputs:
    - outputs/tables/data_summary.csv
    - outputs/tables/target_distribution.csv
    - outputs/report/raw_data_validation.json

Validation checks:
    - expected columns exist
    - target labels are valid
    - missing values are counted
    - duplicated rows are counted
    - company and year coverage are summarized
    - numeric feature columns are identified
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bankruptcy_ml.config import (
    ALIVE_LABEL,
    COMPANY_COLUMN,
    FAILED_LABEL,
    RAW_TARGET_COLUMN,
    YEAR_COLUMN,
)


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw bankruptcy dataset.

    Args:
        path: Path to the raw CSV file.

    Returns:
        The raw dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the raw dataset does not exist.
    """
    if not path.exists():
        msg = f"Raw dataset not found at: {path}"
        raise FileNotFoundError(msg)

    return pd.read_csv(path)


def validate_required_columns(data: pd.DataFrame) -> None:
    """Check whether the raw dataset contains the required columns.

    The project depends on a company identifier, a year column, and a raw target
    label. If any of these columns are missing, the later preprocessing and
    leakage-safe splitting steps cannot be performed.

    Args:
        data: Raw bankruptcy dataset.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    required_columns = {COMPANY_COLUMN, RAW_TARGET_COLUMN, YEAR_COLUMN}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        msg = f"Missing required columns: {sorted(missing_columns)}"
        raise ValueError(msg)


def validate_target_labels(data: pd.DataFrame) -> None:
    """Check whether the target column contains only expected status labels.

    The expected labels are ``alive`` and ``failed``. These labels will later be
    converted into a binary target where failed firms are coded as 1 and alive
    firms are coded as 0.

    Args:
        data: Raw bankruptcy dataset.

    Raises:
        ValueError: If unexpected target labels are found.
    """
    observed_labels = set(data[RAW_TARGET_COLUMN].dropna().unique())
    expected_labels = {ALIVE_LABEL, FAILED_LABEL}
    unexpected_labels = observed_labels.difference(expected_labels)

    if unexpected_labels:
        msg = f"Unexpected target labels found: {sorted(unexpected_labels)}"
        raise ValueError(msg)


def identify_feature_columns(data: pd.DataFrame) -> list[str]:
    """Identify numeric financial feature columns.

    The raw dataset contains identifier columns, a year column, a target label,
    and financial indicators. For the initial data validation step, feature
    columns are defined as numeric columns excluding the year.

    Args:
        data: Raw bankruptcy dataset.

    Returns:
        A list of numeric financial feature columns.
    """
    excluded_columns = {COMPANY_COLUMN, RAW_TARGET_COLUMN, YEAR_COLUMN}

    return [
        column
        for column in data.columns
        if column not in excluded_columns and pd.api.types.is_numeric_dtype(data[column])
    ]


def create_data_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Create a one-row summary table for the raw dataset.

    The summary table documents the most important raw-data properties. It is
    designed for both the project README and the final university paper.

    Args:
        data: Raw bankruptcy dataset.

    Returns:
        A one-row DataFrame with dataset-level summary statistics.
    """
    feature_columns = identify_feature_columns(data)
    target_counts = data[RAW_TARGET_COLUMN].value_counts()

    failed_count = int(target_counts.get(FAILED_LABEL, 0))
    alive_count = int(target_counts.get(ALIVE_LABEL, 0))
    total_rows = int(len(data))

    summary = {
        "n_rows": total_rows,
        "n_columns": int(data.shape[1]),
        "n_companies": int(data[COMPANY_COLUMN].nunique()),
        "min_year": int(data[YEAR_COLUMN].min()),
        "max_year": int(data[YEAR_COLUMN].max()),
        "n_feature_columns": int(len(feature_columns)),
        "n_failed_rows": failed_count,
        "n_alive_rows": alive_count,
        "failure_rate": failed_count / total_rows,
        "n_missing_values": int(data.isna().sum().sum()),
        "n_duplicate_rows": int(data.duplicated().sum()),
    }

    return pd.DataFrame([summary])


def create_target_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """Create a target distribution table.

    This table is important because bankruptcy prediction is an imbalanced
    classification problem. The table records both counts and proportions for
    the alive and failed classes.

    Args:
        data: Raw bankruptcy dataset.

    Returns:
        A DataFrame with one row per raw target label.
    """
    counts = data[RAW_TARGET_COLUMN].value_counts().rename_axis("status_label")
    distribution = counts.reset_index(name="count")
    distribution["share"] = distribution["count"] / distribution["count"].sum()

    return distribution


def create_validation_report(data: pd.DataFrame) -> dict[str, Any]:
    """Create a JSON-serializable raw-data validation report.

    The report captures detailed metadata that is useful for reproducibility and
    later documentation.

    Args:
        data: Raw bankruptcy dataset.

    Returns:
        A dictionary containing raw-data validation metadata.
    """
    feature_columns = identify_feature_columns(data)

    return {
        "n_rows": int(len(data)),
        "n_columns": int(data.shape[1]),
        "columns": list(data.columns),
        "feature_columns": feature_columns,
        "n_feature_columns": int(len(feature_columns)),
        "target_counts": {
            str(key): int(value)
            for key, value in data[RAW_TARGET_COLUMN].value_counts().items()
        },
        "company_count": int(data[COMPANY_COLUMN].nunique()),
        "year_min": int(data[YEAR_COLUMN].min()),
        "year_max": int(data[YEAR_COLUMN].max()),
        "missing_values_by_column": {
            str(key): int(value) for key, value in data.isna().sum().items()
        },
        "n_duplicate_rows": int(data.duplicated().sum()),
    }


def run_raw_data_validation(
    raw_data_path: Path,
    data_summary_path: Path,
    target_distribution_path: Path,
    validation_report_path: Path,
) -> None:
    """Run the complete raw-data validation step.

    This function is called by the pytask pipeline. It loads the raw data,
    performs basic validation checks, and writes the first project outputs.

    Args:
        raw_data_path: Path to the raw CSV file.
        data_summary_path: Output path for the dataset summary table.
        target_distribution_path: Output path for the target distribution table.
        validation_report_path: Output path for the JSON validation report.
    """
    data = load_raw_data(raw_data_path)

    validate_required_columns(data)
    validate_target_labels(data)

    data_summary = create_data_summary(data)
    target_distribution = create_target_distribution(data)
    validation_report = create_validation_report(data)

    data_summary_path.parent.mkdir(parents=True, exist_ok=True)
    target_distribution_path.parent.mkdir(parents=True, exist_ok=True)
    validation_report_path.parent.mkdir(parents=True, exist_ok=True)

    data_summary.to_csv(data_summary_path, index=False)
    target_distribution.to_csv(target_distribution_path, index=False)

    with validation_report_path.open("w", encoding="utf-8") as file:
        json.dump(validation_report, file, indent=2)