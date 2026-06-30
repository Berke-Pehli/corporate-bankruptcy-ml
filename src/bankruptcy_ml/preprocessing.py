"""Preprocess the raw American bankruptcy dataset.

This module converts the raw company-year bankruptcy data into a clean modeling
dataset. The main preprocessing decision is to encode the raw status label into
a binary target variable, where failed companies are coded as 1 and alive
companies are coded as 0.

Inputs:
    - data/raw/american_bankruptcy.csv

Outputs:
    - data/processed/model_dataset.csv
    - outputs/tables/feature_dictionary.csv

Main steps:
    - validate required columns
    - identify financial feature columns from the raw dataset
    - encode the bankruptcy target
    - remove constant feature columns if present
    - keep company and year columns for leakage-safe splitting
    - create a readable feature dictionary for interpretation and documentation

Important design choice:
    The original X1-X18 column names are kept in the modeling dataset for
    traceability. Readable names and descriptions are added to the feature
    dictionary and will be used later in plots, tables, README explanations, and
    the final paper.

Leakage prevention:
    The binary target column failed is never allowed to enter the feature matrix.
    Feature columns are identified before the encoded target is added.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bankruptcy_ml.config import (
    ALIVE_LABEL,
    COMPANY_COLUMN,
    FAILED_LABEL,
    RAW_TARGET_COLUMN,
    TARGET_COLUMN,
    YEAR_COLUMN,
)
from bankruptcy_ml.data_validation import (
    identify_feature_columns,
    validate_required_columns,
    validate_target_labels,
)
from bankruptcy_ml.features import FEATURE_DESCRIPTION_MAP, FEATURE_NAME_MAP


def encode_target(data: pd.DataFrame) -> pd.DataFrame:
    """Encode the raw bankruptcy status label as a binary target.

    The raw dataset uses the labels ``alive`` and ``failed``. For the
    classification models, failed company-year observations are coded as 1 and
    alive observations are coded as 0.

    Args:
        data: Raw bankruptcy dataset.

    Returns:
        A copy of the dataset with a binary target column named ``failed``.

    Raises:
        ValueError: If unexpected target labels are present.
    """
    validate_target_labels(data)

    target_mapping = {
        ALIVE_LABEL: 0,
        FAILED_LABEL: 1,
    }

    data = data.copy()
    data[TARGET_COLUMN] = data[RAW_TARGET_COLUMN].map(target_mapping).astype(int)

    return data


def find_constant_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
    """Find columns with one or fewer unique values.

    Constant predictors contain no useful information for classification and can
    create unnecessary noise in model training.

    Args:
        data: Input dataset.
        columns: Candidate columns to check.

    Returns:
        A list of constant column names.
    """
    return [column for column in columns if data[column].nunique(dropna=False) <= 1]


def create_model_dataset(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create the cleaned modeling dataset.

    The returned dataset keeps company and year identifiers because they are
    needed for leakage-safe splitting and later diagnostics. The raw string
    target column is removed after creating the binary target.

    Feature columns are identified from the raw dataset before the binary target
    is added. This prevents the target variable from accidentally becoming a
    predictor.

    Args:
        raw_data: Raw American bankruptcy dataset.

    Returns:
        A tuple containing:
            - the cleaned modeling DataFrame
            - a list of removed constant feature columns
    """
    validate_required_columns(raw_data)
    validate_target_labels(raw_data)

    feature_columns = identify_feature_columns(raw_data)
    data = encode_target(raw_data)

    constant_columns = find_constant_columns(data, feature_columns)
    retained_feature_columns = [
        column for column in feature_columns if column not in constant_columns
    ]

    selected_columns = [
        COMPANY_COLUMN,
        YEAR_COLUMN,
        TARGET_COLUMN,
        *retained_feature_columns,
    ]

    model_dataset = data[selected_columns].copy()

    return model_dataset, constant_columns


def create_feature_dictionary(
    feature_columns: list[str],
    removed_constant_columns: list[str],
) -> pd.DataFrame:
    """Create a documented feature dictionary.

    The feature dictionary links the raw Kaggle feature names, such as ``X1``,
    to readable financial names and descriptions. This allows later model
    interpretation outputs to remain understandable without changing the raw
    modeling column names.

    Args:
        feature_columns: Features retained for modeling.
        removed_constant_columns: Constant columns removed from the model dataset.

    Returns:
        A feature dictionary with readable names, descriptions, roles, and
        inclusion flags.
    """
    included_features = pd.DataFrame(
        {
            "feature": feature_columns,
            "readable_name": [
                FEATURE_NAME_MAP.get(feature, feature) for feature in feature_columns
            ],
            "description": [
                FEATURE_DESCRIPTION_MAP.get(feature, "")
                for feature in feature_columns
            ],
            "role": "financial_predictor",
            "included_in_model": True,
        }
    )

    if not removed_constant_columns:
        return included_features

    removed_features = pd.DataFrame(
        {
            "feature": removed_constant_columns,
            "readable_name": [
                FEATURE_NAME_MAP.get(feature, feature)
                for feature in removed_constant_columns
            ],
            "description": [
                FEATURE_DESCRIPTION_MAP.get(feature, "")
                for feature in removed_constant_columns
            ],
            "role": "removed_constant_column",
            "included_in_model": False,
        }
    )

    return pd.concat([included_features, removed_features], ignore_index=True)


def save_model_dataset(
    raw_data_path: Path,
    model_dataset_path: Path,
    feature_dictionary_path: Path,
) -> None:
    """Create and save the cleaned modeling dataset.

    This function is called from the pytask pipeline. It reads the raw CSV file,
    creates the model-ready dataset, and writes a feature dictionary documenting
    the predictors retained for modeling.

    Args:
        raw_data_path: Path to the raw CSV file.
        model_dataset_path: Output path for the model-ready dataset.
        feature_dictionary_path: Output path for the feature dictionary table.
    """
    raw_data = pd.read_csv(raw_data_path)
    model_dataset, removed_constant_columns = create_model_dataset(raw_data)

    feature_columns = [
        column
        for column in model_dataset.columns
        if column not in {COMPANY_COLUMN, YEAR_COLUMN, TARGET_COLUMN}
    ]

    feature_dictionary = create_feature_dictionary(
        feature_columns=feature_columns,
        removed_constant_columns=removed_constant_columns,
    )

    model_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    feature_dictionary_path.parent.mkdir(parents=True, exist_ok=True)

    model_dataset.to_csv(model_dataset_path, index=False)
    feature_dictionary.to_csv(feature_dictionary_path, index=False)