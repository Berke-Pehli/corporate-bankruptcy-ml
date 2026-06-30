"""pytask tasks for preprocessing and leakage-safe data splitting.

This task module transforms the raw American bankruptcy dataset into a
model-ready dataset and creates a company-level train-test split.

Run:
    pixi run build

Outputs:
    - data/processed/model_dataset.csv
    - data/processed/train.csv
    - data/processed/test.csv
    - outputs/tables/feature_dictionary.csv
    - outputs/tables/split_summary.csv
"""

from pathlib import Path

from bankruptcy_ml.config import (
    FEATURE_DICTIONARY_PATH,
    MODEL_DATASET_PATH,
    RAW_DATA_PATH,
    SPLIT_SUMMARY_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from bankruptcy_ml.preprocessing import save_model_dataset
from bankruptcy_ml.splitting import save_company_level_split


def task_create_model_dataset(
    depends_on: Path = RAW_DATA_PATH,
    produces: tuple[Path, Path] = (MODEL_DATASET_PATH, FEATURE_DICTIONARY_PATH),
) -> None:
    """Create the cleaned model-ready dataset."""
    model_dataset_path, feature_dictionary_path = produces

    save_model_dataset(
        raw_data_path=depends_on,
        model_dataset_path=model_dataset_path,
        feature_dictionary_path=feature_dictionary_path,
    )


def task_create_company_level_split(
    depends_on: Path = MODEL_DATASET_PATH,
    produces: tuple[Path, Path, Path] = (
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
        SPLIT_SUMMARY_PATH,
    ),
) -> None:
    """Create the leakage-safe company-level train-test split."""
    train_path, test_path, split_summary_path = produces

    save_company_level_split(
        model_dataset_path=depends_on,
        train_path=train_path,
        test_path=test_path,
        split_summary_path=split_summary_path,
    )