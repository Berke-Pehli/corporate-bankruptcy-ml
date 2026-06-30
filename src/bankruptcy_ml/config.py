"""Project configuration for the corporate bankruptcy ML project.

This module centralizes file paths, column names, and project-wide constants.
Keeping these values in one place makes the pipeline easier to maintain and
reduces the risk of hard-coded paths across different scripts.

Inputs:
    - data/raw/american_bankruptcy.csv

Main outputs:
    - data/processed/
    - outputs/tables/
    - outputs/figures/
    - outputs/models/
    - outputs/report/

Core columns:
    - company_name: company identifier used to avoid train-test leakage
    - status_label: raw bankruptcy status label
    - year: company-year observation year
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORT_DIR = OUTPUTS_DIR / "report"

RAW_DATA_PATH = RAW_DATA_DIR / "american_bankruptcy.csv"

COMPANY_COLUMN = "company_name"
RAW_TARGET_COLUMN = "status_label"
TARGET_COLUMN = "failed"
YEAR_COLUMN = "year"

ALIVE_LABEL = "alive"
FAILED_LABEL = "failed"

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_DATASET_PATH = PROCESSED_DATA_DIR / "model_dataset.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

FEATURE_DICTIONARY_PATH = TABLES_DIR / "feature_dictionary.csv"
SPLIT_SUMMARY_PATH = TABLES_DIR / "split_summary.csv"