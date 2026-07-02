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

VALIDATION_MODEL_COMPARISON_PATH = TABLES_DIR / "validation_model_comparison.csv"
VALIDATION_CLASSIFICATION_REPORTS_PATH = (
    TABLES_DIR / "validation_classification_reports.csv"
)
MODEL_SPECIFICATION_PATH = TABLES_DIR / "model_specification.csv"
VALIDATION_PREDICTIONS_PATH = TABLES_DIR / "validation_predictions.csv"

MAJORITY_BASELINE_MODEL_PATH = MODELS_DIR / "majority_baseline.joblib"
INTERPRETABLE_LOGIT_MODEL_PATH = MODELS_DIR / "interpretable_logit.joblib"
REGULARIZED_LOGIT_L1_MODEL_PATH = MODELS_DIR / "regularized_logit_l1.joblib"
REGULARIZED_LOGIT_L2_MODEL_PATH = MODELS_DIR / "regularized_logit_l2.joblib"
DECISION_TREE_MODEL_PATH = MODELS_DIR / "decision_tree.joblib"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
GRADIENT_BOOSTING_MODEL_PATH = MODELS_DIR / "gradient_boosting.joblib"

VALIDATION_SIZE = 0.2
LOGISTIC_C_GRID = [0.01, 0.1, 1.0, 10.0]

VALIDATION_ROC_CURVES_PATH = FIGURES_DIR / "validation_roc_curves.png"
VALIDATION_PRECISION_RECALL_CURVES_PATH = (
    FIGURES_DIR / "validation_precision_recall_curves.png"
)
VALIDATION_CONFUSION_MATRICES_PATH = FIGURES_DIR / "validation_confusion_matrices.png"
VALIDATION_METRIC_COMPARISON_PATH = FIGURES_DIR / "validation_metric_comparison.png"