"""pytask tasks for exploratory data analysis outputs.

This task module creates descriptive finance diagnostics for the corporate
bankruptcy dataset. These outputs are used in the README, final report, and oral
exam preparation.

Run:
    pixi run build

Inputs:
    - data/processed/model_dataset.csv
    - data/processed/train.csv
    - data/processed/test.csv

Outputs:
    - outputs/tables/annual_failure_rate.csv
    - outputs/tables/train_test_year_distribution.csv
    - outputs/tables/class_feature_summary.csv
    - outputs/figures/annual_failure_rate.png
    - outputs/figures/train_test_year_distribution.png
    - outputs/figures/feature_correlation_heatmap.png
    - outputs/figures/key_feature_distributions_by_status.png
    - outputs/figures/key_feature_median_by_status.png
"""

from pathlib import Path

from bankruptcy_ml.config import (
    ANNUAL_FAILURE_RATE_FIGURE_PATH,
    ANNUAL_FAILURE_RATE_PATH,
    CLASS_FEATURE_SUMMARY_PATH,
    FEATURE_CORRELATION_HEATMAP_PATH,
    KEY_FEATURE_DISTRIBUTIONS_FIGURE_PATH,
    KEY_FEATURE_MEDIAN_BY_STATUS_FIGURE_PATH,
    MODEL_DATASET_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    TRAIN_TEST_YEAR_DISTRIBUTION_FIGURE_PATH,
    TRAIN_TEST_YEAR_DISTRIBUTION_PATH,
)
from bankruptcy_ml.eda import save_eda_outputs


def task_create_eda_outputs(
    depends_on: tuple[Path, Path, Path] = (
        MODEL_DATASET_PATH,
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
    ),
    produces: tuple[Path, Path, Path, Path, Path, Path, Path, Path] = (
        ANNUAL_FAILURE_RATE_PATH,
        TRAIN_TEST_YEAR_DISTRIBUTION_PATH,
        CLASS_FEATURE_SUMMARY_PATH,
        ANNUAL_FAILURE_RATE_FIGURE_PATH,
        TRAIN_TEST_YEAR_DISTRIBUTION_FIGURE_PATH,
        FEATURE_CORRELATION_HEATMAP_PATH,
        KEY_FEATURE_DISTRIBUTIONS_FIGURE_PATH,
        KEY_FEATURE_MEDIAN_BY_STATUS_FIGURE_PATH,
    ),
) -> None:
    """Create EDA tables and figures for the bankruptcy dataset."""
    model_dataset_path, train_data_path, test_data_path = depends_on
    (
        annual_failure_rate_path,
        train_test_year_distribution_path,
        class_feature_summary_path,
        annual_failure_rate_figure_path,
        train_test_year_distribution_figure_path,
        feature_correlation_heatmap_path,
        key_feature_distributions_figure_path,
        key_feature_median_by_status_figure_path,
    ) = produces

    save_eda_outputs(
        model_dataset_path=model_dataset_path,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        annual_failure_rate_path=annual_failure_rate_path,
        train_test_year_distribution_path=train_test_year_distribution_path,
        class_feature_summary_path=class_feature_summary_path,
        annual_failure_rate_figure_path=annual_failure_rate_figure_path,
        train_test_year_distribution_figure_path=train_test_year_distribution_figure_path,
        feature_correlation_heatmap_path=feature_correlation_heatmap_path,
        key_feature_distributions_figure_path=key_feature_distributions_figure_path,
        key_feature_median_by_status_figure_path=key_feature_median_by_status_figure_path,
    )