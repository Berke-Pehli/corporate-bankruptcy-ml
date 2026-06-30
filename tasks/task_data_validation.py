"""pytask tasks for raw data validation.

This task module creates the first reproducible outputs of the project. It
validates the raw American bankruptcy dataset, writes summary tables, and saves
a class-balance figure.

Run:
    pixi run build

Outputs:
    - outputs/tables/data_summary.csv
    - outputs/tables/target_distribution.csv
    - outputs/report/raw_data_validation.json
    - outputs/figures/class_balance.png
"""

import pandas as pd

from bankruptcy_ml.config import (
    FIGURES_DIR,
    RAW_DATA_PATH,
    REPORT_DIR,
    TABLES_DIR,
)
from bankruptcy_ml.data_validation import run_raw_data_validation
from bankruptcy_ml.visualization import plot_class_balance


def task_validate_raw_data() -> None:
    """Validate the raw dataset and write summary outputs."""
    run_raw_data_validation(
        raw_data_path=RAW_DATA_PATH,
        data_summary_path=TABLES_DIR / "data_summary.csv",
        target_distribution_path=TABLES_DIR / "target_distribution.csv",
        validation_report_path=REPORT_DIR / "raw_data_validation.json",
    )


def task_plot_class_balance() -> None:
    """Create the class-balance figure from the raw dataset."""
    data = pd.read_csv(RAW_DATA_PATH)
    plot_class_balance(data=data, output_path=FIGURES_DIR / "class_balance.png")