"""pytask tasks for raw data validation.

This task module creates the first reproducible outputs of the project. It
validates the raw American bankruptcy dataset and writes summary tables.

Run:
    pixi run build

Outputs:
    - outputs/tables/data_summary.csv
    - outputs/tables/target_distribution.csv
    - outputs/report/raw_data_validation.json
"""

from bankruptcy_ml.config import (
    RAW_DATA_PATH,
    REPORT_DIR,
    TABLES_DIR,
)
from bankruptcy_ml.data_validation import run_raw_data_validation


def task_validate_raw_data() -> None:
    """Validate the raw dataset and write summary outputs."""
    run_raw_data_validation(
        raw_data_path=RAW_DATA_PATH,
        data_summary_path=TABLES_DIR / "data_summary.csv",
        target_distribution_path=TABLES_DIR / "target_distribution.csv",
        validation_report_path=REPORT_DIR / "raw_data_validation.json",
    )
