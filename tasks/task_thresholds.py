"""pytask tasks for threshold analysis.

This task module studies how bankruptcy classification metrics change when the
probability threshold is varied. The analysis is based on validation predictions
only, so it does not use the final test set for threshold selection.

Run:
    pixi run build

Inputs:
    - outputs/tables/validation_predictions.csv

Outputs:
    - outputs/tables/validation_threshold_analysis.csv
    - outputs/tables/selected_thresholds.csv
    - outputs/figures/validation_threshold_tradeoff.png
"""

from pathlib import Path

from bankruptcy_ml.config import (
    SELECTED_THRESHOLDS_PATH,
    VALIDATION_PREDICTIONS_PATH,
    VALIDATION_THRESHOLD_ANALYSIS_PATH,
    VALIDATION_THRESHOLD_TRADEOFF_FIGURE_PATH,
)
from bankruptcy_ml.thresholds import save_threshold_outputs


def task_create_threshold_outputs(
    depends_on: Path = VALIDATION_PREDICTIONS_PATH,
    produces: tuple[Path, Path, Path] = (
        VALIDATION_THRESHOLD_ANALYSIS_PATH,
        SELECTED_THRESHOLDS_PATH,
        VALIDATION_THRESHOLD_TRADEOFF_FIGURE_PATH,
    ),
) -> None:
    """Create validation threshold-analysis outputs."""
    (
        threshold_analysis_path,
        selected_thresholds_path,
        threshold_tradeoff_figure_path,
    ) = produces

    save_threshold_outputs(
        validation_predictions_path=depends_on,
        threshold_analysis_path=threshold_analysis_path,
        selected_thresholds_path=selected_thresholds_path,
        threshold_tradeoff_figure_path=threshold_tradeoff_figure_path,
    )