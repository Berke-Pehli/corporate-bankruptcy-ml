"""pytask tasks for reproducible paper-ready outputs.

This task creates the curated figures and tables under ``outputs/paper/`` used
by ``reports/paper/main.tex``. It derives them from canonical generated
analysis outputs and does not fit or select models.

Run:
    pixi run build

Inputs:
    - outputs/tables/target_distribution.csv
    - outputs/tables/annual_failure_rate.csv
    - outputs/tables/class_feature_summary.csv
    - outputs/tables/final_test_metrics.csv
    - outputs/tables/final_test_predictions.csv
    - outputs/tables/logistic_coefficients.csv
    - outputs/tables/tree_feature_importance.csv
    - outputs/tables/selected_thresholds.csv
    - outputs/tables/pca_logistic_results.csv

Outputs:
    - outputs/paper/tables/*.csv
    - outputs/paper/figures/*.png
"""

from pathlib import Path

from bankruptcy_ml.config import (
    ANNUAL_FAILURE_RATE_PATH,
    CLASS_FEATURE_SUMMARY_PATH,
    FINAL_TEST_METRICS_PATH,
    FINAL_TEST_PREDICTIONS_PATH,
    LOGISTIC_COEFFICIENTS_PATH,
    OUTPUTS_DIR,
    PCA_LOGISTIC_RESULTS_PATH,
    SELECTED_THRESHOLDS_PATH,
    TABLES_DIR,
    TREE_FEATURE_IMPORTANCE_PATH,
)
from bankruptcy_ml.paper_outputs import save_paper_outputs

PAPER_OUTPUTS_DIR = OUTPUTS_DIR / "paper"
PAPER_TABLES_DIR = PAPER_OUTPUTS_DIR / "tables"
PAPER_FIGURES_DIR = PAPER_OUTPUTS_DIR / "figures"


def task_create_paper_outputs(
    depends_on: dict[str, Path] = {  # noqa: B006
        "target_distribution": TABLES_DIR / "target_distribution.csv",
        "annual_failure_rate": ANNUAL_FAILURE_RATE_PATH,
        "class_feature_summary": CLASS_FEATURE_SUMMARY_PATH,
        "final_test_metrics": FINAL_TEST_METRICS_PATH,
        "final_test_predictions": FINAL_TEST_PREDICTIONS_PATH,
        "logistic_coefficients": LOGISTIC_COEFFICIENTS_PATH,
        "tree_feature_importance": TREE_FEATURE_IMPORTANCE_PATH,
        "selected_thresholds": SELECTED_THRESHOLDS_PATH,
        "pca_logistic_results": PCA_LOGISTIC_RESULTS_PATH,
    },
    produces: tuple[Path, ...] = (
        PAPER_TABLES_DIR / "model_performance_summary.csv",
        PAPER_TABLES_DIR / "confusion_matrix_summary.csv",
        PAPER_TABLES_DIR / "financial_median_summary.csv",
        PAPER_TABLES_DIR / "pca_summary.csv",
        PAPER_TABLES_DIR / "selected_threshold_summary.csv",
        PAPER_FIGURES_DIR / "class_balance.png",
        PAPER_FIGURES_DIR / "annual_failure_rate.png",
        PAPER_FIGURES_DIR / "model_performance_summary.png",
        PAPER_FIGURES_DIR / "confusion_matrix_summary.png",
        PAPER_FIGURES_DIR / "precision_recall_key_models.png",
        PAPER_FIGURES_DIR / "top_logistic_coefficients.png",
        PAPER_FIGURES_DIR / "top_tree_feature_importance.png",
        PAPER_FIGURES_DIR / "pca_explained_variance.png",
    ),
) -> None:
    """Create paper-ready summary tables and figures."""
    save_paper_outputs(
        target_distribution_path=depends_on["target_distribution"],
        annual_failure_rate_path=depends_on["annual_failure_rate"],
        class_feature_summary_path=depends_on["class_feature_summary"],
        final_test_metrics_path=depends_on["final_test_metrics"],
        final_test_predictions_path=depends_on["final_test_predictions"],
        logistic_coefficients_path=depends_on["logistic_coefficients"],
        tree_feature_importance_path=depends_on["tree_feature_importance"],
        selected_thresholds_path=depends_on["selected_thresholds"],
        pca_logistic_results_path=depends_on["pca_logistic_results"],
        paper_tables_dir=produces[0].parent,
        paper_figures_dir=produces[5].parent,
    )
