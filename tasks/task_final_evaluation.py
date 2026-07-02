"""pytask tasks for final test evaluation.

This task module evaluates the already selected bankruptcy prediction models on
the untouched final test set and creates the final evaluation figures.

Run:
    pixi run build

Inputs:
    - data/processed/test.csv
    - outputs/models/*.joblib

Outputs:
    - outputs/tables/final_test_metrics.csv
    - outputs/tables/final_test_classification_reports.csv
    - outputs/tables/final_test_predictions.csv
    - outputs/figures/final_test_roc_curves.png
    - outputs/figures/final_test_precision_recall_curves.png
    - outputs/figures/final_test_confusion_matrices.png
    - outputs/figures/final_test_metric_comparison.png
    - outputs/figures/final_test_core_metric_summary.png
    - outputs/figures/final_test_confusion_matrices_key_models.png
    - outputs/figures/final_test_roc_curves_key_models.png
    - outputs/figures/final_test_precision_recall_curves_key_models.png
"""

from pathlib import Path

import pandas as pd

from bankruptcy_ml.config import (
    DECISION_TREE_MODEL_PATH,
    FINAL_TEST_CLASSIFICATION_REPORTS_PATH,
    FINAL_TEST_CONFUSION_MATRICES_KEY_MODELS_PATH,
    FINAL_TEST_CONFUSION_MATRICES_PATH,
    FINAL_TEST_CORE_METRIC_SUMMARY_PATH,
    FINAL_TEST_METRIC_COMPARISON_PATH,
    FINAL_TEST_METRICS_PATH,
    FINAL_TEST_PRECISION_RECALL_CURVES_KEY_MODELS_PATH,
    FINAL_TEST_PRECISION_RECALL_CURVES_PATH,
    FINAL_TEST_PREDICTIONS_PATH,
    FINAL_TEST_ROC_CURVES_KEY_MODELS_PATH,
    FINAL_TEST_ROC_CURVES_PATH,
    GRADIENT_BOOSTING_MODEL_PATH,
    INTERPRETABLE_LOGIT_MODEL_PATH,
    MAJORITY_BASELINE_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    REGULARIZED_LOGIT_L1_MODEL_PATH,
    REGULARIZED_LOGIT_L2_MODEL_PATH,
    TEST_DATA_PATH,
)
from bankruptcy_ml.final_evaluation import save_final_test_evaluation
from bankruptcy_ml.visualization import (
    plot_final_test_confusion_matrices,
    plot_final_test_confusion_matrices_key_models,
    plot_final_test_core_metric_summary,
    plot_final_test_metric_comparison,
    plot_final_test_precision_recall_curves,
    plot_final_test_precision_recall_curves_key_models,
    plot_final_test_roc_curves,
    plot_final_test_roc_curves_key_models,
)


FINAL_MODEL_PATHS = {
    "Majority-class baseline": MAJORITY_BASELINE_MODEL_PATH,
    "Logistic Regression": INTERPRETABLE_LOGIT_MODEL_PATH,
    "L1 Regularized Logistic Regression": REGULARIZED_LOGIT_L1_MODEL_PATH,
    "L2 Regularized Logistic Regression": REGULARIZED_LOGIT_L2_MODEL_PATH,
    "Decision Tree": DECISION_TREE_MODEL_PATH,
    "Random Forest": RANDOM_FOREST_MODEL_PATH,
    "Gradient Boosting": GRADIENT_BOOSTING_MODEL_PATH,
}


def task_evaluate_models_on_final_test_set(
    depends_on: dict[str, Path] = {
        "test_data": TEST_DATA_PATH,
        "majority_baseline": MAJORITY_BASELINE_MODEL_PATH,
        "logistic_regression": INTERPRETABLE_LOGIT_MODEL_PATH,
        "l1_logit": REGULARIZED_LOGIT_L1_MODEL_PATH,
        "l2_logit": REGULARIZED_LOGIT_L2_MODEL_PATH,
        "decision_tree": DECISION_TREE_MODEL_PATH,
        "random_forest": RANDOM_FOREST_MODEL_PATH,
        "gradient_boosting": GRADIENT_BOOSTING_MODEL_PATH,
    },
    produces: tuple[Path, Path, Path] = (
        FINAL_TEST_METRICS_PATH,
        FINAL_TEST_CLASSIFICATION_REPORTS_PATH,
        FINAL_TEST_PREDICTIONS_PATH,
    ),
) -> None:
    """Evaluate all fitted models on the untouched final test set."""
    metrics_path, classification_reports_path, predictions_path = produces

    model_paths = {
        "Majority-class baseline": depends_on["majority_baseline"],
        "Logistic Regression": depends_on["logistic_regression"],
        "L1 Regularized Logistic Regression": depends_on["l1_logit"],
        "L2 Regularized Logistic Regression": depends_on["l2_logit"],
        "Decision Tree": depends_on["decision_tree"],
        "Random Forest": depends_on["random_forest"],
        "Gradient Boosting": depends_on["gradient_boosting"],
    }

    save_final_test_evaluation(
        model_paths=model_paths,
        test_data_path=depends_on["test_data"],
        final_test_metrics_path=metrics_path,
        final_test_classification_reports_path=classification_reports_path,
        final_test_predictions_path=predictions_path,
    )


def task_plot_final_test_roc_curves(
    depends_on: Path = FINAL_TEST_PREDICTIONS_PATH,
    produces: Path = FINAL_TEST_ROC_CURVES_PATH,
) -> None:
    """Create final test ROC curves for all models."""
    predictions = pd.read_csv(depends_on)
    plot_final_test_roc_curves(predictions=predictions, output_path=produces)


def task_plot_final_test_precision_recall_curves(
    depends_on: Path = FINAL_TEST_PREDICTIONS_PATH,
    produces: Path = FINAL_TEST_PRECISION_RECALL_CURVES_PATH,
) -> None:
    """Create final test precision-recall curves for all models."""
    predictions = pd.read_csv(depends_on)
    plot_final_test_precision_recall_curves(
        predictions=predictions,
        output_path=produces,
    )


def task_plot_final_test_confusion_matrices(
    depends_on: Path = FINAL_TEST_PREDICTIONS_PATH,
    produces: Path = FINAL_TEST_CONFUSION_MATRICES_PATH,
) -> None:
    """Create final test confusion matrices for all models."""
    predictions = pd.read_csv(depends_on)
    plot_final_test_confusion_matrices(predictions=predictions, output_path=produces)


def task_plot_final_test_metric_comparison(
    depends_on: Path = FINAL_TEST_METRICS_PATH,
    produces: Path = FINAL_TEST_METRIC_COMPARISON_PATH,
) -> None:
    """Create final test metric comparison figure for all models."""
    metrics = pd.read_csv(depends_on)
    plot_final_test_metric_comparison(
        model_comparison=metrics,
        output_path=produces,
    )


def task_plot_final_test_core_metric_summary(
    depends_on: Path = FINAL_TEST_METRICS_PATH,
    produces: Path = FINAL_TEST_CORE_METRIC_SUMMARY_PATH,
) -> None:
    """Create a cleaner final test metric summary for key models.

    This figure is intended for the README, final report, and oral exam. It keeps
    only the most important bankruptcy-class metrics and only the key models used
    in the main project story.
    """
    metrics = pd.read_csv(depends_on)
    plot_final_test_core_metric_summary(
        model_comparison=metrics,
        output_path=produces,
    )


def task_plot_final_test_confusion_matrices_key_models(
    depends_on: Path = FINAL_TEST_PREDICTIONS_PATH,
    produces: Path = FINAL_TEST_CONFUSION_MATRICES_KEY_MODELS_PATH,
) -> None:
    """Create final test confusion matrices for key models only.

    The simplified figure is easier to explain than the full model comparison
    because it focuses on the baseline, interpretable Logistic Regression, Random
    Forest, and Gradient Boosting.
    """
    predictions = pd.read_csv(depends_on)
    plot_final_test_confusion_matrices_key_models(
        predictions=predictions,
        output_path=produces,
    )


def task_plot_final_test_roc_curves_key_models(
    depends_on: Path = FINAL_TEST_PREDICTIONS_PATH,
    produces: Path = FINAL_TEST_ROC_CURVES_KEY_MODELS_PATH,
) -> None:
    """Create final test ROC curves for key models only."""
    predictions = pd.read_csv(depends_on)
    plot_final_test_roc_curves_key_models(
        predictions=predictions,
        output_path=produces,
    )


def task_plot_final_test_precision_recall_curves_key_models(
    depends_on: Path = FINAL_TEST_PREDICTIONS_PATH,
    produces: Path = FINAL_TEST_PRECISION_RECALL_CURVES_KEY_MODELS_PATH,
) -> None:
    """Create final test precision-recall curves for key models only.

    This figure is especially useful for the oral exam because precision-recall
    curves focus directly on the minority failed-firm class.
    """
    predictions = pd.read_csv(depends_on)
    plot_final_test_precision_recall_curves_key_models(
        predictions=predictions,
        output_path=produces,
    )