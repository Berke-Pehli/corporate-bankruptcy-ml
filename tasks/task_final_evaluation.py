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
"""

from pathlib import Path

from bankruptcy_ml.config import (
    DECISION_TREE_MODEL_PATH,
    FINAL_TEST_CLASSIFICATION_REPORTS_PATH,
    FINAL_TEST_METRICS_PATH,
    FINAL_TEST_PREDICTIONS_PATH,
    GRADIENT_BOOSTING_MODEL_PATH,
    INTERPRETABLE_LOGIT_MODEL_PATH,
    MAJORITY_BASELINE_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    REGULARIZED_LOGIT_L1_MODEL_PATH,
    REGULARIZED_LOGIT_L2_MODEL_PATH,
    TEST_DATA_PATH,
)
from bankruptcy_ml.final_evaluation import save_final_test_evaluation

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
    depends_on: dict[str, Path] = {  # noqa: B006
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
