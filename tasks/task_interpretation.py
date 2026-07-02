"""pytask tasks for model interpretation outputs.

This task module creates coefficient and feature-importance outputs for the
bankruptcy prediction models.

Run:
    pixi run build

Inputs:
    - outputs/models/interpretable_logit.joblib
    - outputs/models/decision_tree.joblib
    - outputs/models/random_forest.joblib
    - outputs/models/gradient_boosting.joblib
    - data/processed/train.csv
    - outputs/tables/feature_dictionary.csv

Outputs:
    - outputs/tables/logistic_coefficients.csv
    - outputs/tables/tree_feature_importance.csv
    - outputs/figures/logistic_coefficients.png
    - outputs/figures/tree_feature_importance.png
"""

from pathlib import Path

from bankruptcy_ml.config import (
    DECISION_TREE_MODEL_PATH,
    FEATURE_DICTIONARY_PATH,
    GRADIENT_BOOSTING_MODEL_PATH,
    INTERPRETABLE_LOGIT_MODEL_PATH,
    LOGISTIC_COEFFICIENTS_FIGURE_PATH,
    LOGISTIC_COEFFICIENTS_PATH,
    RANDOM_FOREST_MODEL_PATH,
    TRAIN_DATA_PATH,
    TREE_FEATURE_IMPORTANCE_FIGURE_PATH,
    TREE_FEATURE_IMPORTANCE_PATH,
)
from bankruptcy_ml.interpretation import (
    save_interpretation_figures,
    save_interpretation_tables,
)


def task_create_interpretation_tables(
    depends_on: dict[str, Path] = {
        "logistic_model": INTERPRETABLE_LOGIT_MODEL_PATH,
        "decision_tree": DECISION_TREE_MODEL_PATH,
        "random_forest": RANDOM_FOREST_MODEL_PATH,
        "gradient_boosting": GRADIENT_BOOSTING_MODEL_PATH,
        "train_data": TRAIN_DATA_PATH,
        "feature_dictionary": FEATURE_DICTIONARY_PATH,
    },
    produces: tuple[Path, Path] = (
        LOGISTIC_COEFFICIENTS_PATH,
        TREE_FEATURE_IMPORTANCE_PATH,
    ),
) -> None:
    """Create coefficient and feature-importance tables."""
    logistic_coefficients_path, tree_feature_importance_path = produces

    save_interpretation_tables(
        logistic_model_path=depends_on["logistic_model"],
        tree_model_paths={
            "Decision Tree": depends_on["decision_tree"],
            "Random Forest": depends_on["random_forest"],
            "Gradient Boosting": depends_on["gradient_boosting"],
        },
        train_data_path=depends_on["train_data"],
        feature_dictionary_path=depends_on["feature_dictionary"],
        logistic_coefficients_path=logistic_coefficients_path,
        tree_feature_importance_path=tree_feature_importance_path,
    )


def task_create_interpretation_figures(
    depends_on: tuple[Path, Path] = (
        LOGISTIC_COEFFICIENTS_PATH,
        TREE_FEATURE_IMPORTANCE_PATH,
    ),
    produces: tuple[Path, Path] = (
        LOGISTIC_COEFFICIENTS_FIGURE_PATH,
        TREE_FEATURE_IMPORTANCE_FIGURE_PATH,
    ),
) -> None:
    """Create coefficient and feature-importance figures."""
    logistic_coefficients_path, tree_feature_importance_path = depends_on
    logistic_figure_path, tree_figure_path = produces

    save_interpretation_figures(
        logistic_coefficients_path=logistic_coefficients_path,
        tree_feature_importance_path=tree_feature_importance_path,
        logistic_coefficients_figure_path=logistic_figure_path,
        tree_feature_importance_figure_path=tree_figure_path,
    )