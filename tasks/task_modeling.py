"""pytask tasks for baseline and Logistic Regression modeling.

This task module trains the first bankruptcy prediction models. It evaluates
them on an internal company-level validation split created from the training
data. The final test set remains untouched for later final evaluation.

Run:
    pixi run build

Inputs:
    - data/processed/train.csv

Outputs:
    - outputs/tables/validation_model_comparison.csv
    - outputs/tables/validation_classification_reports.csv
    - outputs/tables/logistic_model_specification.csv
    - outputs/tables/validation_predictions.csv
    - outputs/models/majority_baseline.joblib
    - outputs/models/interpretable_logit.joblib
    - outputs/models/regularized_logit_l1.joblib
    - outputs/models/regularized_logit_l2.joblib
"""

from pathlib import Path

import joblib
import pandas as pd

from bankruptcy_ml.baselines import build_majority_class_baseline
from bankruptcy_ml.config import (
    INTERPRETABLE_LOGIT_MODEL_PATH,
    LOGISTIC_C_GRID,
    LOGISTIC_MODEL_SPECIFICATION_PATH,
    MAJORITY_BASELINE_MODEL_PATH,
    RANDOM_STATE,
    REGULARIZED_LOGIT_L1_MODEL_PATH,
    REGULARIZED_LOGIT_L2_MODEL_PATH,
    TRAIN_DATA_PATH,
    VALIDATION_CLASSIFICATION_REPORTS_PATH,
    VALIDATION_MODEL_COMPARISON_PATH,
    VALIDATION_PREDICTIONS_PATH,
    VALIDATION_SIZE,
)
from bankruptcy_ml.evaluation import (
    create_classification_report_table,
    create_prediction_table,
    evaluate_binary_classifier,
    get_probability_failed,
)
from bankruptcy_ml.features import split_features_target
from bankruptcy_ml.logistic_models import (
    build_interpretable_logit,
    select_regularized_logit,
)
from bankruptcy_ml.splitting import create_company_level_split


def task_train_baseline_and_logistic_models(
    depends_on: Path = TRAIN_DATA_PATH,
    produces: tuple[Path, Path, Path, Path, Path, Path, Path, Path] = (
        VALIDATION_MODEL_COMPARISON_PATH,
        VALIDATION_CLASSIFICATION_REPORTS_PATH,
        LOGISTIC_MODEL_SPECIFICATION_PATH,
        VALIDATION_PREDICTIONS_PATH,
        MAJORITY_BASELINE_MODEL_PATH,
        INTERPRETABLE_LOGIT_MODEL_PATH,
        REGULARIZED_LOGIT_L1_MODEL_PATH,
        REGULARIZED_LOGIT_L2_MODEL_PATH,
    ),
) -> None:
    """Train baseline and Logistic Regression models."""
    (
        model_comparison_path,
        classification_reports_path,
        model_specification_path,
        predictions_path,
        majority_model_path,
        interpretable_logit_path,
        regularized_l1_path,
        regularized_l2_path,
    ) = produces

    train_full = pd.read_csv(depends_on)

    model_train, validation = create_company_level_split(
        train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    x_train, y_train = split_features_target(model_train)
    x_valid, y_valid = split_features_target(validation)

    models = {}
    model_specs = []

    majority_model = build_majority_class_baseline()
    majority_model.fit(x_train, y_train)
    models["Majority-class baseline"] = majority_model
    model_specs.append(
        {
            "model": "Majority-class baseline",
            "model_type": "DummyClassifier",
            "penalty": "",
            "selected_c": "",
            "selection_metric": "",
            "validation_pr_auc_during_selection": "",
        }
    )

    interpretable_logit = build_interpretable_logit()
    interpretable_logit.fit(x_train, y_train)
    models["Logistic Regression"] = interpretable_logit
    model_specs.append(
        {
            "model": "Logistic Regression",
            "model_type": "LogisticRegression",
            "penalty": "l2",
            "selected_c": 1.0,
            "selection_metric": "fixed benchmark",
            "validation_pr_auc_during_selection": "",
        }
    )

    l1_logit, best_l1_c, best_l1_score = select_regularized_logit(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        penalty="l1",
        c_grid=LOGISTIC_C_GRID,
    )
    models["L1 Regularized Logistic Regression"] = l1_logit
    model_specs.append(
        {
            "model": "L1 Regularized Logistic Regression",
            "model_type": "LogisticRegression",
            "penalty": "l1",
            "selected_c": best_l1_c,
            "selection_metric": "validation PR-AUC",
            "validation_pr_auc_during_selection": best_l1_score,
        }
    )

    l2_logit, best_l2_c, best_l2_score = select_regularized_logit(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        penalty="l2",
        c_grid=LOGISTIC_C_GRID,
    )
    models["L2 Regularized Logistic Regression"] = l2_logit
    model_specs.append(
        {
            "model": "L2 Regularized Logistic Regression",
            "model_type": "LogisticRegression",
            "penalty": "l2",
            "selected_c": best_l2_c,
            "selection_metric": "validation PR-AUC",
            "validation_pr_auc_during_selection": best_l2_score,
        }
    )

    metric_rows = []
    report_tables = []
    prediction_tables = []

    for model_name, model in models.items():
        y_pred = model.predict(x_valid)
        probability_failed = get_probability_failed(model, x_valid)

        metric_rows.append(
            evaluate_binary_classifier(
                model_name=model_name,
                y_true=y_valid,
                y_pred=y_pred,
                probability_failed=probability_failed,
            )
        )

        report_tables.append(
            create_classification_report_table(
                model_name=model_name,
                y_true=y_valid,
                y_pred=y_pred,
            )
        )

        prediction_tables.append(
            create_prediction_table(
                validation_data=validation,
                model_name=model_name,
                y_pred=y_pred,
                probability_failed=probability_failed,
            )
        )

    model_comparison = pd.DataFrame(metric_rows)
    classification_reports = pd.concat(report_tables, ignore_index=True)
    model_specification = pd.DataFrame(model_specs)
    validation_predictions = pd.concat(prediction_tables, ignore_index=True)

    model_comparison_path.parent.mkdir(parents=True, exist_ok=True)
    classification_reports_path.parent.mkdir(parents=True, exist_ok=True)
    model_specification_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    majority_model_path.parent.mkdir(parents=True, exist_ok=True)

    model_comparison.to_csv(model_comparison_path, index=False)
    classification_reports.to_csv(classification_reports_path, index=False)
    model_specification.to_csv(model_specification_path, index=False)
    validation_predictions.to_csv(predictions_path, index=False)

    joblib.dump(majority_model, majority_model_path)
    joblib.dump(interpretable_logit, interpretable_logit_path)
    joblib.dump(l1_logit, regularized_l1_path)
    joblib.dump(l2_logit, regularized_l2_path)