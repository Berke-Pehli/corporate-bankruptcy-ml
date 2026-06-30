"""Logistic Regression models for corporate bankruptcy prediction.

This module builds interpretable and regularized Logistic Regression pipelines.
Logistic Regression is the main benchmark because bankruptcy is a binary
classification problem and the model produces predicted probabilities.

Inputs:
    - data/processed/train.csv

Outputs:
    - fitted Logistic Regression models saved under outputs/models/

Models:
    - Interpretable Logistic Regression with L2 penalty and fixed C = 1
    - L1-regularized Logistic Regression selected by validation performance
    - L2-regularized Logistic Regression selected by validation performance

Important interpretation note:
    Logistic Regression coefficients represent changes in log-odds, not direct
    percentage-point changes in bankruptcy probability. Later interpretation
    outputs should therefore describe coefficients as predictive associations,
    not causal effects.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_pipeline(
    penalty: str = "l2",
    c_value: float = 1.0,
    class_weight: str | dict | None = "balanced",
) -> Pipeline:
    """Build a scaled Logistic Regression pipeline.

    Financial variables may have different scales. Logistic Regression and its
    regularized variants are scale-sensitive, so the pipeline standardizes the
    features before fitting the classifier.

    The project labels the models as L1 and L2 Logistic Regression because this
    is the terminology used in the lecture notes. Internally, the newer
    scikit-learn parameterization is used:
        - L1 Logistic Regression: l1_ratio = 1.0
        - L2 Logistic Regression: l1_ratio = 0.0

    Args:
        penalty: Regularization type. Use ``"l1"`` or ``"l2"``.
        c_value: Inverse regularization strength. Smaller values imply stronger
            regularization.
        class_weight: Class-weight strategy. ``"balanced"`` gives more weight
            to the minority bankruptcy class.

    Returns:
        An unfitted scikit-learn Pipeline.

    Raises:
        ValueError: If a penalty other than ``"l1"`` or ``"l2"`` is supplied.
    """
    if penalty not in {"l1", "l2"}:
        msg = "Only 'l1' and 'l2' penalties are supported in this project."
        raise ValueError(msg)

    l1_ratio = 1.0 if penalty == "l1" else 0.0

    model = LogisticRegression(
        C=c_value,
        l1_ratio=l1_ratio,
        solver="saga",
        class_weight=class_weight,
        max_iter=5_000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_interpretable_logit() -> Pipeline:
    """Build the main interpretable Logistic Regression benchmark.

    Returns:
        An unfitted L2 Logistic Regression pipeline with fixed C = 1.
    """
    return build_logistic_pipeline(penalty="l2", c_value=1.0)


def select_regularized_logit(
    x_train,
    y_train,
    x_valid,
    y_valid,
    penalty: str,
    c_grid: list[float],
) -> tuple[Pipeline, float, float]:
    """Select a regularized Logistic Regression model using validation PR-AUC.

    PR-AUC is used for model selection because bankruptcy is the minority class.
    Compared with accuracy, PR-AUC better reflects the model's ability to rank
    failed firms above alive firms.

    Args:
        x_train: Training feature matrix.
        y_train: Training target vector.
        x_valid: Validation feature matrix.
        y_valid: Validation target vector.
        penalty: Regularization penalty, either ``"l1"`` or ``"l2"``.
        c_grid: Candidate inverse regularization strengths.

    Returns:
        A tuple containing:
            - best fitted pipeline
            - best C value
            - best validation PR-AUC

    Raises:
        RuntimeError: If no model can be selected.
    """
    from sklearn.metrics import average_precision_score

    best_model = None
    best_c = None
    best_score = -1.0

    for c_value in c_grid:
        candidate = build_logistic_pipeline(penalty=penalty, c_value=c_value)
        candidate.fit(x_train, y_train)

        probability_failed = candidate.predict_proba(x_valid)[:, 1]
        score = average_precision_score(y_valid, probability_failed)

        if score > best_score:
            best_model = candidate
            best_c = c_value
            best_score = score

    if best_model is None or best_c is None:
        msg = "No Logistic Regression model was selected."
        raise RuntimeError(msg)

    return best_model, best_c, best_score