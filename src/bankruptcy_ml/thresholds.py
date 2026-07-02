"""Threshold analysis for bankruptcy prediction models.

This module evaluates how classification metrics change when the probability
threshold for predicting bankruptcy is varied. This is important because
bankruptcy prediction is an imbalanced classification problem and the default
threshold of 0.5 is not necessarily appropriate.

Inputs:
    - outputs/tables/validation_predictions.csv

Outputs:
    - outputs/tables/validation_threshold_analysis.csv
    - outputs/tables/selected_thresholds.csv
    - outputs/figures/validation_threshold_tradeoff.png

Methodological note:
    Threshold selection is performed on validation predictions only. The final
    test set should not be used to choose thresholds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


KEY_MODELS_FOR_THRESHOLD_ANALYSIS = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
]


def create_threshold_analysis(
    predictions: pd.DataFrame,
    thresholds: list[float] | None = None,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate precision, recall, and F1-score across thresholds.

    Args:
        predictions: Validation prediction table with actual labels, model names,
            and predicted failure probabilities.
        thresholds: Probability thresholds to evaluate. If omitted, thresholds
            from 0.01 to 0.99 are used.
        model_names: Optional model names to include. If omitted, all models are
            used.

    Returns:
        Threshold analysis table with one row per model and threshold.
    """
    if thresholds is None:
        thresholds = [value / 100 for value in range(1, 100)]

    if model_names is None:
        model_names = sorted(predictions["model"].unique())

    rows = []

    for model_name in model_names:
        model_predictions = predictions[predictions["model"] == model_name]
        y_true = model_predictions["actual_failed"]
        probability_failed = model_predictions["probability_failed"]

        for threshold in thresholds:
            y_pred = (probability_failed >= threshold).astype(int)

            rows.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    "precision_failed": precision_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    ),
                    "recall_failed": recall_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    ),
                    "f1_failed": f1_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    ),
                    "predicted_failed_share": float(y_pred.mean()),
                    "n_predicted_failed": int(y_pred.sum()),
                    "n_actual_failed": int(y_true.sum()),
                }
            )

    return pd.DataFrame(rows)


def select_thresholds(
    threshold_analysis: pd.DataFrame,
    recall_targets: tuple[float, ...] = (0.6, 0.8),
) -> pd.DataFrame:
    """Select practical thresholds from threshold-analysis results.

    For each model, this function selects:
        - the threshold that maximizes F1-score
        - the highest-precision threshold that reaches each recall target

    Args:
        threshold_analysis: Output from ``create_threshold_analysis``.
        recall_targets: Minimum recall targets to consider.

    Returns:
        Table of selected thresholds and their corresponding metrics.
    """
    selected_rows = []

    for model_name, model_data in threshold_analysis.groupby("model"):
        best_f1_row = model_data.sort_values(
            ["f1_failed", "precision_failed", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]

        selected_rows.append(
            {
                "model": model_name,
                "selection_rule": "maximize_f1",
                "threshold": best_f1_row["threshold"],
                "precision_failed": best_f1_row["precision_failed"],
                "recall_failed": best_f1_row["recall_failed"],
                "f1_failed": best_f1_row["f1_failed"],
                "predicted_failed_share": best_f1_row["predicted_failed_share"],
                "n_predicted_failed": best_f1_row["n_predicted_failed"],
            }
        )

        for recall_target in recall_targets:
            feasible = model_data[model_data["recall_failed"] >= recall_target]

            if feasible.empty:
                continue

            best_recall_target_row = feasible.sort_values(
                ["precision_failed", "f1_failed", "threshold"],
                ascending=[False, False, False],
            ).iloc[0]

            selected_rows.append(
                {
                    "model": model_name,
                    "selection_rule": f"recall_at_least_{recall_target:.1f}",
                    "threshold": best_recall_target_row["threshold"],
                    "precision_failed": best_recall_target_row["precision_failed"],
                    "recall_failed": best_recall_target_row["recall_failed"],
                    "f1_failed": best_recall_target_row["f1_failed"],
                    "predicted_failed_share": best_recall_target_row[
                        "predicted_failed_share"
                    ],
                    "n_predicted_failed": best_recall_target_row[
                        "n_predicted_failed"
                    ],
                }
            )

    return pd.DataFrame(selected_rows)


def plot_threshold_tradeoff(
    threshold_analysis: pd.DataFrame,
    output_path: Path,
    model_names: list[str] | None = None,
) -> None:
    """Plot precision, recall, and F1-score across thresholds.

    Args:
        threshold_analysis: Output from ``create_threshold_analysis``.
        output_path: Path where the figure should be saved.
        model_names: Optional model names to include in the plot.
    """
    if model_names is None:
        model_names = KEY_MODELS_FOR_THRESHOLD_ANALYSIS

    filtered_data = threshold_analysis[
        threshold_analysis["model"].isin(model_names)
    ].copy()

    n_models = len(model_names)
    fig, axes = plt.subplots(n_models, 1, figsize=(9, 4 * n_models), sharex=True)

    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names, strict=False):
        model_data = filtered_data[filtered_data["model"] == model_name]

        ax.plot(
            model_data["threshold"],
            model_data["precision_failed"],
            label="Precision failed",
        )
        ax.plot(
            model_data["threshold"],
            model_data["recall_failed"],
            label="Recall failed",
        )
        ax.plot(
            model_data["threshold"],
            model_data["f1_failed"],
            label="F1 failed",
        )

        ax.set_title(f"Threshold Trade-off: {model_name}")
        ax.set_ylabel("Metric value")
        ax.set_ylim(0, 1)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Probability threshold for predicting failure")

    fig.suptitle("Validation Threshold Trade-off", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_threshold_outputs(
    validation_predictions_path: Path,
    threshold_analysis_path: Path,
    selected_thresholds_path: Path,
    threshold_tradeoff_figure_path: Path,
) -> None:
    """Create and save threshold-analysis tables and figure.

    Args:
        validation_predictions_path: Path to validation prediction table.
        threshold_analysis_path: Output path for threshold analysis table.
        selected_thresholds_path: Output path for selected thresholds table.
        threshold_tradeoff_figure_path: Output path for threshold trade-off figure.
    """
    predictions = pd.read_csv(validation_predictions_path)

    threshold_analysis = create_threshold_analysis(
        predictions=predictions,
        model_names=KEY_MODELS_FOR_THRESHOLD_ANALYSIS,
    )
    selected_thresholds = select_thresholds(threshold_analysis)

    threshold_analysis_path.parent.mkdir(parents=True, exist_ok=True)
    selected_thresholds_path.parent.mkdir(parents=True, exist_ok=True)

    threshold_analysis.to_csv(threshold_analysis_path, index=False)
    selected_thresholds.to_csv(selected_thresholds_path, index=False)

    plot_threshold_tradeoff(
        threshold_analysis=threshold_analysis,
        output_path=threshold_tradeoff_figure_path,
        model_names=KEY_MODELS_FOR_THRESHOLD_ANALYSIS,
    )