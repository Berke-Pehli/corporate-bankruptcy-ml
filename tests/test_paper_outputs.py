"""Tests for paper-ready output helpers."""

import pandas as pd

from bankruptcy_ml.paper_outputs import (
    PAPER_MODELS,
    create_confusion_matrix_summary,
    create_model_performance_summary,
    save_paper_outputs,
)


def _minimal_final_test_metrics() -> pd.DataFrame:
    """Return a compact final-test metric table for paper-output tests."""
    rows = []

    for index, model in enumerate(PAPER_MODELS):
        rows.append(
            {
                "model": model,
                "accuracy": 0.8 + index * 0.01,
                "balanced_accuracy": 0.6,
                "roc_auc": 0.7,
                "pr_auc": 0.15,
                "precision_failed": 0.1,
                "recall_failed": 0.5,
                "f1_failed": 0.17,
                "true_negative": 10,
                "false_positive": 2,
                "false_negative": 3,
                "true_positive": 4,
            }
        )

    return pd.DataFrame(rows)


def test_create_model_performance_summary_keeps_paper_models() -> None:
    """Check that the paper performance table keeps the expected model order."""
    result = create_model_performance_summary(_minimal_final_test_metrics())

    assert result["model"].tolist() == PAPER_MODELS
    assert result["evaluation_sample"].unique().tolist() == ["Final test"]
    assert "pr_auc" in result.columns


def test_create_confusion_matrix_summary_renames_counts() -> None:
    """Check that the paper confusion table uses reader-friendly column names."""
    result = create_confusion_matrix_summary(_minimal_final_test_metrics())

    assert "detected_failures" in result.columns
    assert "missed_failures" in result.columns
    assert "false_alarms" in result.columns
    assert result.loc[0, "detected_failures"] == 4


def test_save_paper_outputs_creates_expected_files(tmp_path) -> None:
    """Check that the complete paper-output writer creates tables and figures."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    target_distribution_path = source_dir / "target_distribution.csv"
    annual_failure_rate_path = source_dir / "annual_failure_rate.csv"
    class_feature_summary_path = source_dir / "class_feature_summary.csv"
    final_test_metrics_path = source_dir / "final_test_metrics.csv"
    final_test_predictions_path = source_dir / "final_test_predictions.csv"
    logistic_coefficients_path = source_dir / "logistic_coefficients.csv"
    tree_feature_importance_path = source_dir / "tree_feature_importance.csv"
    selected_thresholds_path = source_dir / "selected_thresholds.csv"
    pca_logistic_results_path = source_dir / "pca_logistic_results.csv"

    pd.DataFrame(
        {
            "status_label": ["alive", "failed"],
            "count": [90, 10],
            "share": [0.9, 0.1],
        }
    ).to_csv(target_distribution_path, index=False)
    pd.DataFrame(
        {
            "year": [2000, 2001],
            "n_observations": [50, 50],
            "n_failed": [4, 6],
            "n_alive": [46, 44],
            "failure_rate": [0.08, 0.12],
        }
    ).to_csv(annual_failure_rate_path, index=False)
    pd.DataFrame(
        {
            "feature": ["X1", "X1"],
            "readable_name": ["Current assets", "Current assets"],
            "status": ["alive", "failed"],
            "mean": [10.0, 8.0],
            "median": [9.0, 7.0],
            "std": [1.0, 1.0],
            "n_observations": [90, 10],
        }
    ).to_csv(class_feature_summary_path, index=False)
    _minimal_final_test_metrics().to_csv(final_test_metrics_path, index=False)

    prediction_rows = []
    for model in PAPER_MODELS:
        prediction_rows.extend(
            [
                {
                    "model": model,
                    "company_name": "A",
                    "year": 2000,
                    "actual_failed": 0,
                    "predicted_failed": 0,
                    "probability_failed": 0.1,
                },
                {
                    "model": model,
                    "company_name": "B",
                    "year": 2000,
                    "actual_failed": 1,
                    "predicted_failed": 1,
                    "probability_failed": 0.8,
                },
            ]
        )
    pd.DataFrame(prediction_rows).to_csv(final_test_predictions_path, index=False)

    pd.DataFrame(
        {
            "feature": ["X1", "X2"],
            "coefficient": [-1.0, 0.8],
            "absolute_coefficient": [1.0, 0.8],
            "direction": [
                "lower predicted failure risk",
                "higher predicted failure risk",
            ],
            "readable_name": ["Current assets", "Cost of goods sold"],
            "description": ["desc", "desc"],
        }
    ).to_csv(logistic_coefficients_path, index=False)
    pd.DataFrame(
        {
            "model": ["Random Forest", "Gradient Boosting"],
            "feature": ["X1", "X1"],
            "importance": [0.2, 0.3],
            "readable_name": ["Current assets", "Current assets"],
            "description": ["desc", "desc"],
        }
    ).to_csv(tree_feature_importance_path, index=False)
    pd.DataFrame(
        {
            "model": ["Logistic Regression"],
            "selection_rule": ["maximize_f1"],
            "threshold": [0.5],
            "precision_failed": [0.2],
            "recall_failed": [0.3],
            "f1_failed": [0.24],
            "predicted_failed_share": [0.1],
            "n_predicted_failed": [2],
        }
    ).to_csv(selected_thresholds_path, index=False)
    pd.DataFrame(
        {
            "n_components": [2, 3],
            "cumulative_explained_variance": [0.8, 0.9],
            "roc_auc": [0.6, 0.7],
            "pr_auc": [0.1, 0.2],
            "precision_failed": [0.1, 0.2],
            "recall_failed": [0.5, 0.6],
            "f1_failed": [0.17, 0.3],
        }
    ).to_csv(pca_logistic_results_path, index=False)

    paper_tables_dir = tmp_path / "paper" / "tables"
    paper_figures_dir = tmp_path / "paper" / "figures"

    save_paper_outputs(
        target_distribution_path=target_distribution_path,
        annual_failure_rate_path=annual_failure_rate_path,
        class_feature_summary_path=class_feature_summary_path,
        final_test_metrics_path=final_test_metrics_path,
        final_test_predictions_path=final_test_predictions_path,
        logistic_coefficients_path=logistic_coefficients_path,
        tree_feature_importance_path=tree_feature_importance_path,
        selected_thresholds_path=selected_thresholds_path,
        pca_logistic_results_path=pca_logistic_results_path,
        paper_tables_dir=paper_tables_dir,
        paper_figures_dir=paper_figures_dir,
    )

    assert (paper_tables_dir / "model_performance_summary.csv").exists()
    assert (paper_tables_dir / "selected_threshold_summary.csv").exists()
    assert (paper_figures_dir / "precision_recall_key_models.png").exists()
    assert (paper_figures_dir / "roc_curves_key_models.png").exists()
    assert (paper_figures_dir / "top_tree_feature_importance.png").exists()
