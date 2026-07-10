"""Create paper-ready summary tables and figures.

This module builds the curated ``outputs/paper/`` layer used by the LaTeX
paper. It deliberately depends on canonical generated project outputs such as
``outputs/tables/final_test_metrics.csv`` and
``outputs/tables/final_test_predictions.csv``. It does not fit models, choose
thresholds, change random seeds, or modify the empirical methodology.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

PAPER_MODELS = [
    "Majority-class baseline",
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
]

PAPER_CURVE_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
]

PAPER_METRICS = [
    "pr_auc",
    "recall_failed",
    "f1_failed",
]

PAPER_METRIC_LABELS = {
    "pr_auc": "PR-AUC",
    "recall_failed": "Failed recall",
    "f1_failed": "Failed F1",
}

PAPER_FINANCIAL_FEATURES = ["X8", "X6", "X11", "X1", "X17", "X15"]

PAPER_COLORS = {
    "Majority-class baseline": "#8c8c8c",
    "Logistic Regression": "#4c78a8",
    "Random Forest": "#59a14f",
    "Gradient Boosting": "#f28e2b",
}


def _round_columns(
    data: pd.DataFrame,
    columns: list[str],
    decimals: int = 3,
) -> pd.DataFrame:
    """Round selected columns while leaving other columns unchanged."""
    rounded = data.copy()
    rounded[columns] = rounded[columns].round(decimals)
    return rounded


def _paper_model_subset(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return the paper model subset in a stable narrative order."""
    return metrics.set_index("model").loc[PAPER_MODELS].reset_index()


def create_model_performance_summary(final_test_metrics: pd.DataFrame) -> pd.DataFrame:
    """Create the compact final-test performance table used in the paper."""
    summary = _paper_model_subset(final_test_metrics)
    summary.insert(0, "evaluation_sample", "Final test")

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "precision_failed",
        "recall_failed",
        "f1_failed",
    ]

    return _round_columns(
        summary[["evaluation_sample", "model", *metric_columns]],
        metric_columns,
    )


def create_confusion_matrix_summary(final_test_metrics: pd.DataFrame) -> pd.DataFrame:
    """Create the compact confusion-matrix table used in the paper."""
    summary = _paper_model_subset(final_test_metrics).rename(
        columns={
            "true_negative": "correctly_identified_alive",
            "false_positive": "false_alarms",
            "false_negative": "missed_failures",
            "true_positive": "detected_failures",
        }
    )
    summary.insert(0, "evaluation_sample", "Final test")

    columns = [
        "evaluation_sample",
        "model",
        "correctly_identified_alive",
        "false_alarms",
        "missed_failures",
        "detected_failures",
        "precision_failed",
        "recall_failed",
    ]
    summary = summary[columns].rename(
        columns={
            "precision_failed": "failed_precision",
            "recall_failed": "failed_recall",
        }
    )

    return _round_columns(summary, ["failed_precision", "failed_recall"])


def create_financial_median_summary(
    class_feature_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create the selected financial median comparison table."""
    medians = class_feature_summary.pivot(
        index=["feature", "readable_name"],
        columns="status",
        values="median",
    ).reset_index()

    medians = medians.rename(
        columns={
            "alive": "alive_median",
            "failed": "failed_median",
        }
    )
    medians["failed_minus_alive"] = medians["failed_median"] - medians["alive_median"]
    medians = medians.set_index("feature")
    ordered_features = [
        feature for feature in PAPER_FINANCIAL_FEATURES if feature in medians.index
    ]
    remaining_features = [
        feature for feature in medians.index if feature not in ordered_features
    ]
    medians = medians.loc[[*ordered_features, *remaining_features]].reset_index()

    columns = [
        "feature",
        "readable_name",
        "alive_median",
        "failed_median",
        "failed_minus_alive",
    ]

    return medians[columns].round(2)


def create_pca_summary(pca_logistic_results: pd.DataFrame) -> pd.DataFrame:
    """Create the compact PCA validation table used in the paper."""
    columns = [
        "n_components",
        "cumulative_explained_variance",
        "roc_auc",
        "pr_auc",
        "precision_failed",
        "recall_failed",
        "f1_failed",
    ]
    summary = pca_logistic_results[columns].copy()
    summary.insert(0, "evaluation_sample", "Internal validation")

    return _round_columns(summary, columns[1:])


def create_selected_threshold_summary(
    selected_thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """Create the compact threshold table used in the paper appendix."""
    summary = selected_thresholds.copy()
    summary.insert(0, "evaluation_sample", "Internal validation")

    metric_columns = [
        "threshold",
        "precision_failed",
        "recall_failed",
        "f1_failed",
        "predicted_failed_share",
    ]

    return _round_columns(summary, metric_columns)


def plot_paper_class_balance(
    target_distribution: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the paper-ready class distribution figure."""
    plot_data = target_distribution.set_index("status_label").loc[["alive", "failed"]]

    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    bars = ax.bar(
        plot_data.index,
        plot_data["count"],
        color=["#4c78a8", "#f28e2b"],
    )

    ax.set_title("Class Distribution in the Raw Dataset")
    ax.set_xlabel("Company status")
    ax.set_ylabel("Number of company-year observations")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, plot_data["count"].max() * 1.16)

    for bar, (_, row) in zip(bars, plot_data.iterrows(), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + plot_data["count"].max() * 0.02,
            f"{int(row['count']):,}\n({row['share']:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_paper_annual_failure_rate(
    annual_failure_rate: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the paper-ready annual failure-rate figure."""
    fig, ax = plt.subplots(figsize=(7.1, 4.1))
    ax.plot(
        annual_failure_rate["year"],
        annual_failure_rate["failure_rate"],
        marker="o",
        color="#4c78a8",
    )

    ax.set_title("Observed Annual Failure Rate")
    ax.set_xlabel("Year")
    ax.set_ylabel("Failure rate")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_paper_model_performance_summary(
    model_performance_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the paper-ready failed-class performance summary."""
    plot_data = model_performance_summary.set_index("model")[PAPER_METRICS]
    plot_data = plot_data.rename(columns=PAPER_METRIC_LABELS)

    fig, ax = plt.subplots(figsize=(8.1, 4.7))
    colors = [PAPER_COLORS.get(model, "#8c8c8c") for model in plot_data.index]
    plot_data.plot(kind="bar", ax=ax, color=colors)

    ax.set_title("Final-Test Performance on the Failed Class")
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_paper_confusion_matrix_summary(
    confusion_matrix_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot a paper-ready summary of classification outcomes."""
    plot_data = confusion_matrix_summary.set_index("model")[
        ["detected_failures", "missed_failures", "false_alarms"]
    ].rename(
        columns={
            "detected_failures": "Failures detected",
            "missed_failures": "Failures missed",
            "false_alarms": "False alarms",
        }
    )

    fig, ax = plt.subplots(figsize=(10.1, 4.7))
    plot_data.plot(kind="bar", ax=ax, color=["#59a14f", "#e15759", "#f28e2b"])

    ax.set_title("Final-Test Classification Outcomes")
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of observations")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_paper_precision_recall_key_models(
    final_test_predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot final-test precision-recall curves for the key models."""
    fig, ax = plt.subplots(figsize=(6.9, 4.9))

    for model_name in PAPER_CURVE_MODELS:
        model_predictions = final_test_predictions[
            final_test_predictions["model"] == model_name
        ]
        y_true = model_predictions["actual_failed"]
        y_score = model_predictions["probability_failed"]
        average_precision = average_precision_score(y_true, y_score)

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax.plot(
            recall,
            precision,
            color=PAPER_COLORS.get(model_name),
            label=f"{model_name} (AP = {average_precision:.3f})",
        )

    baseline_rate = final_test_predictions["actual_failed"].mean()
    ax.axhline(
        baseline_rate,
        linestyle="--",
        linewidth=1,
        color="#8c8c8c",
        label=f"Failure rate baseline ({baseline_rate:.1%})",
    )

    ax.set_title("Final-Test Precision-Recall Curves")
    ax.set_xlabel("Recall for failed firms")
    ax.set_ylabel("Precision for failed firms")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_paper_logistic_coefficients(
    logistic_coefficients: pd.DataFrame,
    output_path: Path,
    top_n: int = 12,
) -> None:
    """Plot the largest signed Logistic Regression coefficients."""
    plot_data = (
        logistic_coefficients.sort_values("absolute_coefficient", ascending=False)
        .head(top_n)
        .sort_values("coefficient")
    )
    colors = [
        "#e15759" if coefficient > 0 else "#4c78a8"
        for coefficient in plot_data["coefficient"]
    ]

    fig, ax = plt.subplots(figsize=(7.1, 5.1))
    ax.barh(plot_data["readable_name"], plot_data["coefficient"], color=colors)
    ax.axvline(0, linewidth=1, color="black")

    ax.set_title("Largest Logistic Regression Coefficients")
    ax.set_xlabel("Logistic coefficient (standardized predictors)")
    ax.set_ylabel("Financial variable")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_paper_tree_feature_importance(
    tree_feature_importance: pd.DataFrame,
    output_path: Path,
    top_n: int = 8,
) -> None:
    """Plot leading feature importances for Random Forest and Gradient Boosting."""
    models = ["Random Forest", "Gradient Boosting"]
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 5.35), sharex=False)

    for ax, model_name in zip(axes, models, strict=False):
        plot_data = (
            tree_feature_importance[tree_feature_importance["model"] == model_name]
            .sort_values("importance", ascending=False)
            .head(top_n)
            .sort_values("importance")
        )
        ax.barh(
            plot_data["readable_name"],
            plot_data["importance"],
            color=PAPER_COLORS.get(model_name),
        )
        ax.set_title(model_name)
        ax.set_xlabel("Feature importance")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle("Top Tree-Based Feature Importances", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_paper_pca_explained_variance(
    pca_logistic_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot cumulative explained variance for tested PCA component counts."""
    fig, ax = plt.subplots(figsize=(6.9, 4.4))
    ax.plot(
        pca_logistic_results["n_components"],
        pca_logistic_results["cumulative_explained_variance"],
        marker="o",
        color="#4c78a8",
    )

    ax.set_title("PCA Cumulative Explained Variance")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_paper_outputs(
    target_distribution_path: Path,
    annual_failure_rate_path: Path,
    class_feature_summary_path: Path,
    final_test_metrics_path: Path,
    final_test_predictions_path: Path,
    logistic_coefficients_path: Path,
    tree_feature_importance_path: Path,
    selected_thresholds_path: Path,
    pca_logistic_results_path: Path,
    paper_tables_dir: Path,
    paper_figures_dir: Path,
) -> None:
    """Create the complete paper-ready output layer."""
    target_distribution = pd.read_csv(target_distribution_path)
    annual_failure_rate = pd.read_csv(annual_failure_rate_path)
    class_feature_summary = pd.read_csv(class_feature_summary_path)
    final_test_metrics = pd.read_csv(final_test_metrics_path)
    final_test_predictions = pd.read_csv(final_test_predictions_path)
    logistic_coefficients = pd.read_csv(logistic_coefficients_path)
    tree_feature_importance = pd.read_csv(tree_feature_importance_path)
    selected_thresholds = pd.read_csv(selected_thresholds_path)
    pca_logistic_results = pd.read_csv(pca_logistic_results_path)

    model_performance_summary = create_model_performance_summary(final_test_metrics)
    confusion_matrix_summary = create_confusion_matrix_summary(final_test_metrics)
    financial_median_summary = create_financial_median_summary(class_feature_summary)
    pca_summary = create_pca_summary(pca_logistic_results)
    selected_threshold_summary = create_selected_threshold_summary(selected_thresholds)

    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    paper_figures_dir.mkdir(parents=True, exist_ok=True)

    model_performance_summary.to_csv(
        paper_tables_dir / "model_performance_summary.csv",
        index=False,
        float_format="%.3f",
    )
    confusion_matrix_summary.to_csv(
        paper_tables_dir / "confusion_matrix_summary.csv",
        index=False,
        float_format="%.3f",
    )
    financial_median_summary.to_csv(
        paper_tables_dir / "financial_median_summary.csv",
        index=False,
    )
    pca_summary.to_csv(paper_tables_dir / "pca_summary.csv", index=False)
    selected_threshold_summary.to_csv(
        paper_tables_dir / "selected_threshold_summary.csv",
        index=False,
    )

    plot_paper_class_balance(
        target_distribution=target_distribution,
        output_path=paper_figures_dir / "class_balance.png",
    )
    plot_paper_annual_failure_rate(
        annual_failure_rate=annual_failure_rate,
        output_path=paper_figures_dir / "annual_failure_rate.png",
    )
    plot_paper_model_performance_summary(
        model_performance_summary=model_performance_summary,
        output_path=paper_figures_dir / "model_performance_summary.png",
    )
    plot_paper_confusion_matrix_summary(
        confusion_matrix_summary=confusion_matrix_summary,
        output_path=paper_figures_dir / "confusion_matrix_summary.png",
    )
    plot_paper_precision_recall_key_models(
        final_test_predictions=final_test_predictions,
        output_path=paper_figures_dir / "precision_recall_key_models.png",
    )
    plot_paper_logistic_coefficients(
        logistic_coefficients=logistic_coefficients,
        output_path=paper_figures_dir / "top_logistic_coefficients.png",
    )
    plot_paper_tree_feature_importance(
        tree_feature_importance=tree_feature_importance,
        output_path=paper_figures_dir / "top_tree_feature_importance.png",
    )
    plot_paper_pca_explained_variance(
        pca_logistic_results=pca_logistic_results,
        output_path=paper_figures_dir / "pca_explained_variance.png",
    )
