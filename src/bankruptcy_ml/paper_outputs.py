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
from matplotlib.patches import Patch
from sklearn.metrics import average_precision_score, precision_recall_curve

from bankruptcy_ml.visual_style import (
    BASELINE_COLOR,
    DIRECTION_COLORS,
    MODEL_COLORS,
    OUTCOME_COLORS,
    apply_project_style,
    save_figure,
    style_axis,
)

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
    "precision_failed",
    "recall_failed",
    "f1_failed",
]

PAPER_METRIC_LABELS = {
    "pr_auc": "PR-AUC",
    "precision_failed": "Failed precision",
    "recall_failed": "Failed recall",
    "f1_failed": "Failed F1",
}

PAPER_FINANCIAL_FEATURES = ["X8", "X6", "X11", "X1", "X17", "X15"]


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

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(
        plot_data.index,
        plot_data["count"],
        color=[MODEL_COLORS["Logistic Regression"], OUTCOME_COLORS["missed"]],
    )

    ax.set_title("Class Distribution in the Raw Dataset")
    ax.set_xlabel("Company status")
    ax.set_ylabel("Number of company-year observations")
    ax.set_ylim(0, plot_data["count"].max() * 1.12)
    style_axis(ax)

    for bar, (_, row) in zip(bars, plot_data.iterrows(), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + plot_data["count"].max() * 0.015,
            f"{int(row['count']):,}\n({row['share']:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_figure(fig, output_path)


def plot_paper_annual_failure_rate(
    annual_failure_rate: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the paper-ready annual failure-rate figure."""
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.plot(
        annual_failure_rate["year"],
        annual_failure_rate["failure_rate"],
        marker="o",
        color=MODEL_COLORS["Logistic Regression"],
    )

    ax.set_title("Observed Annual Failure Rate")
    ax.set_xlabel("Year")
    ax.set_ylabel("Failure rate")
    ax.set_ylim(0, annual_failure_rate["failure_rate"].max() * 1.18)
    style_axis(ax, grid_axis="both", percent_y=True, integer_x=True)

    peak = annual_failure_rate.loc[annual_failure_rate["failure_rate"].idxmax()]
    final = annual_failure_rate.sort_values("year").iloc[-1]
    for label, row, xytext, text_alignment, vertical_alignment in [
        ("Peak", peak, (12, -34), "left", "center"),
        ("Final year", final, (-28, 34), "right", "bottom"),
    ]:
        ax.annotate(
            f"{label}: {row['failure_rate']:.1%}",
            xy=(row["year"], row["failure_rate"]),
            xytext=xytext,
            textcoords="offset points",
            ha=text_alignment,
            va=vertical_alignment,
            arrowprops={"arrowstyle": "->", "color": "#666666", "linewidth": 0.8},
            fontsize=8.5,
            color="#4d4d4d",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )

    save_figure(fig, output_path)


def plot_paper_model_performance_summary(
    model_performance_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the paper-ready failed-class performance summary."""
    plot_data = model_performance_summary.set_index("model")[PAPER_METRICS]
    plot_data = plot_data.rename(columns=PAPER_METRIC_LABELS)

    fig, ax = plt.subplots(figsize=(8.1, 4.6))
    metric_positions = list(range(len(plot_data.columns)))
    offsets = [-0.18, -0.06, 0.06, 0.18]
    for offset, (model_name, row) in zip(offsets, plot_data.iterrows(), strict=False):
        ax.scatter(
            [position + offset for position in metric_positions],
            row.to_numpy(),
            marker="o",
            s=55,
            label=model_name,
            color=MODEL_COLORS.get(model_name, BASELINE_COLOR),
        )

    ax.set_title("Final-Test Ranking and Failed-Class Performance")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Metric value")
    ax.set_xticks(list(metric_positions))
    ax.set_xticklabels(plot_data.columns)
    ax.set_ylim(0, 1)
    style_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)

    save_figure(fig, output_path)


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

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), width_ratios=[1.1, 1])
    failure_data = plot_data[["Failures detected", "Failures missed"]]
    false_alarm_data = plot_data[["False alarms"]]

    failure_data.plot(
        kind="bar",
        ax=axes[0],
        color=[OUTCOME_COLORS["detected"], OUTCOME_COLORS["missed"]],
    )
    false_alarm_data.plot(
        kind="bar",
        ax=axes[1],
        color=[OUTCOME_COLORS["false_alarm"]],
        legend=False,
    )
    for ax in axes:
        max_bar_height = max(
            patch.get_height() for container in ax.containers for patch in container
        )
        ax.set_ylim(0, max_bar_height * 1.12 if max_bar_height else 1)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", padding=2, fontsize=8)

    axes[0].set_title("Failure detection")
    axes[0].set_ylabel("Number of failed observations")
    axes[0].legend(loc="upper right")
    axes[1].set_title("Screening burden")
    axes[1].set_ylabel("False alarms")
    for ax in axes:
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=20)
        style_axis(ax)

    fig.suptitle("Final-Test Classification Outcomes", fontsize=12)
    save_figure(fig, output_path)


def plot_paper_precision_recall_key_models(
    final_test_predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot final-test precision-recall curves for the key models."""
    fig, ax = plt.subplots(figsize=(7.2, 4.9))

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
            color=MODEL_COLORS.get(model_name),
            linewidth=2.2,
            label=f"{model_name} (AP = {average_precision:.3f})",
        )

    baseline_rate = final_test_predictions["actual_failed"].mean()
    ax.axhline(
        baseline_rate,
        linestyle="--",
        linewidth=1,
        color=BASELINE_COLOR,
        label=f"Prevalence baseline ({baseline_rate:.1%})",
    )

    ax.set_title("Final-Test Precision-Recall Curves", pad=22)
    ax.text(
        0.5,
        1.01,
        "Random Forest and Gradient Boosting have very similar AP.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4d4d4d",
    )
    ax.set_xlabel("Recall for failed firms")
    ax.set_ylabel("Precision for failed firms")
    style_axis(ax, grid_axis="both")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    save_figure(fig, output_path)


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
        DIRECTION_COLORS["positive"]
        if coefficient > 0
        else DIRECTION_COLORS["negative"]
        for coefficient in plot_data["coefficient"]
    ]

    fig, ax = plt.subplots(figsize=(7.3, 5.1))
    bars = ax.barh(plot_data["readable_name"], plot_data["coefficient"], color=colors)
    ax.axvline(0, linewidth=1, color="black")

    ax.set_title("Largest Standardized Logistic Coefficients")
    ax.set_xlabel("Logistic coefficient (standardized predictors)")
    ax.set_ylabel("Financial variable")
    x_min = plot_data["coefficient"].min()
    x_max = plot_data["coefficient"].max()
    ax.set_xlim(x_min - 0.25, x_max + 0.25)
    style_axis(ax, grid_axis="x")

    for bar, coefficient in zip(bars, plot_data["coefficient"], strict=False):
        offset = 0.015 if coefficient >= 0 else -0.015
        ax.text(
            coefficient + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{coefficient:.2f}",
            va="center",
            ha="left" if coefficient >= 0 else "right",
            fontsize=8,
        )
    ax.legend(
        handles=[
            Patch(
                facecolor=DIRECTION_COLORS["positive"],
                label="Higher predicted failure risk",
            ),
            Patch(
                facecolor=DIRECTION_COLORS["negative"],
                label="Lower predicted failure risk",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
    )

    save_figure(fig, output_path)


def plot_paper_tree_feature_importance(
    tree_feature_importance: pd.DataFrame,
    output_path: Path,
    top_n: int = 8,
) -> None:
    """Plot leading feature importances for Random Forest and Gradient Boosting."""
    models = ["Random Forest", "Gradient Boosting"]
    top_features = (
        tree_feature_importance[tree_feature_importance["model"].isin(models)]
        .groupby(["feature", "readable_name"], as_index=False)["importance"]
        .max()
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance")
    )
    feature_order = top_features["readable_name"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.3), sharey=True)

    for ax, model_name in zip(axes, models, strict=False):
        model_data = tree_feature_importance[
            tree_feature_importance["model"] == model_name
        ].set_index("readable_name")
        plot_data = pd.DataFrame(
            {
                "readable_name": feature_order,
                "importance": [
                    model_data.loc[feature, "importance"]
                    if feature in model_data.index
                    else 0.0
                    for feature in feature_order
                ],
            }
        )
        bars = ax.barh(
            plot_data["readable_name"],
            plot_data["importance"],
            color=MODEL_COLORS.get(model_name),
        )
        ax.set_title(model_name)
        ax.set_xlabel("Model-specific feature importance")
        style_axis(ax, grid_axis="x")
        for bar, value in zip(bars, plot_data["importance"], strict=False):
            ax.text(
                value + plot_data["importance"].max() * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                fontsize=7.8,
            )

    fig.suptitle(
        "Top Tree-Based Feature Importances\n"
        "Importance magnitudes are interpreted within each model.",
        fontsize=12,
    )
    save_figure(fig, output_path)


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
        color=MODEL_COLORS["Logistic Regression"],
    )

    for threshold in [0.8, 0.9, 0.95]:
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1,
            color=BASELINE_COLOR,
            alpha=0.7,
        )
        ax.text(
            pca_logistic_results["n_components"].max(),
            threshold + 0.01,
            f"{threshold:.0%}",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#4d4d4d",
        )

    ax.set_title("Explained Variance at Evaluated PCA Specifications")
    ax.set_xlabel("Number of PCA components evaluated")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.05)
    style_axis(ax, grid_axis="both", percent_y=True, integer_x=True)

    save_figure(fig, output_path)


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
    apply_project_style()
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
