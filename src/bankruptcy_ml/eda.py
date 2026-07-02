"""Exploratory data analysis for the corporate bankruptcy dataset.

This module creates descriptive tables and figures that help explain the dataset
before machine learning models are evaluated. The focus is on financial
diagnostics that are useful for the final paper, README, and oral exam.

Inputs:
    - data/processed/model_dataset.csv
    - data/processed/train.csv
    - data/processed/test.csv
    - outputs/tables/feature_dictionary.csv

Outputs:
    - outputs/tables/annual_failure_rate.csv
    - outputs/tables/train_test_year_distribution.csv
    - outputs/tables/class_feature_summary.csv
    - outputs/figures/annual_failure_rate.png
    - outputs/figures/train_test_year_distribution.png
    - outputs/figures/feature_correlation_heatmap.png
    - outputs/figures/key_feature_distributions_by_status.png
    - outputs/figures/key_feature_median_by_status.png

Purpose:
    These outputs document class imbalance over time, check whether train and
    test sets cover similar years, show correlations between financial
    variables, and compare key financial features between alive and failed firms.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bankruptcy_ml.config import TARGET_COLUMN, YEAR_COLUMN
from bankruptcy_ml.features import FEATURE_NAME_MAP, get_feature_columns


KEY_FEATURES_FOR_EDA = ["X8", "X6", "X11", "X1", "X17", "X15"]


def create_annual_failure_rate(data: pd.DataFrame) -> pd.DataFrame:
    """Create annual failure-rate statistics.

    Args:
        data: Model-ready company-year dataset.

    Returns:
        DataFrame with yearly observation counts, failed counts, alive counts,
        and failure rates.
    """
    annual = (
        data.groupby(YEAR_COLUMN)[TARGET_COLUMN]
        .agg(n_observations="count", n_failed="sum")
        .reset_index()
    )
    annual["n_alive"] = annual["n_observations"] - annual["n_failed"]
    annual["failure_rate"] = annual["n_failed"] / annual["n_observations"]

    return annual


def create_train_test_year_distribution(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create a train/test year distribution table.

    Args:
        train_data: Company-level training split.
        test_data: Company-level test split.

    Returns:
        DataFrame with counts and shares by split and year.
    """
    train_years = (
        train_data.groupby(YEAR_COLUMN)
        .size()
        .reset_index(name="n_observations")
        .assign(split="train")
    )
    test_years = (
        test_data.groupby(YEAR_COLUMN)
        .size()
        .reset_index(name="n_observations")
        .assign(split="test")
    )

    distribution = pd.concat([train_years, test_years], ignore_index=True)
    distribution["share_within_split"] = distribution[
        "n_observations"
    ] / distribution.groupby("split")["n_observations"].transform("sum")

    return distribution[["split", YEAR_COLUMN, "n_observations", "share_within_split"]]


def create_class_feature_summary(
    data: pd.DataFrame,
    key_features: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize key financial features by bankruptcy status.

    Args:
        data: Model-ready company-year dataset.
        key_features: Optional list of raw feature names to summarize.

    Returns:
        Summary table with mean and median values by class.
    """
    if key_features is None:
        key_features = KEY_FEATURES_FOR_EDA

    rows = []

    for feature in key_features:
        for status_value, status_name in [(0, "alive"), (1, "failed")]:
            subset = data.loc[data[TARGET_COLUMN] == status_value, feature]
            rows.append(
                {
                    "feature": feature,
                    "readable_name": FEATURE_NAME_MAP.get(feature, feature),
                    "status": status_name,
                    "mean": subset.mean(),
                    "median": subset.median(),
                    "std": subset.std(),
                    "n_observations": int(subset.shape[0]),
                }
            )

    return pd.DataFrame(rows)


def plot_annual_failure_rate(
    annual_failure_rate: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot annual failure rates.

    Args:
        annual_failure_rate: Output from ``create_annual_failure_rate``.
        output_path: Path where the figure should be saved.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        annual_failure_rate[YEAR_COLUMN],
        annual_failure_rate["failure_rate"],
        marker="o",
    )

    ax.set_title("Annual Failure Rate in the Bankruptcy Dataset")
    ax.set_xlabel("Year")
    ax.set_ylabel("Failure rate")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.1%}")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_train_test_year_distribution(
    year_distribution: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot train/test observation counts by year.

    Args:
        year_distribution: Output from ``create_train_test_year_distribution``.
        output_path: Path where the figure should be saved.
    """
    plot_data = year_distribution.pivot(
        index=YEAR_COLUMN,
        columns="split",
        values="n_observations",
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data.plot(kind="bar", ax=ax)

    ax.set_title("Train/Test Observation Distribution by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of company-year observations")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Split")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_feature_correlation_heatmap(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot a correlation heatmap for financial predictors.

    Args:
        data: Model-ready company-year dataset.
        output_path: Path where the heatmap should be saved.
    """
    feature_columns = get_feature_columns(data)
    correlation = data[feature_columns].corr()

    readable_labels = [
        FEATURE_NAME_MAP.get(feature, feature) for feature in feature_columns
    ]

    fig, ax = plt.subplots(figsize=(13, 10))
    image = ax.imshow(correlation, vmin=-1, vmax=1)

    ax.set_xticks(range(len(readable_labels)))
    ax.set_yticks(range(len(readable_labels)))
    ax.set_xticklabels(readable_labels, rotation=45, ha="right")
    ax.set_yticklabels(readable_labels)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Correlation Heatmap of Financial Predictors")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_key_feature_distributions_by_status(
    data: pd.DataFrame,
    output_path: Path,
    key_features: list[str] | None = None,
) -> None:
    """Plot key financial feature distributions by bankruptcy status.

    The variables are clipped visually by plotting values between the 1st and
    99th percentiles. This avoids extreme outliers making the distributions
    unreadable while leaving the underlying data unchanged.

    Args:
        data: Model-ready company-year dataset.
        output_path: Path where the figure should be saved.
        key_features: Optional list of raw feature names to plot.
    """
    if key_features is None:
        key_features = KEY_FEATURES_FOR_EDA

    n_cols = 2
    n_rows = (len(key_features) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 4 * n_rows))
    axes = axes.flatten()

    for ax, feature in zip(axes, key_features, strict=False):
        lower = data[feature].quantile(0.01)
        upper = data[feature].quantile(0.99)

        alive_values = data.loc[data[TARGET_COLUMN] == 0, feature].clip(lower, upper)
        failed_values = data.loc[data[TARGET_COLUMN] == 1, feature].clip(lower, upper)

        ax.hist(alive_values, bins=40, alpha=0.6, label="alive", density=True)
        ax.hist(failed_values, bins=40, alpha=0.6, label="failed", density=True)

        ax.set_title(FEATURE_NAME_MAP.get(feature, feature))
        ax.set_xlabel("Value, clipped at 1st/99th percentiles")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for unused_ax in axes[len(key_features) :]:
        unused_ax.axis("off")

    fig.suptitle("Key Financial Feature Distributions by Status", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_key_feature_median_by_status(
    class_feature_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot median values of key financial features by bankruptcy status.

    This chart is easier to interpret than distribution histograms for highly
    skewed financial variables. It compares typical values for alive and failed
    firms using medians rather than means, which are more sensitive to outliers.

    Args:
        class_feature_summary: Output from ``create_class_feature_summary``.
        output_path: Path where the median comparison figure should be saved.
    """
    plot_data = class_feature_summary.pivot(
        index="readable_name",
        columns="status",
        values="median",
    )

    preferred_order = [
        FEATURE_NAME_MAP.get(feature, feature) for feature in KEY_FEATURES_FOR_EDA
    ]
    plot_data = plot_data.reindex(preferred_order)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data.plot(kind="barh", ax=ax)

    ax.set_title("Median Financial Feature Values by Bankruptcy Status")
    ax.set_xlabel("Median value")
    ax.set_ylabel("Financial variable")
    ax.axvline(0, linewidth=1)
    ax.legend(title="Status")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_eda_outputs(
    model_dataset_path: Path,
    train_data_path: Path,
    test_data_path: Path,
    annual_failure_rate_path: Path,
    train_test_year_distribution_path: Path,
    class_feature_summary_path: Path,
    annual_failure_rate_figure_path: Path,
    train_test_year_distribution_figure_path: Path,
    feature_correlation_heatmap_path: Path,
    key_feature_distributions_figure_path: Path,
    key_feature_median_by_status_figure_path: Path,
) -> None:
    """Create and save all EDA tables and figures.

    Args:
        model_dataset_path: Path to the full model-ready dataset.
        train_data_path: Path to the train split.
        test_data_path: Path to the final test split.
        annual_failure_rate_path: Output path for annual failure-rate table.
        train_test_year_distribution_path: Output path for split-year table.
        class_feature_summary_path: Output path for class feature summary.
        annual_failure_rate_figure_path: Output path for annual failure figure.
        train_test_year_distribution_figure_path: Output path for year split figure.
        feature_correlation_heatmap_path: Output path for correlation heatmap.
        key_feature_distributions_figure_path: Output path for distribution figure.
        key_feature_median_by_status_figure_path: Output path for median comparison
            figure.
    """
    data = pd.read_csv(model_dataset_path)
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    annual_failure_rate = create_annual_failure_rate(data)
    year_distribution = create_train_test_year_distribution(train_data, test_data)
    class_feature_summary = create_class_feature_summary(data)

    annual_failure_rate_path.parent.mkdir(parents=True, exist_ok=True)
    train_test_year_distribution_path.parent.mkdir(parents=True, exist_ok=True)
    class_feature_summary_path.parent.mkdir(parents=True, exist_ok=True)

    annual_failure_rate.to_csv(annual_failure_rate_path, index=False)
    year_distribution.to_csv(train_test_year_distribution_path, index=False)
    class_feature_summary.to_csv(class_feature_summary_path, index=False)

    plot_annual_failure_rate(
        annual_failure_rate=annual_failure_rate,
        output_path=annual_failure_rate_figure_path,
    )
    plot_train_test_year_distribution(
        year_distribution=year_distribution,
        output_path=train_test_year_distribution_figure_path,
    )
    plot_feature_correlation_heatmap(
        data=data,
        output_path=feature_correlation_heatmap_path,
    )
    plot_key_feature_distributions_by_status(
        data=data,
        output_path=key_feature_distributions_figure_path,
    )
    plot_key_feature_median_by_status(
        class_feature_summary=class_feature_summary,
        output_path=key_feature_median_by_status_figure_path,
    )