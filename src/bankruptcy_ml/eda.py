"""Exploratory data analysis for the corporate bankruptcy dataset.

This module creates descriptive tables and one retained supporting figure that
help explain the dataset before machine learning models are evaluated. The
focus is on financial diagnostics that are useful for the README and final
paper.

Inputs:
    - data/processed/model_dataset.csv
    - data/processed/train.csv
    - data/processed/test.csv
    - outputs/tables/feature_dictionary.csv

Outputs:
    - outputs/tables/annual_failure_rate.csv
    - outputs/tables/train_test_year_distribution.csv
    - outputs/tables/class_feature_summary.csv
    - outputs/figures/key_feature_median_by_status.png

Purpose:
    These outputs document class imbalance over time, check whether train and
    test sets cover similar years, and compare key financial features between
    alive and failed firms.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bankruptcy_ml.config import TARGET_COLUMN, YEAR_COLUMN
from bankruptcy_ml.features import FEATURE_NAME_MAP
from bankruptcy_ml.visual_style import apply_project_style, save_figure, style_axis

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
    medians = class_feature_summary.pivot(
        index=["feature", "readable_name"],
        columns="status",
        values="median",
    )
    plot_data = medians.copy()
    relative_scale = (plot_data["failed"].abs() + plot_data["alive"].abs()) / 2
    plot_data["relative_median_difference"] = (
        plot_data["failed"] - plot_data["alive"]
    ) / relative_scale.replace(0, pd.NA)
    plot_data = plot_data.reset_index()
    plot_data["readable_name"] = pd.Categorical(
        plot_data["readable_name"],
        categories=[
            FEATURE_NAME_MAP.get(feature, feature) for feature in KEY_FEATURES_FOR_EDA
        ],
        ordered=True,
    )
    plot_data = plot_data.sort_values("readable_name")

    apply_project_style()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = [
        "#c44e52" if value > 0 else "#4c78a8"
        for value in plot_data["relative_median_difference"]
    ]
    bars = ax.barh(
        plot_data["readable_name"],
        plot_data["relative_median_difference"],
        color=colors,
    )
    max_abs_difference = plot_data["relative_median_difference"].abs().max()
    label_offset = max_abs_difference * 0.04

    ax.set_title("Relative Median Gap by Bankruptcy Status", pad=22)
    ax.text(
        0.5,
        1.01,
        "Descriptive comparison only; values do not imply causality.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4d4d4d",
    )
    ax.set_xlabel("Difference in group medians divided by average absolute magnitude")
    ax.set_ylabel("Financial variable")
    ax.axvline(0, linewidth=1, color="black")
    ax.set_xlim(-max_abs_difference * 1.25, max_abs_difference * 1.25)
    style_axis(ax, grid_axis="x", percent_y=False)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")

    for bar, value in zip(
        bars,
        plot_data["relative_median_difference"],
        strict=False,
    ):
        ax.text(
            value + (label_offset if value >= 0 else -label_offset),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0%}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
        )

    save_figure(fig, output_path)


def save_eda_outputs(
    model_dataset_path: Path,
    train_data_path: Path,
    test_data_path: Path,
    annual_failure_rate_path: Path,
    train_test_year_distribution_path: Path,
    class_feature_summary_path: Path,
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

    plot_key_feature_median_by_status(
        class_feature_summary=class_feature_summary,
        output_path=key_feature_median_by_status_figure_path,
    )
