"""Interpret fitted bankruptcy prediction models.

This module creates coefficient and feature-importance outputs for the
bankruptcy prediction project. The goal is to connect model results back to
financial interpretation, while avoiding causal claims.

Inputs:
    - fitted model objects from outputs/models/
    - data/processed/train.csv
    - outputs/tables/feature_dictionary.csv

Outputs:
    - outputs/tables/logistic_coefficients.csv
    - outputs/tables/tree_feature_importance.csv
    - outputs/figures/logistic_coefficients.png
    - outputs/figures/tree_feature_importance.png

Interpretation:
    - Logistic Regression coefficients indicate associations with log-odds of
      failure, not direct probability changes.
    - Tree feature importances indicate predictive contribution inside the
      fitted model, not causal effects.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from bankruptcy_ml.features import get_feature_columns


def load_feature_dictionary(feature_dictionary_path: Path) -> pd.DataFrame:
    """Load the feature dictionary.

    Args:
        feature_dictionary_path: Path to the feature dictionary CSV.

    Returns:
        Feature dictionary containing raw feature names, readable names, and
        descriptions.
    """
    return pd.read_csv(feature_dictionary_path)


def add_feature_metadata(
    data: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
) -> pd.DataFrame:
    """Add readable feature names and descriptions to an interpretation table.

    Args:
        data: Interpretation table containing a ``feature`` column.
        feature_dictionary: Feature dictionary with readable names.

    Returns:
        Interpretation table enriched with readable names and descriptions.
    """
    metadata_columns = ["feature", "readable_name", "description"]

    return data.merge(
        feature_dictionary[metadata_columns],
        on="feature",
        how="left",
    )


def extract_logistic_coefficients(
    model_path: Path,
    train_data_path: Path,
    feature_dictionary_path: Path,
) -> pd.DataFrame:
    """Extract coefficients from a fitted Logistic Regression pipeline.

    The Logistic Regression model is stored as a scikit-learn pipeline with a
    scaler and a classifier. Coefficients are extracted from the final model
    step. Since the model is fitted on standardized predictors, coefficient
    magnitudes are comparable across financial variables.

    Args:
        model_path: Path to the fitted Logistic Regression pipeline.
        train_data_path: Path to the training dataset.
        feature_dictionary_path: Path to the feature dictionary.

    Returns:
        DataFrame containing coefficients, absolute coefficients, and readable
        feature metadata.
    """
    model = joblib.load(model_path)
    train_data = pd.read_csv(train_data_path)
    feature_dictionary = load_feature_dictionary(feature_dictionary_path)

    feature_columns = get_feature_columns(train_data)
    coefficients = model.named_steps["model"].coef_[0]

    coefficient_table = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": coefficients,
        }
    )
    coefficient_table["absolute_coefficient"] = coefficient_table[
        "coefficient"
    ].abs()
    coefficient_table["direction"] = coefficient_table["coefficient"].apply(
        lambda value: "higher predicted failure risk" if value > 0 else "lower predicted failure risk"
    )

    coefficient_table = add_feature_metadata(
        data=coefficient_table,
        feature_dictionary=feature_dictionary,
    )

    return coefficient_table.sort_values(
        "absolute_coefficient",
        ascending=False,
    )


def extract_tree_feature_importance(
    model_paths: dict[str, Path],
    train_data_path: Path,
    feature_dictionary_path: Path,
) -> pd.DataFrame:
    """Extract feature importances from fitted tree-based models.

    Args:
        model_paths: Dictionary mapping model names to fitted model paths.
        train_data_path: Path to the training dataset.
        feature_dictionary_path: Path to the feature dictionary.

    Returns:
        DataFrame containing feature importances for each tree-based model.
    """
    train_data = pd.read_csv(train_data_path)
    feature_dictionary = load_feature_dictionary(feature_dictionary_path)
    feature_columns = get_feature_columns(train_data)

    rows = []

    for model_name, model_path in model_paths.items():
        model = joblib.load(model_path)

        if not hasattr(model, "feature_importances_"):
            msg = f"Model does not expose feature_importances_: {model_name}"
            raise ValueError(msg)

        for feature, importance in zip(
            feature_columns,
            model.feature_importances_,
            strict=False,
        ):
            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "importance": importance,
                }
            )

    importance_table = pd.DataFrame(rows)
    importance_table = add_feature_metadata(
        data=importance_table,
        feature_dictionary=feature_dictionary,
    )

    return importance_table.sort_values(
        ["model", "importance"],
        ascending=[True, False],
    )


def save_interpretation_tables(
    logistic_model_path: Path,
    tree_model_paths: dict[str, Path],
    train_data_path: Path,
    feature_dictionary_path: Path,
    logistic_coefficients_path: Path,
    tree_feature_importance_path: Path,
) -> None:
    """Create and save interpretation tables.

    Args:
        logistic_model_path: Path to the fitted interpretable Logistic Regression.
        tree_model_paths: Paths to fitted tree-based models.
        train_data_path: Path to training data.
        feature_dictionary_path: Path to feature dictionary.
        logistic_coefficients_path: Output path for coefficient table.
        tree_feature_importance_path: Output path for tree-importance table.
    """
    logistic_coefficients = extract_logistic_coefficients(
        model_path=logistic_model_path,
        train_data_path=train_data_path,
        feature_dictionary_path=feature_dictionary_path,
    )
    tree_feature_importance = extract_tree_feature_importance(
        model_paths=tree_model_paths,
        train_data_path=train_data_path,
        feature_dictionary_path=feature_dictionary_path,
    )

    logistic_coefficients_path.parent.mkdir(parents=True, exist_ok=True)
    tree_feature_importance_path.parent.mkdir(parents=True, exist_ok=True)

    logistic_coefficients.to_csv(logistic_coefficients_path, index=False)
    tree_feature_importance.to_csv(tree_feature_importance_path, index=False)


def plot_logistic_coefficients(
    coefficients: pd.DataFrame,
    output_path: Path,
    top_n: int = 12,
) -> None:
    """Plot the largest Logistic Regression coefficients by absolute magnitude.

    Args:
        coefficients: Coefficient table created by
            ``extract_logistic_coefficients``.
        output_path: Path where the figure should be saved.
        top_n: Number of features to display.
    """
    plot_data = (
        coefficients.sort_values("absolute_coefficient", ascending=False)
        .head(top_n)
        .sort_values("coefficient")
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(plot_data["readable_name"], plot_data["coefficient"])
    ax.axvline(0, linewidth=1)

    ax.set_title("Largest Logistic Regression Coefficients")
    ax.set_xlabel("Coefficient on standardized feature")
    ax.set_ylabel("Financial variable")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_tree_feature_importance(
    feature_importance: pd.DataFrame,
    output_path: Path,
    model_name: str = "Gradient Boosting",
    top_n: int = 12,
) -> None:
    """Plot top feature importances for one tree-based model.

    Gradient Boosting is used by default because it currently has the strongest
    validation PR-AUC among the fitted models.

    Args:
        feature_importance: Tree feature-importance table.
        output_path: Path where the figure should be saved.
        model_name: Tree-based model to visualize.
        top_n: Number of features to display.
    """
    plot_data = (
        feature_importance[feature_importance["model"] == model_name]
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance")
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(plot_data["readable_name"], plot_data["importance"])

    ax.set_title(f"Top Feature Importances: {model_name}")
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Financial variable")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_interpretation_figures(
    logistic_coefficients_path: Path,
    tree_feature_importance_path: Path,
    logistic_coefficients_figure_path: Path,
    tree_feature_importance_figure_path: Path,
) -> None:
    """Create and save interpretation figures.

    Args:
        logistic_coefficients_path: Path to coefficient table.
        tree_feature_importance_path: Path to tree-importance table.
        logistic_coefficients_figure_path: Output path for coefficient figure.
        tree_feature_importance_figure_path: Output path for importance figure.
    """
    logistic_coefficients = pd.read_csv(logistic_coefficients_path)
    tree_feature_importance = pd.read_csv(tree_feature_importance_path)

    plot_logistic_coefficients(
        coefficients=logistic_coefficients,
        output_path=logistic_coefficients_figure_path,
    )
    plot_tree_feature_importance(
        feature_importance=tree_feature_importance,
        output_path=tree_feature_importance_figure_path,
    )