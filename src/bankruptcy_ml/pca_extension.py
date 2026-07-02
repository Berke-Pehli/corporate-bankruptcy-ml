"""PCA Logistic Regression extension for bankruptcy prediction.

This module adds a PCA-based Logistic Regression extension to the bankruptcy
prediction project. PCA is used as a dimensionality-reduction method before
Logistic Regression.

Inputs:
    - data/processed/train.csv
    - outputs/tables/feature_dictionary.csv

Outputs:
    - outputs/tables/pca_logistic_results.csv
    - outputs/tables/pca_component_loadings.csv
    - outputs/figures/pca_explained_variance.png
    - outputs/figures/pca_logistic_metric_comparison.png

Methodological note:
    PCA is included as a lecture-related extension, not as the main model. It can
    summarize correlated financial predictors into orthogonal components, but it
    makes interpretation less direct than models using original financial
    variables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bankruptcy_ml.config import RANDOM_STATE, VALIDATION_SIZE
from bankruptcy_ml.evaluation import (
    evaluate_binary_classifier,
    get_probability_failed,
)
from bankruptcy_ml.features import get_feature_columns, split_features_target
from bankruptcy_ml.splitting import create_company_level_split


def build_pca_logistic_pipeline(n_components: int) -> Pipeline:
    """Build a PCA plus Logistic Regression pipeline.

    The pipeline first standardizes financial predictors, then applies PCA, and
    finally fits a class-weighted Logistic Regression model.

    Args:
        n_components: Number of principal components to retain.

    Returns:
        An unfitted scikit-learn Pipeline.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    l1_ratio=0.0,
                    solver="saga",
                    class_weight="balanced",
                    max_iter=5_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def get_valid_component_grid(
    n_features: int,
    component_grid: list[int],
) -> list[int]:
    """Return PCA component values that do not exceed the number of features.

    Args:
        n_features: Number of available predictors.
        component_grid: Candidate component counts.

    Returns:
        Valid PCA component counts.
    """
    return sorted({value for value in component_grid if 1 <= value <= n_features})


def evaluate_pca_logistic_models(
    train_data: pd.DataFrame,
    component_grid: list[int],
) -> tuple[pd.DataFrame, Pipeline]:
    """Evaluate PCA Logistic Regression models on an internal validation split.

    Args:
        train_data: Company-level training dataset.
        component_grid: Candidate PCA component counts.

    Returns:
        A tuple containing:
            - validation metric table for PCA Logistic Regression models
            - the fitted PCA Logistic Regression model with the best PR-AUC
    """
    model_train, validation = create_company_level_split(
        train_data,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    x_train, y_train = split_features_target(model_train)
    x_valid, y_valid = split_features_target(validation)

    valid_components = get_valid_component_grid(
        n_features=x_train.shape[1],
        component_grid=component_grid,
    )

    rows = []
    best_model = None
    best_pr_auc = -1.0

    for n_components in valid_components:
        model_name = f"PCA Logistic Regression ({n_components} components)"
        model = build_pca_logistic_pipeline(n_components=n_components)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_valid)
        probability_failed = get_probability_failed(model, x_valid)

        metrics = evaluate_binary_classifier(
            model_name=model_name,
            y_true=y_valid,
            y_pred=y_pred,
            probability_failed=probability_failed,
        )
        metrics["n_components"] = n_components
        metrics["cumulative_explained_variance"] = float(
            model.named_steps["pca"].explained_variance_ratio_.sum()
        )

        rows.append(metrics)

        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            best_model = model

    if best_model is None:
        msg = "No PCA Logistic Regression model was fitted."
        raise RuntimeError(msg)

    return pd.DataFrame(rows), best_model


def create_pca_explained_variance_table(train_data: pd.DataFrame) -> pd.DataFrame:
    """Create cumulative explained variance table for all PCA components.

    Args:
        train_data: Company-level training dataset.

    Returns:
        Table with component numbers, explained variance ratios, and cumulative
        explained variance.
    """
    x_train, _ = split_features_target(train_data)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=x_train.shape[1], random_state=RANDOM_STATE)),
        ]
    )
    pipeline.fit(x_train)

    explained_variance = pipeline.named_steps["pca"].explained_variance_ratio_

    return pd.DataFrame(
        {
            "component": range(1, len(explained_variance) + 1),
            "explained_variance_ratio": explained_variance,
            "cumulative_explained_variance": explained_variance.cumsum(),
        }
    )


def create_pca_component_loadings(
    fitted_model: Pipeline,
    train_data: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
) -> pd.DataFrame:
    """Create PCA component loading table for the selected PCA model.

    Args:
        fitted_model: Fitted PCA Logistic Regression pipeline.
        train_data: Training dataset used to identify feature columns.
        feature_dictionary: Feature dictionary with readable names.

    Returns:
        Table of feature loadings for each retained principal component.
    """
    feature_columns = get_feature_columns(train_data)
    pca = fitted_model.named_steps["pca"]

    loadings = pd.DataFrame(
        pca.components_,
        columns=feature_columns,
        index=[f"PC{index}" for index in range(1, pca.n_components_ + 1)],
    )

    loading_table = (
        loadings.reset_index(names="principal_component")
        .melt(
            id_vars="principal_component",
            var_name="feature",
            value_name="loading",
        )
    )
    loading_table["absolute_loading"] = loading_table["loading"].abs()

    metadata = feature_dictionary[["feature", "readable_name", "description"]]

    loading_table = loading_table.merge(metadata, on="feature", how="left")

    return loading_table.sort_values(
        ["principal_component", "absolute_loading"],
        ascending=[True, False],
    )


def plot_pca_explained_variance(
    explained_variance: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot cumulative explained variance by number of PCA components.

    Args:
        explained_variance: PCA explained variance table.
        output_path: Path where the figure should be saved.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        explained_variance["component"],
        explained_variance["cumulative_explained_variance"],
        marker="o",
    )

    ax.axhline(0.8, linestyle="--", linewidth=1, label="80% variance")
    ax.axhline(0.9, linestyle="--", linewidth=1, label="90% variance")
    ax.axhline(0.95, linestyle="--", linewidth=1, label="95% variance")

    ax.set_title("PCA Cumulative Explained Variance")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_pca_logistic_metric_comparison(
    pca_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot PCA Logistic Regression metrics by component count.

    Args:
        pca_results: PCA Logistic Regression validation metric table.
        output_path: Path where the figure should be saved.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        pca_results["n_components"],
        pca_results["roc_auc"],
        marker="o",
        label="ROC-AUC",
    )
    ax.plot(
        pca_results["n_components"],
        pca_results["pr_auc"],
        marker="o",
        label="PR-AUC",
    )
    ax.plot(
        pca_results["n_components"],
        pca_results["f1_failed"],
        marker="o",
        label="F1 failed",
    )

    ax.set_title("PCA Logistic Regression Validation Performance")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_pca_extension_outputs(
    train_data_path: Path,
    feature_dictionary_path: Path,
    pca_logistic_results_path: Path,
    pca_component_loadings_path: Path,
    pca_explained_variance_figure_path: Path,
    pca_logistic_metric_comparison_figure_path: Path,
    component_grid: list[int],
) -> None:
    """Create and save all PCA extension outputs.

    Args:
        train_data_path: Path to the company-level training dataset.
        feature_dictionary_path: Path to the feature dictionary.
        pca_logistic_results_path: Output path for PCA Logistic Regression metrics.
        pca_component_loadings_path: Output path for PCA component loadings.
        pca_explained_variance_figure_path: Output path for explained variance plot.
        pca_logistic_metric_comparison_figure_path: Output path for PCA metric plot.
        component_grid: Candidate PCA component counts.
    """
    train_data = pd.read_csv(train_data_path)
    feature_dictionary = pd.read_csv(feature_dictionary_path)

    pca_results, best_model = evaluate_pca_logistic_models(
        train_data=train_data,
        component_grid=component_grid,
    )
    explained_variance = create_pca_explained_variance_table(train_data)
    component_loadings = create_pca_component_loadings(
        fitted_model=best_model,
        train_data=train_data,
        feature_dictionary=feature_dictionary,
    )

    pca_logistic_results_path.parent.mkdir(parents=True, exist_ok=True)
    pca_component_loadings_path.parent.mkdir(parents=True, exist_ok=True)

    pca_results.to_csv(pca_logistic_results_path, index=False)
    component_loadings.to_csv(pca_component_loadings_path, index=False)

    plot_pca_explained_variance(
        explained_variance=explained_variance,
        output_path=pca_explained_variance_figure_path,
    )
    plot_pca_logistic_metric_comparison(
        pca_results=pca_results,
        output_path=pca_logistic_metric_comparison_figure_path,
    )