"""pytask tasks for the PCA Logistic Regression extension.

This task module adds a PCA-based Logistic Regression extension to the
bankruptcy prediction project. PCA is included as a course-related
dimensionality-reduction method and not as the main model.

Run:
    pixi run build

Inputs:
    - data/processed/train.csv
    - outputs/tables/feature_dictionary.csv

Outputs:
    - outputs/tables/pca_logistic_results.csv
    - outputs/tables/pca_component_loadings.csv
    - outputs/figures/pca_explained_variance.png
    - outputs/figures/pca_logistic_metric_comparison.png
"""

from pathlib import Path

from bankruptcy_ml.config import (
    FEATURE_DICTIONARY_PATH,
    PCA_COMPONENT_GRID,
    PCA_COMPONENT_LOADINGS_PATH,
    PCA_EXPLAINED_VARIANCE_FIGURE_PATH,
    PCA_LOGISTIC_METRIC_COMPARISON_FIGURE_PATH,
    PCA_LOGISTIC_RESULTS_PATH,
    TRAIN_DATA_PATH,
)
from bankruptcy_ml.pca_extension import save_pca_extension_outputs


def task_create_pca_extension_outputs(
    depends_on: tuple[Path, Path] = (
        TRAIN_DATA_PATH,
        FEATURE_DICTIONARY_PATH,
    ),
    produces: tuple[Path, Path, Path, Path] = (
        PCA_LOGISTIC_RESULTS_PATH,
        PCA_COMPONENT_LOADINGS_PATH,
        PCA_EXPLAINED_VARIANCE_FIGURE_PATH,
        PCA_LOGISTIC_METRIC_COMPARISON_FIGURE_PATH,
    ),
) -> None:
    """Create PCA Logistic Regression extension outputs."""
    train_data_path, feature_dictionary_path = depends_on
    (
        pca_logistic_results_path,
        pca_component_loadings_path,
        pca_explained_variance_figure_path,
        pca_logistic_metric_comparison_figure_path,
    ) = produces

    save_pca_extension_outputs(
        train_data_path=train_data_path,
        feature_dictionary_path=feature_dictionary_path,
        pca_logistic_results_path=pca_logistic_results_path,
        pca_component_loadings_path=pca_component_loadings_path,
        pca_explained_variance_figure_path=pca_explained_variance_figure_path,
        pca_logistic_metric_comparison_figure_path=pca_logistic_metric_comparison_figure_path,
        component_grid=PCA_COMPONENT_GRID,
    )