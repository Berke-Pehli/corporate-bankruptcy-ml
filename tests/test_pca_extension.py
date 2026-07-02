"""Tests for the PCA Logistic Regression extension."""

import pandas as pd

from bankruptcy_ml.pca_extension import (
    build_pca_logistic_pipeline,
    create_pca_component_loadings,
    get_valid_component_grid,
)


def test_get_valid_component_grid_removes_too_large_values() -> None:
    """Check that component counts larger than feature count are removed."""
    result = get_valid_component_grid(
        n_features=5,
        component_grid=[2, 3, 5, 8, 10],
    )

    assert result == [2, 3, 5]


def test_build_pca_logistic_pipeline_contains_expected_steps() -> None:
    """Check that the PCA Logistic Regression pipeline has the expected steps."""
    pipeline = build_pca_logistic_pipeline(n_components=2)

    assert list(pipeline.named_steps) == ["scaler", "pca", "model"]


def test_create_pca_component_loadings_returns_readable_metadata() -> None:
    """Check that PCA loading table includes readable feature metadata."""
    data = pd.DataFrame(
        {
            "company_name": ["A", "B", "C", "D"],
            "year": [2000, 2000, 2001, 2001],
            "failed": [0, 1, 0, 1],
            "X1": [1.0, 2.0, 3.0, 4.0],
            "X2": [4.0, 3.0, 2.0, 1.0],
        }
    )
    feature_dictionary = pd.DataFrame(
        {
            "feature": ["X1", "X2"],
            "readable_name": ["Current assets", "Cost of goods sold"],
            "description": ["desc 1", "desc 2"],
        }
    )

    x = data[["X1", "X2"]]
    y = data["failed"]

    model = build_pca_logistic_pipeline(n_components=2)
    model.fit(x, y)

    result = create_pca_component_loadings(
        fitted_model=model,
        train_data=data,
        feature_dictionary=feature_dictionary,
    )

    assert "principal_component" in result.columns
    assert "readable_name" in result.columns
    assert set(result["feature"]) == {"X1", "X2"}