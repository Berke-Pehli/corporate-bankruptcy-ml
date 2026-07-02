"""Tests for model interpretation utilities."""

import pandas as pd

from bankruptcy_ml.interpretation import add_feature_metadata


def test_add_feature_metadata_adds_readable_names() -> None:
    """Check that readable feature names are merged into interpretation tables."""
    interpretation_table = pd.DataFrame(
        {
            "feature": ["X1", "X2"],
            "coefficient": [0.5, -0.2],
        }
    )
    feature_dictionary = pd.DataFrame(
        {
            "feature": ["X1", "X2"],
            "readable_name": ["Current assets", "Cost of goods sold"],
            "description": ["desc 1", "desc 2"],
        }
    )

    result = add_feature_metadata(
        data=interpretation_table,
        feature_dictionary=feature_dictionary,
    )

    assert "readable_name" in result.columns
    assert list(result["readable_name"]) == ["Current assets", "Cost of goods sold"]