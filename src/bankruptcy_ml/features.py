"""Feature utilities for the bankruptcy prediction models.

This module defines helper functions for separating identifiers, predictors, and
the binary bankruptcy target. The modeling pipeline keeps company identifiers
out of the feature matrix to avoid memorizing firm identities.

The raw Kaggle dataset names financial statement variables as X1-X18. For
traceability, the modeling dataset keeps these original column names. For
tables, figures, README explanations, and the final paper, this module also
provides readable names and descriptions.

Inputs:
    - data/processed/train.csv
    - data/processed/test.csv

Feature design:
    - company_name is used for splitting only, not modeling
    - year is kept for diagnostics but excluded from the baseline feature matrix
    - X1-X18 financial variables are used as predictors
    - failed is the binary target

Readable feature metadata:
    - X1: Current assets
    - X2: Cost of goods sold
    - X3: Depreciation and amortization
    - X4: EBITDA
    - X5: Inventory
    - X6: Net income
    - X7: Total receivables
    - X8: Market value
    - X9: Net sales
    - X10: Total assets
    - X11: Total long-term debt
    - X12: EBIT
    - X13: Gross profit
    - X14: Total current liabilities
    - X15: Retained earnings
    - X16: Total revenue
    - X17: Total liabilities
    - X18: Total operating expenses
"""

from __future__ import annotations

import pandas as pd

from bankruptcy_ml.config import COMPANY_COLUMN, TARGET_COLUMN, YEAR_COLUMN


FEATURE_NAME_MAP = {
    "X1": "Current assets",
    "X2": "Cost of goods sold",
    "X3": "Depreciation and amortization",
    "X4": "EBITDA",
    "X5": "Inventory",
    "X6": "Net income",
    "X7": "Total receivables",
    "X8": "Market value",
    "X9": "Net sales",
    "X10": "Total assets",
    "X11": "Total long-term debt",
    "X12": "EBIT",
    "X13": "Gross profit",
    "X14": "Total current liabilities",
    "X15": "Retained earnings",
    "X16": "Total revenue",
    "X17": "Total liabilities",
    "X18": "Total operating expenses",
}


FEATURE_DESCRIPTION_MAP = {
    "X1": "Assets expected to be sold, converted into cash, or used within one year.",
    "X2": "Direct costs related to producing or selling the firm's goods and services.",
    "X3": "Depreciation of tangible assets and amortization of intangible assets.",
    "X4": "Earnings before interest, taxes, depreciation, and amortization.",
    "X5": "Goods and raw materials held by the firm for production or sale.",
    "X6": "Profit after expenses and costs have been deducted from revenue.",
    "X7": "Money owed to the firm by customers for delivered goods or services.",
    "X8": "Market capitalization or market value of the publicly traded company.",
    "X9": "Gross sales minus returns, allowances, and discounts.",
    "X10": "Total assets owned or controlled by the company.",
    "X11": "Debt obligations due after more than one year.",
    "X12": "Earnings before interest and taxes.",
    "X13": "Profit after subtracting costs related to manufacturing and selling.",
    "X14": "Short-term obligations due within one year.",
    "X15": "Accumulated profits retained in the business after dividends and losses.",
    "X16": "Total income from sales before subtracting expenses.",
    "X17": "Total debts and obligations owed to outside parties.",
    "X18": "Expenses incurred through normal business operations.",
}


def get_readable_feature_name(feature: str) -> str:
    """Return a readable name for a raw financial feature code.

    Args:
        feature: Raw feature name from the Kaggle dataset, such as ``X1``.

    Returns:
        A readable feature name if available, otherwise the original feature name.
    """
    return FEATURE_NAME_MAP.get(feature, feature)


def get_feature_description(feature: str) -> str:
    """Return a short financial description for a raw feature code.

    Args:
        feature: Raw feature name from the Kaggle dataset, such as ``X1``.

    Returns:
        A short description if available, otherwise an empty string.
    """
    return FEATURE_DESCRIPTION_MAP.get(feature, "")


def get_feature_columns(data: pd.DataFrame, include_year: bool = False) -> list[str]:
    """Return the financial predictor columns used for modeling.

    Args:
        data: Model-ready dataset.
        include_year: Whether to include the observation year as a predictor.

    Returns:
        A list of feature column names.
    """
    excluded_columns = {COMPANY_COLUMN, TARGET_COLUMN}

    if not include_year:
        excluded_columns.add(YEAR_COLUMN)

    return [column for column in data.columns if column not in excluded_columns]


def split_features_target(
    data: pd.DataFrame,
    include_year: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a model-ready dataset into predictors and target.

    Args:
        data: Model-ready dataset.
        include_year: Whether to include year as a predictor.

    Returns:
        A tuple containing the feature matrix and target vector.
    """
    feature_columns = get_feature_columns(data, include_year=include_year)

    x = data[feature_columns].copy()
    y = data[TARGET_COLUMN].copy()

    return x, y