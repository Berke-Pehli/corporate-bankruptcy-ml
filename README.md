# Corporate Bankruptcy Prediction with Machine Learning

This project predicts corporate bankruptcy risk using company-year financial data
from American public companies. It compares an interpretable Logistic Regression
benchmark with more flexible tree-based machine learning models.

## Research Question

Can machine learning models predict corporate bankruptcy using company-year
financial indicators, and do tree-based models improve out-of-sample performance
compared with Logistic Regression?

## Project Motivation

Bankruptcy prediction is an important financial risk-management problem for
investors, creditors, analysts, and policymakers. The project focuses on
out-of-sample classification performance and the interpretation of financial
variables that are useful for identifying firms with elevated bankruptcy risk.

## Dataset

The project uses the American Companies Bankruptcy Prediction dataset from
Kaggle. The raw file should be stored as:

```text
data/raw/american_bankruptcy.csv
```

The dataset contains company-year observations and a binary status label
indicating whether a company is alive or failed.

## Methods

The planned methods are:

- Majority-class baseline
- Logistic Regression
- L1/L2 regularized Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The project is designed around concepts from the Machine Learning for Finance
course, especially classification, Logistic Regression, regularization,
resampling, tree-based models, and the interpretability-flexibility trade-off.

## Reproducibility

The project uses Pixi, pytask, and pytest.

To build the project outputs:

```bash
pixi run build
```
To run tests:
```bash
pixi run test
```
## Project Status

Phase 0 is the initial reproducible project setup. Modeling and evaluation will
be added in later phases.

## 9. Install/check the Pixi environment
```bash
pixi install
```
Then check:
```bash
pixi run python --version
pixi run test
pixi run build
```
At this stage:

- pixi run test may say no tests collected. That is okay for Phase 0.
- pixi run build may say no tasks found. That is also okay for Phase 0.