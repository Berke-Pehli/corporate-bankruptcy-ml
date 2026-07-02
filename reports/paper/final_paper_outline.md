# Final Paper Outline

## Working title

**Corporate Bankruptcy Risk Prediction with Machine Learning: Imbalanced Classification, Company-Level Validation, and the Interpretability–Performance Trade-off**

## 1. Introduction

### Purpose

- Motivate bankruptcy screening for lenders, investors, analysts, and risk managers.
- Explain why failure prediction is challenging: rarity, asymmetric error costs, repeated firm observations, correlated accounting variables, and time variation.
- Frame the work as predictive and educational, not causal or production-ready.

### Proposed opening argument

High accuracy is not enough in bankruptcy prediction. When failures are rare, a classifier can appear accurate by labeling every observation alive. A credible evaluation must instead ask how many failures are detected, how many false alarms are generated, and whether evaluation prevents company leakage.

### Contribution

- Compare interpretable Logistic Regression with regularized and tree-based models.
- Use company-level train/validation/test splitting.
- Evaluate the failed class using PR-AUC, precision, recall, F1, balanced accuracy, ROC-AUC, and confusion matrices.
- examine validation-only threshold tuning;
- add PCA as a dimensionality-reduction extension.

Suggested figures: `outputs/figures/class_balance.png` and `outputs/figures/annual_failure_rate.png`.

## 2. Research question

### Main question

How well can company-year financial variables identify failed observations out of sample, and do flexible tree-based methods improve on an interpretable Logistic Regression benchmark?

### Subquestions

1. Why does the majority-class baseline reveal the weakness of accuracy?
2. Which models provide the strongest failed-class ranking and classification trade-offs?
3. How does the probability threshold alter precision and recall?
4. Which financial variables are most associated with the models' predictions?
5. Does PCA preserve useful predictive information, and what interpretability is lost?

### Scope statement

The study estimates predictive associations. It does not identify causal drivers of bankruptcy or validate a production credit-decision system.

## 3. Related course concepts

Organize this section as a compact conceptual framework:

- binary classification and predicted probabilities;
- Logistic Regression and log-odds;
- L1 and L2 regularization;
- bias–variance trade-off in Decision Trees and Random Forests;
- sequential error correction in Gradient Boosting;
- training, validation, and untouched test evaluation;
- confusion matrices and class-specific metrics;
- ROC and precision–recall curves;
- class imbalance and class weighting;
- threshold tuning as a decision problem;
- PCA and dimensionality reduction;
- interpretability versus predictive flexibility.

Key point: PR-AUC is central because the failed class is rare, while ROC-AUC remains a useful standard ranking measure.

## 4. Data

### Source and sample

- American Companies Bankruptcy Prediction dataset.
- 78,682 company-year observations.
- 8,971 companies.
- Coverage from 1999 to 2018.
- 21 raw columns and 18 retained financial predictor columns.
- Generated summary reports no missing values or duplicate rows.

Reference: `outputs/tables/data_summary.csv`.

### Unit of observation and target

- One row is a company-year.
- Raw target labels are alive and failed.
- Encode failed as 1 and alive as 0.
- Full sample: 5,220 failed (6.63%) and 73,462 alive (93.37%).

Reference: `outputs/tables/target_distribution.csv`.

### Predictors

Describe the 18 variables by financial theme rather than repeating a long list in prose:

- liquidity/assets: current assets, inventory, receivables, total assets;
- profitability: EBITDA, EBIT, net income, gross profit, retained earnings;
- debt/liabilities: long-term debt, current liabilities, total liabilities;
- activity/scale: cost of goods sold, net sales, revenue, operating expenses;
- market information: market value;
- non-cash charges: depreciation and amortization.

Include the complete variable table from `outputs/tables/feature_dictionary.csv` in an appendix if space permits.

### Descriptive evidence

- Discuss the severe class imbalance.
- Describe annual failure-rate variation without assigning causes.
- Compare selected medians across status groups as unconditional associations.

Suggested figures:

- `outputs/figures/class_balance.png`
- `outputs/figures/annual_failure_rate.png`
- `outputs/figures/key_feature_median_by_status.png`

## 5. Methodology

### Preprocessing

- Validate required columns and target labels.
- Encode the binary target.
- retain 18 non-constant financial features;
- preserve company and year identifiers for split diagnostics, but exclude them from predictors;
- standardize within Logistic/PCA pipelines to avoid scale leakage.

### Data splitting

- Use `GroupShuffleSplit` with company as the group and random seed 42.
- Allocate about 80% of companies to training and 20% to final testing.
- Training: 62,988 rows, 7,176 firms, 6.42% failed.
- Test: 15,694 rows, 1,795 firms, 7.50% failed.
- Create a second company-level validation split inside training.
- Keep the final test set out of hyperparameter and threshold selection.

Reference: `outputs/tables/split_summary.csv`.

### Model specifications

1. Majority-class `DummyClassifier`.
2. Fixed Logistic Regression benchmark: standardized inputs, balanced class weights, L2, `C = 1.0`.
3. L1 Logistic Regression: selected `C = 0.1` using validation PR-AUC.
4. L2 Logistic Regression: selected `C = 0.1` using validation PR-AUC.
5. Decision Tree: selected depth and leaf size.
6. Random Forest: selected tree count, depth, leaf size, and feature subsampling.
7. Gradient Boosting: selected estimators, learning rate, depth, and leaf size, fitted with balanced sample weights.

Report exact selected tree settings from `outputs/tables/model_specification.csv` rather than reconstructing them manually.

### Interpretability methods

- Standardized Logistic coefficients: direction and magnitude in log-odds units.
- Impurity-based tree feature importances: relative model usage without direction.
- Explicit warning against causal interpretation.

References:

- `outputs/tables/logistic_coefficients.csv`
- `outputs/tables/tree_feature_importance.csv`
- `outputs/figures/logistic_coefficients.png`
- `outputs/figures/tree_feature_importance.png`

## 6. Evaluation strategy

### Why accuracy is insufficient

Use the majority baseline as the decisive example: 0.925 final-test accuracy but zero detected failures.

### Metrics

- Accuracy: overall correct share.
- Balanced accuracy: equal-weight average of class recalls.
- ROC-AUC: threshold-free ranking across both classes.
- PR-AUC: precision–recall summary focused on rare failures.
- Failed precision: alert quality.
- Failed recall: failure coverage.
- Failed F1: harmonic balance of precision and recall.
- Confusion matrix: operational counts of correct classifications, false alarms, and missed failures.

### Selection discipline

- Use validation PR-AUC for tuned model settings.
- Analyze thresholds on validation predictions only.
- Use final test data once for the core model comparison.
- Clarify that PCA tables report internal validation, not final-test performance.

## 7. Results

### Descriptive results

- Report the 6.63% full-sample failure share.
- Discuss changing annual rates cautiously.
- Describe class-median patterns from `outputs/tables/class_feature_summary.csv`.

### Final test model comparison

Build the main results table from `outputs/tables/final_test_metrics.csv`.

Core values to discuss:

| Model | Accuracy | Balanced accuracy | ROC-AUC | PR-AUC | Failed precision | Failed recall | Failed F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.925 | 0.500 | 0.500 | 0.075 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.360 | 0.601 | 0.690 | 0.153 | 0.095 | 0.884 | 0.172 |
| Random Forest | 0.814 | 0.612 | 0.699 | 0.159 | 0.168 | 0.376 | 0.232 |
| Gradient Boosting | 0.679 | 0.638 | 0.688 | 0.158 | 0.132 | 0.590 | 0.216 |

### Interpretation of comparisons

- The baseline demonstrates the accuracy paradox.
- Random Forest has the strongest PR-AUC and failed F1 at the default rule.
- Gradient Boosting has the strongest balanced accuracy and higher recall than Random Forest.
- Logistic Regression catches most failures but imposes the largest false-positive burden.
- Regularized Logistic variants are close to the benchmark.
- Avoid declaring a universal best model; ranking changes with the objective.

### Confusion-matrix discussion

Use exact final-test counts from `outputs/tables/final_test_metrics.csv`:

- Baseline: TP 0, FN 1,177, FP 0, TN 14,517.
- Logistic: TP 1,041, FN 136, FP 9,906, TN 4,611.
- Random Forest: TP 442, FN 735, FP 2,190, TN 12,327.
- Gradient Boosting: TP 695, FN 482, FP 4,563, TN 9,954.

Suggested figures:

- `outputs/figures/final_test_core_metric_summary.png`
- `outputs/figures/final_test_precision_recall_curves_key_models.png`
- `outputs/figures/final_test_confusion_matrices_key_models.png`

## 8. Interpretation

### Logistic Regression

- Largest negative coefficients: current assets (`X1`) and market value (`X8`).
- Largest positive coefficient: total current liabilities (`X14`).
- EBIT, retained earnings, EBITDA, and net income also have negative coefficients.
- Explain that standardized coefficients concern conditional changes in log-odds.
- Note that correlated level variables can generate unintuitive conditional signs.

### Tree models

- Market value, net income, and long-term debt are prominent across ensembles.
- Decision Tree gives especially high importance to net income, market value, and long-term debt.
- Importance is neither an effect direction nor causal evidence.
- Correlated variables can split or redistribute importance.

### Finance interpretation

The patterns are consistent with the broad idea that weaker liquidity/profitability and debt pressure can help identify distress, but the paper should not turn that consistency into a causal mechanism. Model outputs are risk-screening signals conditional on this dataset.

## 9. PCA extension

### Motivation and method

- Standardize all predictors.
- Apply PCA to produce orthogonal components.
- Fit balanced Logistic Regression using candidate component counts 2, 3, 5, 8, 10, 12, 15, and 18.
- Compare on the same internal company-level validation design.

### Results to report

- 2 components: 83.8% explained variance, PR-AUC 0.063.
- 5 components: 93.9% explained variance, PR-AUC 0.124.
- 10 components: 99.2% explained variance, PR-AUC 0.147.
- 12 components: highest tested PR-AUC, 0.151, with 99.8% explained variance.

Reference: `outputs/tables/pca_logistic_results.csv`.

### Interpretation

- Few components preserve much total predictor variance but lose label-relevant information.
- Performance recovers when most dimensions are retained.
- The small validation difference versus original-feature Logistic Regression does not clearly compensate for reduced interpretability.
- Do not compare PCA validation numbers as though they were PCA final-test results.

Suggested figures:

- `outputs/figures/pca_explained_variance.png`
- `outputs/figures/pca_logistic_metric_comparison.png`

## 10. Limitations

Organize limitations into four groups.

### Data and target

- Observational dataset; no causal identification.
- Company-year dependence and unclear operational prediction horizon.
- Possible reporting, coverage, and sample-composition biases.
- Failure prevalence changes over time.

### Validation

- Company grouping prevents firm overlap but is not chronological forecasting.
- Single split and no confidence intervals or repeated grouped validation.
- Limited hyperparameter grids.

### Features and models

- Financial levels are scale- and size-dependent.
- No explicit industry, macroeconomic, governance, market-return, or qualitative predictors.
- No formal probability-calibration assessment.
- Coefficients and feature importances may be unstable under correlation.

### Decision relevance

- No explicit monetary loss function or review-capacity constraint.
- Low precision means many false positives.
- Selected validation thresholds are examples, not deployment settings.
- No monitoring, fairness, stress-testing, or model-risk framework.
- Not production-ready.

## 11. Conclusion

### Findings to restate

- Financial statements contain out-of-sample signal beyond the majority baseline.
- Accuracy alone hides complete failure-detection weakness.
- Random Forest produces the best final-test PR-AUC and failed F1.
- Gradient Boosting provides higher recall and balanced accuracy than Random Forest.
- Logistic Regression remains valuable for interpretability and high recall, despite many false positives.
- Threshold choice is part of the finance decision, not merely a technical default.
- PCA illustrates both compression and interpretability costs without a decisive advantage.

### Final cautious claim

The evidence supports machine learning as a structured screening and ranking exercise within this dataset. It does not establish causal effects or demonstrate readiness for automated credit, investment, or regulatory decisions.

### Future work

- define a forward failure horizon;
- conduct rolling or chronological backtests;
- engineer ratios and trends;
- add industry and macroeconomic variables;
- use repeated grouped validation and uncertainty intervals;
- assess probability calibration;
- select thresholds using explicit financial costs;
- test robustness under temporal distribution shift.

## Suggested appendices

- Full feature dictionary: `outputs/tables/feature_dictionary.csv`.
- Model settings: `outputs/tables/model_specification.csv`.
- Full classification reports: `outputs/tables/final_test_classification_reports.csv`.
- Threshold candidates: `outputs/tables/selected_thresholds.csv`.
- PCA component loadings: `outputs/tables/pca_component_loadings.csv`.
- Reproducibility commands: `pixi run build` and `pixi run test`.
