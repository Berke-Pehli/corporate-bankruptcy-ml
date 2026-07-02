# Project Study Guide

## 1. Project objective and central argument

The project asks whether company-year financial variables can identify failed observations out of sample and whether tree-based machine-learning models improve on Logistic Regression.

The central argument is not that one model “solves” bankruptcy prediction. It is that rare-event screening requires leakage-aware splitting, metrics focused on the failed class, and an explicit choice between missed failures and false alarms. The majority baseline proves that accuracy alone is inadequate.

## 2. Data and unit of observation

The American Companies Bankruptcy Prediction dataset has:

- 78,682 rows and 21 raw columns;
- 8,971 distinct companies;
- company-year observations from 1999 through 2018;
- 18 financial predictors (`X1`–`X18`);
- no missing values or duplicate rows in the generated data summary;
- a raw status label of `alive` or `failed`.

The unit of observation is a **company-year**, not a company. A company can contribute several rows across years. The modeled target is `failed = 1` for failed and `failed = 0` for alive.

This distinction matters. The model classifies the status attached to a company-year record; the documentation should not silently reinterpret it as a rigorously defined multi-year probability of future bankruptcy unless the label timing supports that claim.

## 3. Features

The model uses all 18 documented financial predictors:

| Code | Financial variable | Code | Financial variable |
|---|---|---|---|
| X1 | Current assets | X10 | Total assets |
| X2 | Cost of goods sold | X11 | Total long-term debt |
| X3 | Depreciation and amortization | X12 | EBIT |
| X4 | EBITDA | X13 | Gross profit |
| X5 | Inventory | X14 | Total current liabilities |
| X6 | Net income | X15 | Retained earnings |
| X7 | Total receivables | X16 | Total revenue |
| X8 | Market value | X17 | Total liabilities |
| X9 | Net sales | X18 | Total operating expenses |

These are mostly financial levels rather than ratios. They can differ strongly by company size and can be correlated—for example, sales and revenue. This motivates standardization for Logistic Regression and the PCA extension. It also complicates coefficient interpretation: a coefficient is conditional on the other included variables.

## 4. Why bankruptcy prediction is difficult

1. **Rare outcome:** only 6.63% of full-sample observations are failed.
2. **Asymmetric costs:** missing a failure and investigating an alive firm usually have different financial consequences.
3. **Repeated firms:** a row-level random split could leak the same company's characteristics across train and test data.
4. **Correlated accounting variables:** financial statement items move together and can make linear coefficients unstable.
5. **Nonlinearity and interactions:** risk may change differently at different leverage or profitability levels.
6. **Time variation:** the observed annual failure rate falls from higher early-sample values to lower late-sample values; this can reflect economic conditions and sample composition.
7. **Definition and timing:** classification usefulness depends on exactly when the financial data and failure label are observed.

## 5. Class imbalance

There are 5,220 failed and 73,462 alive observations. A classifier that always predicts alive is correct 93.37% of the time in the full sample and 92.50% in the final test set, yet it detects no failures.

That is why accuracy is misleading: the abundant alive class dominates it. The project responds by:

- including the majority baseline;
- using class-weighted models or balanced sample weights;
- selecting tuned models using validation PR-AUC;
- reporting balanced accuracy and failed-class precision, recall, and F1;
- inspecting precision–recall curves and confusion matrices;
- analyzing alternative probability thresholds.

Class weighting does not create more failed observations. It changes the fitting loss so errors on minority observations matter more.

## 6. Leakage-aware splitting and validation

The first `GroupShuffleSplit` assigns whole companies to either training or final test data with a fixed random seed:

| Split | Rows | Companies | Failed observations | Failure rate |
|---|---:|---:|---:|---:|
| Train | 62,988 | 7,176 | 4,043 | 6.42% |
| Final test | 15,694 | 1,795 | 1,177 | 7.50% |

A second company-level split inside training creates model-training and validation subsets. Hyperparameters, PCA component comparisons, and thresholds are evaluated on internal validation data. The final test set is reserved for evaluation of the selected core models.

Why group by company? If one firm's 2010 record is in training and its 2011 record is in testing, persistent firm characteristics can make the test artificially easy. Company grouping asks the harder question: can the model generalize to unseen companies?

Why is this still limited? It is not a chronological split, so training and test periods overlap. It measures generalization across companies under this sample design, not a future-period backtest.

## 7. Models and why each is included

### Majority-class baseline

Always predicts alive. It sets a minimum reference and makes the accuracy paradox visible: 92.5% final-test accuracy, but zero failed recall and zero failed F1.

### Logistic Regression

The interpretable benchmark models a linear relationship between standardized predictors and failure log-odds. It outputs scores/probabilities and coefficient directions. The model uses balanced class weights and fixed L2 regularization with `C = 1.0`.

Interpretation: holding the other standardized variables fixed, a positive coefficient is associated with higher predicted failure log-odds. It is not a direct percentage-point effect and not causal.

### L1-regularized Logistic Regression

L1 regularization can push coefficients to zero, which is useful for sparse feature selection. Candidate `C` values are compared using validation PR-AUC; the selected value is 0.1.

### L2-regularized Logistic Regression

L2 regularization shrinks coefficients toward zero and can stabilize estimation when variables are correlated. It normally retains all predictors. The selected `C` is 0.1.

### Decision Tree

A tree creates nonlinear decision rules and interactions without scaling. It is more flexible than a linear model and can be visualized conceptually, but one tree can be unstable and overfit. Depth and leaf-size controls regularize it.

### Random Forest

Random Forest averages many trees trained with resampling and random feature subsets. Averaging reduces the high variance of individual trees. In this project it gives the best final-test ROC-AUC, PR-AUC, failed precision, and failed F1 among the core comparisons.

### Gradient Boosting

Gradient Boosting builds shallow trees sequentially, emphasizing errors left by previous trees. It can capture nonlinear structure efficiently. Here it gives the best final-test balanced accuracy and more failed recall than Random Forest, but lower precision.

### PCA + Logistic Regression

PCA replaces original standardized variables with orthogonal linear combinations. It tests whether correlated accounting information can be compressed before classification. The cost is that components are harder to explain financially than current assets, income, or debt.

## 8. Metrics in plain language

Let a “positive” mean a failed observation.

- **True positive (TP):** failed and correctly flagged.
- **False positive (FP):** alive but flagged as failed—a false alarm.
- **False negative (FN):** failed but predicted alive—a missed failure.
- **True negative (TN):** alive and correctly predicted alive.

### Accuracy

`(TP + TN) / all observations`. Easy to understand, but dominated by true negatives when alive firms are common.

### Balanced accuracy

Average of failed recall and alive recall. It gives both classes equal weight, regardless of prevalence.

### Precision for failed

`TP / (TP + FP)`. Of all flags, how many are actual failures? Low precision means a heavy review burden.

### Recall for failed

`TP / (TP + FN)`. Of all failures, how many did the model catch? Low recall means many costly misses.

### F1 for failed

`2 × precision × recall / (precision + recall)`. It rewards a balance of precision and recall but assumes they deserve symmetric treatment. Business costs may not be symmetric.

### ROC-AUC

Measures ranking across all classification thresholds. It can be read as the chance that a random failed observation is scored above a random alive observation. A value of 0.5 is random ranking. Under severe imbalance, its many true negatives can make performance look more comfortable than the operational alert problem feels.

### PR-AUC

Summarizes failed-class precision versus recall over thresholds. It focuses on the positive class and is particularly useful here. Its no-skill reference is approximately the positive prevalence—7.50% on the final test set—not 0.5.

## 9. Reading the important figures

### `class_balance.png`

The alive bar overwhelms the failed bar. Say: “This is why a high accuracy number can coexist with no useful failure detection.”

### `annual_failure_rate.png`

The observed rate changes substantially over 1999–2018 and generally declines later in the sample. Say: “The data are not temporally homogeneous; the plot is descriptive and may reflect composition as well as economic conditions.”

### `key_feature_median_by_status.png`

Compares robust class medians for selected financial variables. Failed observations have lower median market value, net income, current assets, and retained earnings, and higher median long-term debt and total liabilities in the reported table. Say: “These are unconditional associations, not causal effects.”

### `final_test_core_metric_summary.png`

Compares the core models on metrics relevant to failure detection. Use it to show that model rankings depend on the metric: Random Forest leads F1/PR-AUC, while Gradient Boosting leads balanced accuracy and has higher recall.

### `final_test_precision_recall_curves_key_models.png`

At each recall level, inspect attainable precision. Curves above the prevalence baseline show useful ranking. Emphasize that PR curves are more directly connected to rare failure alerts than accuracy.

### `final_test_confusion_matrices_key_models.png`

This makes operational costs concrete. Logistic Regression catches 1,041 of 1,177 failures but creates 9,906 false positives. Random Forest catches 442 with 2,190 false positives. Gradient Boosting catches 695 with 4,563 false positives. The baseline catches none.

### `logistic_coefficients.png`

Inputs are standardized, so coefficient magnitudes are more comparable. Current assets and market value have the largest negative coefficients; current liabilities has the largest positive coefficient. Discuss signs as conditional predictive associations. Correlation and financial scale structure mean signs need care.

### `tree_feature_importance.png`

Market value, net income, and long-term debt are prominent across tree models. Importance indicates use in prediction, not direction, causality, or stability across samples.

### `validation_threshold_tradeoff.png`

Moving the cutoff changes predicted labels without retraining. Higher thresholds generally reduce flags and recall while potentially improving precision. Selected validation F1 cutoffs are 0.53 (Logistic), 0.57 (Random Forest), and 0.68 (Gradient Boosting).

### `pca_explained_variance.png`

Shows how much standardized predictor variance is retained as components accumulate. Two components explain 83.8%; five explain 93.9%; ten explain 99.2%. Explain that high explained variance does not guarantee high bankruptcy discrimination.

### `pca_logistic_metric_comparison.png`

Performance generally improves when more components are retained. The best tested PCA validation PR-AUC is 0.151 with 12 components. Compression to very few components loses predictive information even though it retains much total variance.

## 10. Main empirical results

| Model | Accuracy | Balanced accuracy | ROC-AUC | PR-AUC | Failed precision | Failed recall | Failed F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.925 | 0.500 | 0.500 | 0.075 | 0.000 | 0.000 | 0.000 |
| Logistic | 0.360 | 0.601 | 0.690 | 0.153 | 0.095 | 0.884 | 0.172 |
| L1 Logistic | 0.358 | 0.602 | 0.692 | 0.154 | 0.095 | 0.889 | 0.172 |
| L2 Logistic | 0.358 | 0.602 | 0.692 | 0.153 | 0.095 | 0.888 | 0.172 |
| Decision Tree | 0.600 | 0.581 | 0.633 | 0.122 | 0.102 | 0.558 | 0.173 |
| Random Forest | 0.814 | 0.612 | 0.699 | 0.159 | 0.168 | 0.376 | 0.232 |
| Gradient Boosting | 0.679 | 0.638 | 0.688 | 0.158 | 0.132 | 0.590 | 0.216 |

Interpretation:

- The baseline is useless for failure detection despite excellent accuracy.
- All substantive models rank failures better than chance, but absolute PR-AUC and precision remain modest.
- Regularization changes Logistic Regression results only slightly.
- Random Forest gives the strongest precision/F1 trade-off at its default rule.
- Gradient Boosting catches more failures and has the highest balanced accuracy.
- Logistic Regression prioritizes recall very strongly because of balanced class weighting, leading to many false positives.
- These models are better viewed as screening/ranking tools than automatic decision makers.

## 11. Why threshold tuning matters

A classifier first produces a continuous failure score or probability. The cutoff turns that score into a binary decision. The conventional 0.50 cutoff is a convention, not a finance optimum.

Suppose a lender views missed failures as much more expensive than manual reviews. It may lower the threshold to raise recall. If review capacity is limited, it may raise the threshold to increase precision. The appropriate cutoff needs:

- costs of false negatives and false positives;
- review capacity;
- desired recall or precision constraints;
- probability calibration;
- validation on representative, preferably later-period data.

The project explores cutoffs on validation predictions. It does not tune on final-test outcomes, which would leak test information. Selected thresholds demonstrate trade-offs; they are not production recommendations.

## 12. What PCA adds

PCA addresses multicollinearity and compression by finding directions with maximum predictor variance. Components are mutually orthogonal and ranked by explained variance.

Benefits:

- compresses correlated inputs;
- can reduce redundant dimensions;
- provides a course-relevant comparison with original-feature Logistic Regression.

Costs:

- each component mixes many financial variables;
- coefficient interpretation moves from recognizable accounts to abstract components;
- directions that explain predictor variance are not chosen to separate failed from alive firms;
- performance numbers are from internal validation, not the final test set.

The study's result is nuanced: a few components retain much variance but not enough discriminative information; using more components recovers performance, reducing the practical value of compression.

## 13. Interpretability versus predictive performance

Logistic Regression is easier to communicate because it provides signed coefficients tied to original variables. Its linear structure is restrictive, and class weighting leads to many flags here.

Tree ensembles capture nonlinear relationships and interactions and obtain stronger final-test trade-offs on some metrics. Their global feature importances are less informative: they do not say whether a high value raises or lowers risk and are not causal.

PCA is an instructive third point: it keeps a linear classifier but makes the inputs abstract. Simpler mathematics does not always mean simpler business interpretation.

## 14. Limitations and critical reflection

- The split is company-level but not time-based; genuine forward forecasting may be harder.
- Repeated company-year observations may have within-company dependence.
- Failure-label timing is crucial for determining a real prediction horizon.
- Dataset coverage and the declining failure rate may create temporal or composition shift.
- Financial levels omit useful ratios and differ greatly with company size.
- Macroeconomic, industry, governance, market-return, and qualitative variables are absent.
- A single split and limited grids do not quantify sampling uncertainty.
- Low failed precision means many false alarms.
- Calibration is not established; scores should not automatically be treated as literal probabilities.
- Coefficients and feature importances are associations, not causal evidence.
- There is no economic cost function, fairness analysis, stress test, or live monitoring design.
- The work is educational and reproducible, not production-ready.

## 15. Possible professor questions and suggested answers

### Why not use accuracy as the main metric?

Because failed observations are rare. The majority baseline obtains 92.5% final-test accuracy while detecting zero of 1,177 failures. I therefore emphasize failed recall, precision, F1, PR-AUC, balanced accuracy, and the confusion matrix.

### Why split by company rather than row?

The same company appears in multiple years. A row split could place near-related firm records on both sides and overstate generalization. Grouping ensures the final test contains unseen companies.

### Is the split a true forecasting test?

No. It tests cross-company generalization, but train and test years overlap. A chronological or rolling-origin design would be a valuable extension.

### Why use PR-AUC for model selection?

PR-AUC focuses on precision and recall for the minority failed class. It more directly measures how cleanly the model ranks rare failure alerts than overall accuracy.

### Why report ROC-AUC as well?

ROC-AUC is a standard threshold-independent ranking measure and makes comparison familiar. I report both but give PR-AUC more weight under imbalance.

### Which model is best?

There is no metric-free winner. Random Forest has the best final-test PR-AUC and failed F1. Gradient Boosting has the best balanced accuracy and higher failed recall. Logistic catches the most failures but creates far more false alarms. The correct choice depends on error costs.

### Why is Logistic Regression accuracy so low?

Balanced class weighting makes minority failures more influential. At the default decision rule the model predicts many observations as failed, raising recall to 88.4% but creating 9,906 false positives. This is a deliberate trade-off, not necessarily evidence that the ranking scores are useless.

### Does a positive Logistic coefficient mean the variable causes bankruptcy?

No. It is a conditional association with predicted log-odds in this dataset. Correlation, omitted variables, and lack of causal design prevent causal interpretation.

### Can tree importance show the direction of risk?

No. It shows how much the model used a feature, not whether higher values raise risk. It can also be affected by correlation and the fitting sample.

### Why tune the threshold?

The default 0.50 cutoff ignores business costs and prevalence. Changing the cutoff controls the recall–precision and false-negative–false-positive trade-off without retraining.

### Why must threshold selection use validation data?

Choosing it on test outcomes would optimize to the supposedly unseen test sample and make final performance optimistic.

### What does PCA accomplish?

It turns correlated standardized financial variables into orthogonal components and can compress the feature space. Here, very few components retain much variance but lose classification performance, and components are harder to interpret.

### Why is explained variance not the same as predictive value?

PCA is unsupervised: it finds directions with high variation in `X` without using the failure label. Low-variance directions may still contain information that separates classes.

### Are the model outputs calibrated bankruptcy probabilities?

The project evaluates discrimination and classification, not formal calibration. I would call them model scores or predicted probabilities and validate calibration before operational use.

### What would you improve next?

I would define a clear prediction horizon, use chronological backtesting, add financial ratios and industry/macro controls, use repeated grouped validation, assess calibration and uncertainty, and choose a threshold with an explicit economic cost function.

## 16. A strong closing statement

“The project shows why rare-event finance classification must be evaluated as a decision problem. Company-level splitting reduces leakage, PR-AUC and failed-class metrics expose performance hidden by accuracy, and threshold analysis makes the operational trade-off explicit. The models contain useful signal, but modest precision, temporal limitations, and non-causal interpretation mean they should be treated as educational screening models rather than production systems.”
