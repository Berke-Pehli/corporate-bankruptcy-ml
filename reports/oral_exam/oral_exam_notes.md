# Oral Exam Notes

## 1. One-minute project explanation

“My project predicts failed versus alive company-year observations using 18 financial statement variables from 8,971 American companies between 1999 and 2018. Bankruptcy is a rare event: only 6.63% of the 78,682 observations are failed. That makes accuracy misleading—the majority baseline reaches 92.5% test accuracy but catches no failures.

I split by company, not row, so the same firm cannot appear in training and testing. I compare Logistic Regression and its L1/L2 variants with a Decision Tree, Random Forest, and Gradient Boosting. I select settings on internal validation data and evaluate the chosen core models on an untouched test set using PR-AUC, ROC-AUC, balanced accuracy, and precision, recall, and F1 for failures.

Random Forest has the best test PR-AUC and failed-class F1, while Gradient Boosting has better recall and balanced accuracy. Logistic Regression is easier to interpret and catches most failures, but creates many false alarms. The main conclusion is that model and threshold choice must reflect the financial cost of missed failures versus unnecessary reviews. These are predictive associations, not causal or production-ready results.”

## 2. Three-minute project explanation

### Problem and data

- Unit: one company in one year.
- Sample: 78,682 rows, 8,971 companies, 1999–2018.
- Target: failed = 1; alive = 0.
- Predictors: 18 accounting and market variables, including current assets, net income, market value, long-term debt, and liabilities.
- Failed observations: 5,220, or 6.63%.

“Bankruptcy prediction is difficult because failures are rare, financial variables are correlated, the same firm appears repeatedly, and false positives and false negatives have different costs.”

### Design

- Hold out 20% of companies for final testing: 1,795 companies and 15,694 observations.
- Use another company-level split inside training for model and threshold choices.
- Standardize predictors for Logistic Regression and PCA.
- Use class weighting or balanced sample weights to make failures matter more during training.
- Select tuned models using validation PR-AUC.

“Company grouping prevents leakage from persistent firm characteristics. It is still not a chronological backtest, which is an important limitation.”

### Models

- Majority baseline: exposes the accuracy trap.
- Logistic: interpretable linear benchmark.
- L1/L2 Logistic: regularization and correlated predictors.
- Decision Tree: readable nonlinear rules.
- Random Forest: lower-variance ensemble.
- Gradient Boosting: sequential nonlinear correction.
- PCA + Logistic: dimensionality-reduction extension.

### Evaluation and conclusion

“On the final test set, the baseline has 92.5% accuracy but zero recall. Random Forest has PR-AUC 0.159 and failed F1 0.232, both best among the compared models. Gradient Boosting has failed recall 0.590 and balanced accuracy 0.638, both higher than Random Forest. Logistic Regression has recall 0.884 but precision only 0.095, illustrating the false-alarm cost of aggressive screening.

Threshold analysis uses validation data only. Lower thresholds usually increase recall and false positives; higher thresholds tend to improve precision but miss failures. PCA shows that preserving predictor variance is not the same as preserving failure information, and it reduces financial interpretability.”

## 3. Main findings

- Accuracy is the wrong headline: the baseline detects zero failures.
- All substantive models rank failures above chance, but precision remains modest.
- Random Forest: best final-test ROC-AUC (0.699), PR-AUC (0.159), failed precision (0.168), and failed F1 (0.232).
- Gradient Boosting: best balanced accuracy (0.638) and a stronger precision–recall balance when more recall is desired; failed recall is 0.590.
- Logistic Regression: highest core-model recall (0.884), but 9,906 false positives and precision of 0.095.
- L1 and L2 regularization only slightly change Logistic results.
- No model is unconditionally best; the economic objective decides.
- Interpretations are predictive associations, not causes of bankruptcy.

## 4. Important figures to mention

### Class balance

Point: only 6.63% failed. Connect directly to the misleading baseline accuracy.

### Annual failure rate

Point: failure prevalence changes over time. Mention possible temporal/composition shift and avoid a causal macroeconomic story.

### Key-feature medians

Point: failed observations show weaker median profitability/capital measures and different debt/liability profiles. These are descriptive class comparisons.

### Core metric summary

Point: rankings change by metric. Random Forest leads F1/PR-AUC; Gradient Boosting leads balanced accuracy and recall relative to Random Forest.

### Precision–recall curves

Point: appropriate focus for the rare failed class. The no-skill level is near test prevalence, 7.5%.

### Confusion matrices

- Baseline: 0 of 1,177 failures caught.
- Logistic: 1,041 caught; 9,906 false alarms.
- Random Forest: 442 caught; 2,190 false alarms.
- Gradient Boosting: 695 caught; 4,563 false alarms.

### Logistic coefficients

Point: current assets and market value have the strongest negative coefficients; current liabilities has the strongest positive one. Inputs are standardized. Coefficients describe conditional log-odds associations.

### Tree importance

Point: market value, net income, and long-term debt repeatedly rank highly. Importance has no direction and is not causal.

### Threshold trade-off

Point: the binary decision is adjustable. Validation F1 cutoffs are 0.53 Logistic, 0.57 Random Forest, and 0.68 Gradient Boosting.

### PCA figures

Point: two components explain 83.8% of variance, but their PR-AUC is only 0.063. The best tested PCA PR-AUC is 0.151 at 12 components. Unsupervised variance is not identical to discriminative information.

## 5. Possible questions and answers

**Why not accuracy?**  Because 92.5% test accuracy is available by predicting every firm alive, with zero failure detection.

**Why company-level splitting?**  Repeated yearly records from one company are related. Grouping prevents the same firm from teaching and testing the model.

**Is it a future forecasting exercise?**  Not strictly. Companies are separated, but years overlap. A chronological backtest is a natural extension.

**Why PR-AUC?**  It summarizes the precision–recall trade-off for rare failures and is less distracted by abundant true negatives.

**ROC-AUC versus PR-AUC?**  ROC-AUC measures general ranking using true- and false-positive rates. PR-AUC emphasizes alert quality and failure coverage, so I prioritize it under imbalance.

**Which model wins?**  Random Forest for test PR-AUC/F1; Gradient Boosting for balanced accuracy and more recall; Logistic for maximum recall at a very high false-positive cost. The use case selects the winner.

**Why is Logistic accuracy low?**  Balanced class weighting plus the default rule leads to aggressive failure flags. It catches 88.4% of failures but misclassifies many alive observations.

**What does regularization do?**  L1 can set coefficients to zero; L2 shrinks them smoothly. Both reduce overfitting and help with correlated features. Results here are very close to the benchmark.

**Can coefficients be interpreted causally?**  No. They are conditional predictive associations in observational data, expressed in log-odds for standardized inputs.

**Why tune thresholds?**  A 0.50 cutoff does not encode credit losses, review capacity, or risk appetite. Thresholds translate scores into different operational trade-offs.

**Why tune only on validation?**  Using test outcomes would leak information and make the final result optimistic.

**Why PCA?**  It compresses correlated variables into orthogonal components and connects to course material. It also demonstrates the interpretability cost of dimensionality reduction.

**Why can two components explain 84% of variance but classify poorly?**  PCA ignores the target. High-variance directions are not necessarily the directions that separate failures.

**Are predicted scores calibrated probabilities?**  Calibration was not established. I evaluate discrimination and classification, so production probability claims would be premature.

**What next?**  Define a fixed prediction horizon, use rolling time validation, add ratios and macro/industry variables, repeat grouped validation, test calibration, and optimize an explicit financial cost function.

## 6. Critical reflection

Strengths:

- reproducible Pixi/pytask/pytest workflow;
- company-level leakage control;
- untouched final test for core models;
- honest majority baseline;
- minority-focused metrics and threshold analysis;
- comparison of interpretability and flexibility.

Limitations:

- no chronological holdout or rolling backtest;
- repeated company-year dependence;
- uncertain operational prediction horizon from the label definition;
- limited feature set and hyperparameter grids;
- no confidence intervals, calibration study, or economic loss function;
- low failed precision and substantial false-positive burden;
- feature effects are not causal;
- PCA results are validation results, not final-test results;
- not production-ready.

## 7. Final conclusion

“The project finds useful but limited bankruptcy signal in financial statements. Its most important contribution is methodological: avoid firm leakage, judge the failed class directly, and treat the threshold as a financial decision. Random Forest and Gradient Boosting improve particular trade-offs over Logistic Regression, while Logistic remains the clearest benchmark. The evidence supports cautious screening use in an educational setting—not causal claims or automated real-world decisions.”
