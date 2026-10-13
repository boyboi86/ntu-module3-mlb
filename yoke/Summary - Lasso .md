# Moneyball Analytics — Lasso Regression Complete Summary

## 1. Research Objective

The objective is to predict the total number of MLB team-season wins (`W`) using historical baseball statistics.

Lasso Regression was applied to investigate three research questions:

1. Does alpha tuning improve Lasso Regression performance?
2. Does feature engineering improve prediction accuracy?
3. Which features contribute most strongly to model predictions?

Mean Absolute Error (MAE) was used as the primary evaluation metric. A lower MAE indicates better predictive accuracy.

---

## 2. Dataset and Preparation

| Description | Original Dataset | Feature-Engineered Dataset |
|---|---:|---:|
| Input features | 44 | 47 |
| Training observations | 1,129 | 1,129 |
| Testing observations | 283 | 283 |
| Total observations | 1,412 | 1,412 |

**Target variable:** `W` — total MLB season wins.

Three engineered features were introduced for the 47-feature experiment:

| Feature | Calculation |
|---|---|
| R_per_game | Runs Scored / Games |
| RA_per_game | Runs Allowed / Games |
| Run_Differential | Runs Scored − Runs Allowed |

Numerical variables were standardized, while binary era and decade indicators were retained without scaling.

The same training and testing observations were used in the 44-feature and 47-feature experiments.

---

## 3. Experimental Design

Two feature configurations were evaluated.

### A. Original 44-Feature Lasso

This analysis follows seven steps:

| Step | Experiment | Purpose |
|---|---|---|
| 1 | Baseline Lasso | Evaluate alpha = 1.0 |
| 2 | Alpha Tuning | Select the best alpha through cross-validation |
| 3 | Tuned Lasso | Evaluate alpha = 0.01 |
| 4 | Actual vs Predicted | Visualize prediction accuracy |
| 5 | Residual Analysis | Examine prediction bias |
| 6 | Feature Selection | Identify non-zero and zero coefficients |
| 7 | Permutation Importance | Evaluate model reliance on features |

### B. Original E1–E5 Experimental Work

The earlier experiments are retained without deleting or renaming existing code.

| Original Experiment | Description |
|---|---|
| E1 | Baseline Lasso using 47 features |
| E2 | Alpha tuning using 47 features |
| E3 | Tuned Lasso using 47 features |
| E4 | Permutation importance using 47 features |
| E5 | Tuned Lasso and permutation importance using 44 features |

The new 44-feature Steps 2–7 reuse relevant model objects and results created in E5.

---

## 4. Step 1 — Baseline Lasso (44 Features)

### Objective

Establish the prediction performance of Lasso Regression before hyperparameter tuning.

**Model configuration:**

- Original features: 44
- Alpha: 1.0
- Regularization: L1

### Results

| Metric | Training | Testing |
|---|---:|---:|
| MAE | 3.3834 | 3.3231 |
| RMSE | 4.2713 | 3.9909 |
| R² | 0.8991 | 0.9003 |

### Feature Selection

- Total features: 44
- Selected features: 5
- Zero coefficients: 39

### Interpretation

At alpha = 1.0, Lasso applied strong regularization and retained only five active features.

The testing MAE was 3.3231 wins.

The strong penalty may have eliminated useful predictors, suggesting that alpha tuning was necessary.

---

## 5. Step 2 — Alpha Tuning (44 Features)

### Objective

Identify the alpha value that minimizes cross-validation MAE.

### Methodology

Six alpha values were evaluated:

**1.0, 0.5, 0.1, 0.05, 0.01 and 0.001**

Five-fold cross-validation was performed on the training dataset.

For each alpha:

1. Divide the training observations into five folds.
2. Train Lasso on four folds.
3. Predict the remaining validation fold.
4. Calculate validation MAE.
5. Repeat until every fold has been used for validation.
6. Calculate the average MAE across all five folds.
7. Compare the average MAE across the six alpha values.
8. Select the alpha with the lowest average CV MAE.

Preprocessing was fitted within the training folds to prevent validation-data leakage.

### Results

| Measurement | Result |
|---|---:|
| Original features | 44 |
| Cross-validation folds | 5 |
| Alpha candidates | 6 |
| Best alpha | **0.01** |
| Best CV MAE | **2.8190** |

### Interpretation

Higher alpha values imposed stronger regularization, potentially excluding useful predictive information.

Reducing alpha improved cross-validation performance, with alpha = 0.01 achieving the lowest average MAE among the six tested values.

Reducing alpha further to 0.001 did not improve the average CV MAE.

**Conclusion:** Alpha = 0.01 was selected as the best tested regularization parameter for the 44-feature model.

The held-out testing dataset was not used to select alpha.

---

## 6. Step 3 — Tuned Lasso Regression (44 Features)

### Objective

Evaluate the performance of the 44-feature Lasso model using the selected alpha of 0.01.

### Results

| Metric | Training | Testing |
|---|---:|---:|
| MAE | 2.6956 | 2.6898 |
| RMSE | 3.4084 | 3.3248 |
| R² | 0.9357 | 0.9308 |

### Baseline vs Tuned Lasso

| Metric | Baseline | Tuned |
|---|---:|---:|
| Alpha | 1.0 | **0.01** |
| Test MAE | 3.3231 | **2.6898** |
| Test RMSE | 3.9909 | **3.3248** |
| Test R² | 0.9003 | **0.9308** |
| Selected features | 5 | 33 |
| Zero coefficients | 39 | 11 |

### Key Findings

**Finding 1 — Alpha tuning improved predictive accuracy.**

Testing MAE decreased by 0.6333 wins, representing an improvement of approximately 19.1%.

**Finding 2 — More useful features were retained.**

The tuned model retained 33 features compared with only five in the baseline model.

**Finding 3 — Training and testing performance were similar.**

Training MAE was 2.6956, while testing MAE was 2.6898.

This indicates a small observed generalization gap for the current training/testing split.

**Conclusion:** Reducing alpha from 1.0 to 0.01 improved the predictive performance of Lasso Regression on this dataset.

---

## 7. Step 4 — Actual vs Predicted Wins

### Objective

Visualize how closely model predictions match actual MLB season wins.

### Results

| Measurement | Result |
|---|---:|
| Testing observations | 283 |
| Testing MAE | 2.6898 |
| Testing R² | 0.9308 |

### Graph Interpretation

- Points on the diagonal represent perfect predictions.
- Points above the diagonal represent overpredicted wins.
- Points below the diagonal represent underpredicted wins.
- Greater distance from the diagonal indicates larger prediction errors.

The model achieved an average absolute error of approximately 2.69 wins.

The testing R² of 0.9308 indicates that approximately 93.08% of the variation in wins was explained within the testing dataset.

**Conclusion:** The tuned model demonstrated strong predictive performance according to its MAE and R² results, although individual predictions may still contain substantial errors.

---

## 8. Step 5 — Residual Analysis

### Objective

Investigate whether the tuned model systematically overpredicts or underpredicts team-season wins.

**Residual = Actual Wins − Predicted Wins**

### Results

| Measurement | Result |
|---|---:|
| Mean residual | -0.0692 |
| Underpredicted observations | 134 |
| Overpredicted observations | 149 |
| Total observations | 283 |

### Interpretation

The mean residual of -0.0692 indicates a very small average overprediction of approximately 0.07 wins.

The model overpredicted 149 observations and underpredicted 134 observations.

These represent approximately 52.7% and 47.3% of the testing observations, respectively.

The small mean residual suggests minimal overall signed prediction bias.

However, the mean residual should not be confused with MAE because positive and negative residuals can cancel one another.

**Conclusion:** The tuned Lasso model exhibited little overall signed bias on the testing dataset.

---

## 9. Step 6 — Lasso Feature Selection

### Objective

Identify the features retained by L1 regularization and those assigned zero coefficients.

### Results

| Description | Count |
|---|---:|
| Total features | 44 |
| Selected features | 33 |
| Zero coefficients | 11 |

### Largest Coefficients

| Feature | Coefficient |
|---|---:|
| R — Runs Scored | +8.8861 |
| RA — Runs Allowed | -6.4736 |
| AB — At Bats | -5.0588 |
| IPouts — Pitching Outs | +5.0059 |
| SV — Saves | +4.0285 |
| CG — Complete Games | +3.2655 |
| era_1 | -2.6729 |
| H — Hits | +2.5698 |

### Zero-Coefficient Features

The following 11 features received zero coefficients:

- ERA
- HRA
- FP
- era_2
- era_3
- era_5
- era_8
- decade_1960
- decade_1980
- decade_2000
- decade_2010

### Interpretation

Lasso assigned zero coefficients to 11 of the 44 original features, leaving 33 active predictors.

Runs Scored had a positive coefficient, while Runs Allowed had a negative coefficient.

These signs are consistent with the expected associations between scoring, run prevention and wins.

However, regression coefficients describe conditional associations, not causal effects. Coefficient magnitudes should also be interpreted cautiously when numerical and indicator variables use different scales.

**Conclusion:** L1 regularization performed automatic feature selection while retaining a predictive model with a testing MAE of 2.6898.

---

## 10. Step 7 — Permutation Feature Importance (44 Features)

### Objective

Evaluate the importance of the original 44 features based on their impact on prediction accuracy.

### Methodology

Each feature was randomly shuffled while other features remained unchanged.

The trained model generated new predictions, and the change in MAE was recorded.

The process was repeated ten times per feature.

A larger positive importance indicates that shuffling the feature causes a greater deterioration in predictive accuracy.

### Results — Top 7 Features

| Rank | Feature | Mean MAE Increase |
|---|---|---:|
| 1 | R — Runs Scored | **6.9199** |
| 2 | RA — Runs Allowed | **4.7082** |
| 3 | IPouts | 3.8395 |
| 4 | AB | 3.5519 |
| 5 | SV | 2.4201 |
| 6 | CG | 1.8815 |
| 7 | H | 1.2422 |

### Interpretation

Runs Scored and Runs Allowed recorded the highest permutation importance values.

Shuffling Runs Scored increased testing MAE by approximately 6.9199 wins.

Shuffling Runs Allowed increased testing MAE by approximately 4.7082 wins.

These results indicate strong model reliance on scoring and run-prevention information.

However, several baseball statistics are correlated. Independent shuffling can create unrealistic combinations of observations and distort importance estimates.

**Conclusion:** Runs Scored and Runs Allowed were the two most influential features in the 44-feature model under the permutation test.

---

## 11. E1–E5 — Original 47-Feature Experiments

The earlier experiments using 47 features were retained for comparison.

### Original E1 — Baseline Lasso

- Features: 47
- Alpha: 1.0
- Training MAE: 3.2763
- Testing MAE: 3.1609
- Selected features: 3

### Original E2 — Alpha Tuning

Five-fold cross-validation evaluated the same six candidate alpha values.

| Alpha | CV MAE |
|---|---:|
| 1.000 | 3.3000 |
| 0.500 | 3.1373 |
| 0.100 | 2.9599 |
| 0.050 | 2.8752 |
| **0.010** | **2.8195** |
| 0.001 | 2.8244 |

Best alpha: **0.01**

### Original E3 — Tuned Lasso

- Features: 47
- Alpha: 0.01
- Training MAE: 2.6970
- Testing MAE: 2.6883
- Testing RMSE: 3.3242
- Testing R²: 0.9308
- Selected features: 32
- Zero coefficients: 15

### Original E4 — Permutation Importance

The engineered Run Differential feature achieved the highest permutation importance of 6.3306.

Other influential features included IPouts, AB, SV and CG.

### Original E5 — Tuned Lasso Using 44 Features

E5 separately performed alpha tuning and model evaluation using the original 44 features.

It selected alpha = 0.01 and achieved a testing MAE of 2.6898.

The trained E5 model and tuning results were subsequently reused in the organized 44-feature Steps 2–7.

**E5 remains unchanged in the notebook to preserve reproducibility and existing code dependencies.**

---

## 12. Final Model Comparison

| Model | Features | Alpha | Testing MAE | Testing RMSE | Testing R² |
|---|---:|---:|---:|---:|---:|
| Instructor Linear Regression | 44 | N/A | 2.7025 | 3.3310 | 0.9306 |
| Baseline Lasso | 44 | 1.0 | 3.3231 | 3.9909 | 0.9003 |
| Tuned Lasso | 44 | 0.01 | **2.6898** | 3.3248 | 0.9308 |
| Baseline Lasso | 47 | 1.0 | 3.1609 | 3.8578 | 0.9069 |
| Tuned Lasso | 47 | 0.01 | **2.6883** | **3.3242** | 0.9308 |

### Comparison 1 — Impact of Alpha Tuning

For the original 44 features:

- Baseline MAE: 3.3231
- Tuned MAE: 2.6898
- Improvement: 0.6333 wins, approximately 19.1%

Alpha tuning produced a substantial improvement in predictive performance.

### Comparison 2 — Impact of Feature Engineering

| Metric | 44 Features | 47 Features |
|---|---:|---:|
| Best alpha | 0.01 | 0.01 |
| CV MAE | 2.8190 | 2.8195 |
| Testing MAE | 2.6898 | 2.6883 |
| Testing RMSE | 3.3248 | 3.3242 |
| Selected features | 33 | 32 |

Adding three engineered features reduced testing MAE by only 0.0015 wins.

This improvement is practically negligible.

The 44-feature model also achieved a marginally lower cross-validation MAE.

### Comparison 3 — Permutation Importance

| Feature | 44-Feature Model | 47-Feature Model |
|---|---:|---:|
| R | 6.9199 | 0.8639 |
| RA | 4.7082 | 0.0000 |
| Run_Differential | Not included | 6.3306 |
| IPouts | 3.8395 | 3.7905 |
| AB | 3.5519 | 3.5255 |
| SV | 2.4201 | 2.4091 |

Introducing Run Differential changed how Lasso distributed its reliance across related features.

Because Run Differential is calculated from Runs Scored and Runs Allowed, this change in importance does not necessarily indicate an improvement in predictive information.

---

## 13. Key Findings

**Finding 1 — Alpha tuning substantially improved Lasso Regression.**

Five-fold cross-validation selected alpha = 0.01 for both feature configurations.

The 44-feature model achieved approximately 19.1% lower testing MAE after tuning.

**Finding 2 — Lasso performed automatic feature selection.**

The tuned 44-feature model retained 33 features and assigned zero coefficients to 11 features.

**Finding 3 — Feature engineering produced negligible improvement.**

The 47-feature model achieved only 0.0015 wins lower testing MAE compared with the 44-feature model.

**Finding 4 — Runs Scored and Runs Allowed were influential predictors.**

The 44-feature model's permutation analysis identified R and RA as the two highest-ranked features.

**Finding 5 — Feature importance changed with correlated predictors.**

The 47-feature model relied strongly on Run Differential, illustrating how correlated variables can change coefficient selection and permutation importance.

**Finding 6 — The simpler model provided comparable accuracy.**

The 44-feature model achieved virtually the same predictive performance as the 47-feature model without additional feature preparation.

---

## 14. Limitations

Several limitations should be considered:

1. Model performance was evaluated using a single training/testing split. Additional repeated or time-aware validation would be useful for assessing robustness.

2. Alpha tuning evaluated six candidate values. Alpha = 0.01 was the best tested value, not necessarily the global optimum.

3. Correlated baseball statistics can influence Lasso coefficient selection and permutation importance rankings.

4. Very small differences in MAE do not demonstrate statistically meaningful superiority.

5. The dataset contains historical full-season statistics. Therefore, the analysis predicts historical team-season wins using season-level information rather than establishing a validated preseason forecasting model.

6. Repeated examination of the testing results can influence model-selection decisions. Final performance should ideally be confirmed on an untouched external test set or the Kaggle evaluation data.

---

## 15. Final Conclusion and Model Recommendation

The experiments demonstrate that **hyperparameter tuning was substantially more beneficial than adding the three engineered features**.

The baseline 44-feature Lasso model achieved a testing MAE of 3.3231. Five-fold cross-validation identified alpha = 0.01, and the tuned model reduced testing MAE to 2.6898.

The tuned model also retained 33 active features, demonstrating Lasso's ability to combine prediction with automatic feature selection.

Feature engineering introduced Run Differential, Runs per Game and Runs Allowed per Game. Although Run Differential became influential within the 47-feature model, the additional features did not meaningfully improve predictive accuracy.

**Recommended model: Tuned Lasso Regression using 44 original features, alpha = 0.01.**

This model is preferred because it:

- Achieves a testing MAE of 2.6898 wins.
- Uses fewer input features than the engineered model.
- Achieves slightly better cross-validation MAE.
- Retains automatic feature selection through L1 regularization.
- Provides comparable predictive accuracy without additional feature engineering.

The E3 and E4 experiments remain valuable as evidence that adding features can change model interpretation without necessarily improving prediction accuracy.

**Overall conclusion:** Careful hyperparameter tuning, systematic model evaluation and appropriate feature selection are more important than simply increasing the number of input variables.
