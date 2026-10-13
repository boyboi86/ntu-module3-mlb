## Feature Engineering — Data Dictionary

### 1. Objective

Document the three engineered features created to support MLB season wins prediction (`W`) using Lasso Regression.

### 2. Original Features

| Feature | Description | Data Type | Unit |
|---|---|---|---|
| `R` | Total runs scored by a team during a season | Integer | Runs |
| `RA` | Total runs allowed by a team during a season | Integer | Runs |
| `G` | Total games played during a season | Integer | Games |
| `W` | Total games won during a season (target variable) | Integer | Wins |

### 3. Engineered Features

| Feature | Formula | Data Type | Description |
|---|---|---|---|
| `R_per_game` | `R / G` | Float | Average runs scored per game |
| `RA_per_game` | `RA / G` | Float | Average runs allowed per game |
| `Run_Differential` | `R - RA` | Integer | Difference between total runs scored and runs allowed |

### 4. Feature Importance and Rationale

| Feature | Rationale | Interpretation |
|---|---|---|
| `R_per_game` | Standardizes offensive performance across seasons of different lengths | Higher values indicate more runs scored per game |
| `RA_per_game` | Standardizes run prevention across seasons of different lengths | Lower values generally indicate better run prevention |
| `Run_Differential` | Combines offensive and run-prevention performance | Higher values indicate a stronger net scoring advantage |

### 5. Feature Engineering Results

The following results were obtained from the first five observations of `data_df`.

| Row | R | RA | G | R_per_game | RA_per_game | Run_Differential |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 718 | 732 | 154 | 4.6623 | 4.7532 | -14 |
| 1 | 835 | 751 | 162 | 5.1543 | 4.6358 | 84 |
| 2 | 768 | 707 | 162 | 4.7407 | 4.3642 | 61 |
| 3 | 713 | 598 | 154 | 4.6299 | 3.8831 | 115 |
| 4 | 928 | 921 | 162 | 5.7284 | 5.6852 | 7 |

### 6. Relevance to Lasso Regression

These engineered features provide additional representations of offensive performance, run prevention and overall scoring advantage.

Lasso Regression uses L1 regularization, which can shrink less useful feature coefficients to zero.

However, the engineered features are derived from existing columns and may introduce redundancy. Their predictive usefulness and importance have not yet been established.

### 7. Implementation Status

| Feature | Training Dataset (`data_df`) | Prediction Dataset (`predict_df`) |
|---|---|---|
| `R_per_game` | Completed | Completed |
| `RA_per_game` | Completed | Completed |
| `Run_Differential` | Completed | Completed |

### 8. Conclusion

Three features have been successfully engineered and their calculations verified.

No additional feature engineering is planned for the initial Lasso model.

The next stage is to evaluate whether these features improve prediction accuracy compared with the original Linear Regression baseline, using Mean Absolute Error (MAE).
