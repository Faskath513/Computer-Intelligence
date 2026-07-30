# Model Comparison -- Playground Series S5E1

**Competition:** Forecasting Sticker Sales
**Metric:** Mean Absolute Percentage Error (MAPE)
**Data:** 90 time series (6 countries × 3 stores × 5 products), 2010-2016 train, 2017-2019 test

## Feature Engineering (41 features)

- **Date features:** year, month, dayofyear, quarter, dayofweek, weekend, cyclic encodings
- **Trend:** days_since_start, elapsed_years
- **Yearly lags:** lag_365, lag_730, lag_1095
- **Series statistics:** overall mean/std/median/min/max per (country, store, product)
- **Monthly stats:** monthly mean/std/median
- **Recent stats:** last year mean/std/median, last 90 days mean/std/median/min/max
- **Year-over-year:** yearly_mean, yearly_change
- **Categorical:** country_encoded, store_encoded, product_encoded

## Model Results (Validation: 2016 holdout, Train: 2010-2015)

| # | Model | Train MAPE | Val MAPE | Time (s) |
|---|---|---|---|---|
| 1 | **RandomForest** | 3.49% | **7.43%** 🏆 | 77 |
| 2 | XGBoost | 5.21% | 8.61% | 17 |
| 3 | LightGBM | 5.39% | 8.75% | 13 |
| 4 | HistGradientBoost | 6.59% | 10.97% | 13 |
| 5 | MLP Neural Net | 12.43% | 20.46% | 125 |

> **Winner: RandomForest** — val MAPE 7.43%

## Submissions Generated

| File | Model |
|---|---|
| `submission_s5e1_rf.csv` | RandomForest (best) |
| `submission_s5e1_xgb.csv` | XGBoost |
| `submission_s5e1_lgb.csv` | LightGBM |
| `submission_s5e1_hgb.csv` | HistGradientBoost |
| `submission_s5e1_nn.csv` | MLP Neural Net |
| `submission_s5e1_ensemble.csv` | Average of all 5 |
| `submission_s5e1_best_rf.csv` | Best model (RandomForest) |

## Key Learnings

1. **RandomForest generalizes best** for this dataset — lowest validation MAPE
2. **Tree-based models dominate** regression tasks with mixed categorical/numeric features
3. **MLP underperforms** on smaller datasets — needs more data or better tuning
4. **Yearly lags and target encodings** are the most powerful features for time series forecasting
5. **Time-based validation is essential** — random CV leaks future info in time series
