# Phase 5 — Modeling

Scripts: `src/train_5_models_s5e1.py`, `src/submit_s5e1_final.py`

Task: **Time series regression** — target `num_sold`. Scored on **MAPE** (Mean Absolute Percentage Error).

## Validation Strategy

Time-based split: Train on 2010-2015 (189,492 rows), validate on 2016 (31,767 rows).

## 5 Models Trained

### 1. RandomForest Regressor
```python
RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_leaf=5)
```
**Val MAPE: 7.43%** 🏆 Best

### 2. XGBoost Regressor
```python
XGBRegressor(max_depth=8, lr=0.01, reg_alpha=1.0, reg_lambda=2.0)
```
**Val MAPE: 8.61%** 🥈

### 3. LightGBM Regressor
```python
LGBMRegressor(num_leaves=127, max_depth=12, lr=0.01, reg_alpha=1.0, reg_lambda=1.0)
```
**Val MAPE: 8.75%** 🥉

### 4. HistGradientBoosting Regressor
```python
HistGradientBoostingRegressor(max_iter=1000, lr=0.05, max_depth=8)
```
**Val MAPE: 10.97%**

### 5. MLP Neural Network
```python
MLPRegressor(hidden_layer_sizes=(256, 128, 64), lr_init=0.001, batch_size=2048)
```
**Val MAPE: 20.46%**

## Full Results

| Model | Train MAPE | Val MAPE | Time |
|---|---|---|---|
| **RandomForest** | 3.49% | **7.43%** | 77s |
| XGBoost | 5.21% | 8.61% | 17s |
| LightGBM | 5.39% | 8.75% | 13s |
| HistGradientBoost | 6.59% | 10.97% | 13s |
| MLP Neural Net | 12.43% | 20.46% | 125s |

## Key Learnings

1. **RandomForest generalizes best** — lowest validation MAPE for this dataset
2. **Tree-based models dominate** regression with mixed features — top 4 are tree ensembles
3. **MLP underperforms** — needs more data or deeper architecture for time series
4. **Yearly lags matter most** — lag_365, lag_730, lag_1095 are the strongest predictors
5. **MAPE penalizes low-volume errors** — Kenya series (avg 5-18 sales) require careful handling

## Submission Files Generated

| File | Model |
|---|---|
| `submissions/submission_s5e1_rf.csv` | RandomForest (best) |
| `submissions/submission_s5e1_xgb.csv` | XGBoost |
| `submissions/submission_s5e1_lgb.csv` | LightGBM |
| `submissions/submission_s5e1_hgb.csv` | HistGradientBoost |
| `submissions/submission_s5e1_nn.csv` | MLP Neural Net |
| `submissions/submission_s5e1_ensemble.csv` | Average of all 5 |
| `submissions/submission_s5e1_best_rf.csv` | Best model (RandomForest) |

## Deliverable checklist

- [x] 5 modeling techniques trained and compared
- [x] All models evaluated with MAPE on time-based validation set
- [x] `experiments.md` filled in with comparison table
- [x] Models saved to `models/s5e1/`
- [x] 7 submission CSVs generated

Next: `06_kaggle_submission.md`