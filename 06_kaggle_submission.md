# Phase 6 — Kaggle Submission

Competition: **Playground Series S5E1 — Forecasting Sticker Sales**
Slug: `playground-series-s5e1`
Target column: `num_sold`
Metric: **Mean Absolute Percentage Error (MAPE)**
Final deadline: **February 1, 2025 (11:59 PM UTC)** — late submission accepted by university

## 1. Load the trained model and generate predictions

```python
import joblib, numpy as np, pandas as pd

model = joblib.load("models/s5e1/3_rf_s5e1.pkl")
test_fe = pd.read_parquet("data/processed/test_s5e1_fe.parquet")
sample = pd.read_csv("data/raw/sample_submission_s5e1.csv")

feature_cols = [c for c in test_fe.columns if c not in ['id', 'date', 'country', 'store', 'product']]
X_test = test_fe[feature_cols]
preds = np.maximum(model.predict(X_test), 0)
```

## 2. Format the submission file

```python
submission = pd.DataFrame({"id": sample["id"], "num_sold": preds})
submission.to_csv("submissions/submission_s5e1.csv", index=False)
```

## 3. Submission Files Generated

| # | File | Model | Val MAPE |
|---|---|---|---|
| 1 | `submission_s5e1_rf.csv` | RandomForest | **7.43%** 🏆 |
| 2 | `submission_s5e1_xgb.csv` | XGBoost | 8.61% |
| 3 | `submission_s5e1_lgb.csv` | LightGBM | 8.75% |
| 4 | `submission_s5e1_hgb.csv` | HistGradientBoost | 10.97% |
| 5 | `submission_s5e1_nn.csv` | MLP Neural Net | 20.46% |
| 6 | `submission_s5e1_ensemble.csv` | Ensemble (avg of 5) | — |
| 7 | `submission_s5e1_best_rf.csv` | RandomForest (best) | **7.43%** |

## 4. Recommended submission for university

Use `submission_s5e1_best_rf.csv` — RandomForest with 7.43% validation MAPE.

## Deliverable checklist

- [x] All 5 models produce valid submission CSVs (id + num_sold columns)
- [x] Best model identified (RandomForest, 7.43% val MAPE)
- [x] Ensemble submission generated
- [x] Submission CSVs saved in `submissions/` (7 files)

Next: `07_app_development.md`