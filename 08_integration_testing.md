# Phase 8 — Integration Testing & Demo Recording

## 1. Clean-environment test

Simulate someone else (or future-you, months later) trying to run this from scratch.

```bash
git clone <your-repo> test-clone
cd test-clone
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```
- [x] It runs with no missing-file or missing-package errors

## 2. Functional test cases

Test the app with different filter combinations:

| Case | Input | Expected behavior |
|---|---|---|
| Default load | No filters | Shows all 90 time series summed |
| Single country | Select "Kenya" | Shows only Kenya sales |
| Single store | Select "Premium Sticker Mart" | Shows only that store |
| Single product | Select "Holographic Goose" | Shows only NaN in training (cold start) |
| All filters | Country=Norway, Store=Premium, Product=Kaggle | Highest volume series (~3198 avg) |
| Forecast view | Any selection with model loaded | Shows blue historical + orange forecast lines |

- [x] Log the actual output for each row in a small table

### Test Results

| Case | Visual Output | Predictions |
|---|---|---|
| Default (all) | 6 coloured country lines | Predictions for all 98,550 test rows |
| Kenya only | Single line, very low values | ~8-18 range (correct for Kenya) |
| Holographic Goose | NaN shown as gaps in training | Valid predictions in test period |
| Norway Premium Kaggle | Highest sales line (~3000) | Valid forecast, smooth continuation |

All cases run without errors. Predictions are plausible across all scenarios.

## 3. Consistency check between training and app pipeline

Run features_s5e1.py then compare model predictions with the app's prediction path.
Confirm they produce the **same output** for the same input.

- [x] Consistency check passed: both paths use same feature engineering + model artifacts

## 4. Model artifact consistency

| Artifact | Status | Notes |
|---|---|---|
| `models/s5e1/1_lgb_s5e1.pkl` | ✅ | LightGBM, val MAPE 8.75% |
| `models/s5e1/2_xgb_s5e1.pkl` | ✅ | XGBoost, val MAPE 8.61% |
| `models/s5e1/3_rf_s5e1.pkl` | ✅ | RandomForest, val MAPE 7.43% (best) |
| `models/s5e1/4_hgb_s5e1.pkl` | ✅ | HistGradientBoost, val MAPE 10.97% |
| `models/s5e1/5_nn_s5e1.pkl` | ✅ | MLP Neural Net, val MAPE 20.46% |
| `models/s5e1/5_scaler_s5e1.pkl` | ✅ | StandardScaler for MLP |
| `models/s5e1/model_comparison.csv` | ✅ | Full comparison table |
| `data/processed/train_s5e1_fe.parquet` | ✅ | 221,259 rows, 42 columns |
| `data/processed/test_s5e1_fe.parquet` | ✅ | 98,550 rows, 41 columns |

### Submission Files

| File | Model | Val MAPE |
|---|---|---|
| `submissions/submission_s5e1_rf.csv` | RandomForest | **7.43%** |
| `submissions/submission_s5e1_xgb.csv` | XGBoost | 8.61% |
| `submissions/submission_s5e1_lgb.csv` | LightGBM | 8.75% |
| `submissions/submission_s5e1_hgb.csv` | HistGradientBoost | 10.97% |
| `submissions/submission_s5e1_nn.csv` | MLP Neural Net | 20.46% |
| `submissions/submission_s5e1_ensemble.csv` | Ensemble | — |
| `submissions/submission_s5e1_best_rf.csv` | RandomForest (best) | **7.43%** |

## 5. App Verification

- [x] App loads LightGBM model correctly
- [x] App shows historical sales visualizations (line, box, seasonality)
- [x] Time series explorer works with all filter combinations
- [x] Forecast overlay shows predictions for selected series
- [x] Download button produces valid CSV
- [x] "About this model" expander shows correct model info

## 6. Demo recording

- [ ] Screen-record a 2-4 minute walkthrough: launch app → explore filters → show predictions → briefly show the EDA/feature engineering scripts to tie it together
- [ ] Keep it simple and narrated in plain language
- [ ] Save the recording somewhere retrievable

## 7. Final repo state

- [x] All model artifacts rebuilt for S5E1 Sticker Sales
- [x] `experiments.md` has final 5-model comparison
- [x] `submissions/` has 7 submission CSVs
- [x] App runs standalone with LightGBM model
- [x] Everything committed to git with a clear final commit message

## 8. Files Summary

### Source Code
- `src/features_s5e1.py` — Feature engineering pipeline
- `src/train_s5e1.py` — 2-model training (LGBM, XGBoost)
- `src/train_5_models_s5e1.py` — 5-model training + comparison
- `src/submit_s5e1_final.py` — Full-data training + ensemble
- `app/app.py` — Streamlit web app (S5E1)

### Model Files
- `models/s5e1/1_lgb_s5e1.pkl` — LightGBM model
- `models/s5e1/2_xgb_s5e1.pkl` — XGBoost model
- `models/s5e1/3_rf_s5e1.pkl` — RandomForest model (best)
- `models/s5e1/4_hgb_s5e1.pkl` — HistGradientBoost model
- `models/s5e1/5_nn_s5e1.pkl` — MLP Neural Net model

### Documentation
- `experiments.md` — Updated with S5E1 results
- `08_integration_testing.md` — This file

## Build phase — done

At this point the technical build is complete:
- 5 trained and compared models (RF, XGBoost, LightGBM, HGB, MLP)
- 7 submission CSVs ready for university submission
- Working web app with interactive forecasting dashboard
- Tested end-to-end

Everything from here (the report) is a separate phase, built from
what you've documented in `experiments.md`, your EDA findings, and this demo.