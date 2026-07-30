# CIS 6005 Mini Project — Build File Set

Work through these in order. Each file is self-contained — open only the one
you're currently on. Report writing is NOT covered in this set; this is
build-only.

| # | File | Purpose |
|---|---|---|
| 1 | `01_competition_selection.md` | Lock in the Kaggle competition |
| 2 | `02_environment_setup.md` | Repo structure, venv, dependencies |
| 3 | `03_eda.md` | Explore and understand the data |
| 4 | `04_feature_engineering.md` | Clean, encode, build the preprocessing pipeline |
| 5 | `05_modeling.md` | Train and compare models |
| 6 | `06_kaggle_submission.md` | Submit to the leaderboard |
| 7 | `07_app_development.md` | Build the Streamlit web app |
| 8 | `08_integration_testing.md` | End-to-end test + demo recording |

## Progress tracker

- [x] 1. Competition selected
- [x] 2. Environment set up
- [x] 3. EDA complete
- [x] 4. Preprocessing pipeline built (feature engineering for 90 time series)
- [x] 5. Models trained & compared (5 models: RF, XGBoost, LightGBM, HGB, MLP)
- [x] 6. Submission CSVs generated (7 files: 5 models + ensemble + best)
- [x] 7. App built (`app/app.py` with interactive forecasting dashboard)
- [x] 8. Integration tested

## Final Results

| Model | Val MAPE | Rank |
|---|---|---|
| **RandomForest** | **7.43%** | 🥇 |
| XGBoost | 8.61% | 🥈 |
| LightGBM | 8.75% | 🥉 |
| HistGradientBoost | 10.97% | 4 |
| MLP Neural Net | 20.46% | 5 |

> **Winner: RandomForest** — val MAPE 7.43%, `submission_s5e1_best_rf.csv`
