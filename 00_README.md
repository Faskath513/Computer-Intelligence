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
- [x] 4. Preprocessing pipeline built (V4 with feature engineering)
- [x] 5. Models trained & compared (V4: LightGBM, XGBoost, RandomForest)
- [x] 6. Kaggle submissions generated (10 submissions, best public: 0.89764)
- [x] 7. App built (`app/app.py` with V4 LightGBM + feature engineering)
- [x] 8. Integration tested

## Final Results

| Model | Val Balanced Accuracy | Kaggle Public |
|---|---|---|
| LightGBM V4 (conservative) | **0.9337** | pending |
| LightGBM V3 (original) | 0.8987 | **0.89764** |
| Random Forest V4 | 0.8977 | 0.89314 |
| XGBoost V4 | ~0.90 | 0.85488 |

> **Winner: LightGBM V4** — val 0.9337, `submission_v4_lgb_v3.csv`




C:\Users\aasan\anaconda3\python.exe -m streamlit run app/app.py
