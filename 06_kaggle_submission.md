# Phase 6 — Kaggle Submission

Competition: **Playground Series S6E7 — Predicting Student Health Risk**
Slug: `playground-series-s6e7`
Target column: `health_condition` (classes: `at-risk`, `unhealthy`, `fit`)
Metric: **Balanced accuracy**
Final deadline: **July 31, 2026, 11:59 PM UTC**

## 1. Apply the SAME fitted pipeline to Kaggle's test set

```python
import joblib, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

preprocessor = joblib.load("models/preprocessor_v4.pkl")
model = lgb.Booster(model_file="models/model_lgb_v3.txt")
le = joblib.load("models/label_encoder.pkl")

test = pd.read_csv("data/raw/test.csv")
test_ids = test["id"]

X_test = add_features(test.drop(columns=["id"]))  # feature engineering
X_test_proc = preprocessor.transform(X_test)
preds_enc = np.argmax(model.predict(X_test_proc), axis=1)
preds = le.inverse_transform(preds_enc)
```

Do NOT re-fit the preprocessor. Fit only happened on training data in Phase 4.

## 2. Format the submission file

```python
submission = pd.DataFrame({"id": test_ids, "health_condition": preds})
submission.to_csv("submissions/submission_v4_lgb_v3.csv", index=False)
```

## 3. Submit via manual upload

- URL: https://www.kaggle.com/competitions/playground-series-s6e7/submit
- Upload the chosen submission CSV

## 4. Submission History

| # | File | Public Score | Model | Notes |
|---|---|---|---|---|
| 1 | submission_v1.csv | - | Logistic Regression | Baseline |
| 2 | submission_v2_rf_final.csv | 0.89314 | Random Forest | V2 |
| 3 | submission_v3_lgb_final.csv | **0.89764** | LightGBM | Best public so far |
| 4 | submission_v3_xgb_final.csv | 0.85488 | XGBoost | Overfit |
| 5 | submission_v4_lgb_v2.csv | - | LightGBM V2 | Val 0.9333 |
| 6 | submission_v4_lgb_v3.csv | - | LightGBM V3 | Val 0.9337, best candidate |
| 7 | submission_v4_rf.csv | - | RF V4 | Val 0.8977 |
| 8 | submission_v4_xgb.csv | - | XGBoost V4 | Val ~0.90 |

## 5. Final submission before deadline

- [x] Last best submission uploaded before July 31, 2026
- [ ] Screenshot the final/private leaderboard result once revealed
- [ ] Save both public and private/final screenshots

## Deliverable checklist

- [x] At least one public leaderboard submission made (0.89764)
- [ ] Final/private leaderboard submission made
- [x] Submission CSVs saved in `submissions/` (10 files)
- [ ] Screenshots of leaderboard results saved

Next: `07_app_development.md`
