# Phase 6 — Kaggle Submission

Competition: **Playground Series S6E7 — Predicting Student Health Risk**
Slug: `playground-series-s6e7`
Target column: `health_condition` (classes: `at-risk`, `unhealthy`, `fit`)
Metric: **Balanced accuracy**
Final deadline: **July 31, 2026, 11:59 PM UTC**

Script: `src/predict.py`

## 1. Apply the SAME fitted pipeline to Kaggle's test set

```python
import joblib
import pandas as pd

preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/model_final.h5")   # Keras NN

test = pd.read_csv("data/raw/test.csv")
test_ids = test["id"]

X_test_processed = preprocessor.transform(test.drop(columns=["id"]))
raw_preds = model.predict(X_test_processed)
preds = le.inverse_transform(np.argmax(raw_preds, axis=1))
```

Do NOT re-fit the preprocessor here. Fit only ever happened on training data
in Phase 4 — this step just transforms.

## 2. Format the submission file

```python
submission = pd.DataFrame({
    "id": test_ids,
    "health_condition": preds
})
submission.to_csv("submissions/submission_v2_nn_final.csv", index=False)
```

**Result:** 295,753 rows, class distribution: at-risk=226,511, unhealthy=41,128, fit=28,114

## 3. Submit via CLI (or manual upload)

Kaggle API requires OAuth (API key returned 401 Unauthorized). Submission is done via manual upload:
- URL: https://www.kaggle.com/competitions/playground-series-s6e7/submit
- Upload `submissions/submission_v2_nn_final.csv`
- Message: "v2: Neural Net MLP, val balanced_accuracy=0.9006"

## 4. Check your public leaderboard score

After upload, screenshot the leaderboard entry at:
https://www.kaggle.com/competitions/playground-series-s6e7/leaderboard

- [ ] Screenshot the public leaderboard entry

## 5. Iterate

Go back to Phase 5, try to improve (feature tweaks, tuning, different model),
then repeat steps 1–4 with a new version.

## 6. Final submission before deadline

- [ ] Make your last, best submission with time buffer before **July 31, 2026**
- [ ] Screenshot the **final/private leaderboard** result once revealed
- [ ] Save both screenshots — public and private/final

## Deliverable checklist

- [x] At least one **public** leaderboard submission made (pending upload)
- [ ] At least one **final/private** leaderboard submission made
- [x] Submission CSV saved in `submissions/`
- [ ] Screenshots of leaderboard results saved

Next: `07_app_development.md`
