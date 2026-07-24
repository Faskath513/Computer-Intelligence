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
model = joblib.load("models/model_final.pkl")   # or tf.keras.models.load_model(...) for NN

test = pd.read_csv("data/raw/test.csv")
test_ids = test["id"]

X_test_processed = preprocessor.transform(test.drop(columns=["id"]))
preds = model.predict(X_test_processed)   # returns 'at-risk' / 'unhealthy' / 'fit'
```

Do NOT re-fit the preprocessor here. Fit only ever happened on training data
in Phase 4 — this step just transforms.

If your model outputs encoded integers instead of the string labels (common
with sklearn's `LabelEncoder` or a Keras softmax output), decode them back to
`at-risk` / `unhealthy` / `fit` before building the submission file — Kaggle
expects the string labels exactly as shown in the sample submission.

## 2. Format the submission file

```python
sample_sub = pd.read_csv("data/raw/sample_submission.csv")
print(sample_sub.columns)   # should be: id, health_condition

submission = pd.DataFrame({
    "id": test_ids,
    "health_condition": preds
})
submission.to_csv("submissions/submission_v1.csv", index=False)
```

## 3. Submit via CLI (or the Kaggle website upload)

```bash
kaggle competitions submit -c playground-series-s6e7 \
  -f submissions/submission_v1.csv \
  -m "v1: baseline RF, val balanced_accuracy=0.84"
```

## 4. Check your public leaderboard score

```bash
kaggle competitions leaderboard -c playground-series-s6e7 -s
```
- [ ] Screenshot the public leaderboard entry — you'll need this later

## 5. Iterate

Go back to Phase 5, try to improve (feature tweaks, tuning, different model,
double-check `class_weight="balanced"` is set given the balanced-accuracy
metric), then repeat steps 1–4 with a new version:

```bash
kaggle competitions submit -c playground-series-s6e7 \
  -f submissions/submission_v2.csv \
  -m "v2: tuned XGBoost, val balanced_accuracy=0.87"
```

## 6. Final submission before deadline

- [ ] Make your last, best submission with time buffer before **July 31, 2026**
      (not your coursework deadline — check both separately)
- [ ] Screenshot the **final/private leaderboard** result once it's revealed
      (private scores often only appear after the competition closes)
- [ ] Save both screenshots — public and private/final — you'll need them as
      evidence later

## Deliverable checklist

- [ ] At least one **public** leaderboard submission made
- [ ] At least one **final/private** leaderboard submission made
- [ ] Submission CSVs saved in `submissions/`
- [ ] Screenshots of both leaderboard results saved somewhere safe

Next: `07_app_development.md`