# Phase 6 — Kaggle Submission

Script: `src/predict.py`

## 1. Apply the SAME fitted pipeline to Kaggle's test set

```python
import joblib
import pandas as pd

preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/model_final.pkl")   # or tf.keras.models.load_model(...) for NN

test = pd.read_csv("data/raw/test.csv")
test_ids = test["id"]   # or whatever the id column is called

X_test_processed = preprocessor.transform(test.drop(columns=["id"]))
preds = model.predict(X_test_processed)
```

Do NOT re-fit the preprocessor here. Fit only ever happened on training data
in Phase 4 — this step just transforms.

## 2. Format the submission file

```python
sample_sub = pd.read_csv("data/raw/sample_submission.csv")
print(sample_sub.columns)   # match this exactly

submission = pd.DataFrame({
    "id": test_ids,
    "<target_col>": preds
})
submission.to_csv("submissions/submission_v1.csv", index=False)
```

## 3. Submit via CLI (or the Kaggle website upload)

```bash
kaggle competitions submit -c <competition-slug> \
  -f submissions/submission_v1.csv \
  -m "v1: baseline RF, val f1=0.84"
```

## 4. Check your public leaderboard score

```bash
kaggle competitions leaderboard -c <competition-slug> -s
```
- [ ] Screenshot the public leaderboard entry — you'll need this later

## 5. Iterate

Go back to Phase 5, try to improve (feature tweaks, tuning, different model),
then repeat steps 1–4 with a new version:

```bash
kaggle competitions submit -c <competition-slug> \
  -f submissions/submission_v2.csv \
  -m "v2: tuned XGBoost, val f1=0.87"
```

## 6. Final submission before deadline

- [ ] Make your last, best submission with time buffer before the competition's
      own deadline (not your coursework deadline — check both)
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
