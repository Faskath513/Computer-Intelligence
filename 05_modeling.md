# Phase 5 — Modeling

Scripts: `src/train.py`, `src/train_all_models.py`, `src/train_v4.py`

Task: **3-class classification** — target `health_condition` with classes
`at-risk`, `unhealthy`, `fit`. Scored on **balanced accuracy** (average per-class recall).

## 0. Class balance (from EDA)

```
at-risk      85.9%
unhealthy     8.4%
fit           5.8%
```

Both minority classes below 20% — every model uses `class_weight="balanced"`.

## V1-V2 Models (Original Features, 25 features)

### Logistic Regression (baseline)

```python
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
```
**Result:** 0.8574 balanced accuracy

### Random Forest (GridSearchCV tuned)

```python
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                            class_weight="balanced", random_state=42)
```
**Result:** 0.8790 balanced accuracy | CV: 0.8771 ± 0.0004

### Neural Network (Keras MLP)

```python
nn = Sequential([
    Dense(128, activation="relu"), Dropout(0.3),
    Dense(64, activation="relu"), Dropout(0.2),
    Dense(3, activation="softmax"),
])
```
**Result:** 0.9006 balanced accuracy (15 epochs, early stopping)

## V3 Models (No Feature Engineering, 25 features)

| Model | Val Balanced Accuracy | Notes |
|---|---|---|
| XGBoost | 0.9027 | Overfit — public score only 0.85488 |
| LightGBM | 0.8987 | Best generalization — public 0.89764 |
| RandomForest | 0.8762 | Solid baseline |

## V4 Models (Feature Engineering + Regularization, 56 features)

| Model | Val Balanced Accuracy | Notes |
|---|---|---|
| **LightGBM v3 (conservative)** | **0.9337** | max_depth=6, lr=0.005, reg_alpha=1.0 |
| LightGBM v2 (regularized) | 0.9333 | max_depth=7, lr=0.01, reg_alpha=0.5 |
| XGBoost v4 (regularized) | ~0.90 | max_depth=6, gamma=0.5, reg_lambda=2.0 |
| RandomForest v4 | 0.8977 | n_estimators=1500, min_samples_leaf=5 |

### V4 LightGBM v3 Best Config

```python
lgb.LGBMClassifier(
    n_estimators=5000, max_depth=6, learning_rate=0.005,
    subsample=0.65, colsample_bytree=0.65, min_child_weight=15,
    num_leaves=40, reg_alpha=1.0, reg_lambda=3.0,
    class_weight="balanced", min_gain_to_split=0.2, max_bin=200,
    random_state=42, n_jobs=-1, verbose=-1,
)
```

## Key Learnings

1. **XGBoost overfits** — high val (0.9027) but low public (0.85488)
2. **LightGBM generalizes best** — val and public scores closely match
3. **Feature engineering helps** — V4 val scores jumped from ~0.90 to ~0.93
4. **Regularization is critical** — more regularization = better generalization
5. **class_weight="balanced"** is essential for all models given 85.9% / 8.4% / 5.8% split

## Kaggle Public Scores

| Submission | Public Score |
|---|---|
| submission_v3_lgb_final.csv | **0.89764** |
| submission_v2_rf_final.csv | 0.89314 |
| submission_v3_xgb_final.csv | 0.85488 |

## Deliverable checklist

- [x] Confirmed class balance before modeling
- [x] 3+ modeling techniques trained, all handling class imbalance
- [x] All models evaluated with `balanced_accuracy_score`
- [x] Best candidates cross-validated
- [x] `experiments.md` filled in with V4 numbers
- [x] Final models saved to `models/`

Next: `06_kaggle_submission.md`
