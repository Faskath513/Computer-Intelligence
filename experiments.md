# Model Comparison -- Playground Series S6E7

## V1-V2 (Initial)

| Model | Val balanced accuracy | CV mean +/- std | Notes |
|---|---|---|---|
| LogisticRegression (balanced) | 0.8574 | - | Baseline |
| RandomForest (tuned, balanced) | 0.8790 | 0.8771 +/- 0.0004 | n_estimators=300, min_samples_leaf=5 |
| Neural Net (MLP, class-weighted) | 0.9006 | - | 15 epochs, early stopping |

> **V2 Winner: Neural Net (MLP, class-weighted)**

### Kaggle Public Scores (V2/V3)

| Submission | Public Score |
|---|---|
| submission_v3_lgb_final.csv | **0.89764** |
| submission_v2_rf_final.csv | 0.89314 |
| submission_v3_xgb_final.csv | 0.85488 |

## V3 (No Feature Engineering, 3 Models)

| Model | Val balanced accuracy | Kaggle Public | Notes |
|---|---|---|---|
| XGBoost | 0.9027 | 0.85488 | Overfit to training data |
| LightGBM | 0.8987 | 0.89764 | Best generalization |
| RandomForest | 0.8762 | 0.89314 | Solid baseline |

> **V3 Winner: LightGBM** — best generalization (val 0.8987, public 0.89764)

## V4 (Feature Engineering + Regularization)

Added 31 engineered features: BMI categories, sleep/exercise flags, interaction features,
risk scores, encoded numerics. Better regularization to reduce overfitting.

| Model | Val balanced accuracy | Notes |
|---|---|---|
| LightGBM v3 (conservative) | **0.9337** | max_depth=6, lr=0.005, reg_alpha=1.0 |
| LightGBM v2 (regularized) | 0.9333 | max_depth=7, lr=0.01, reg_alpha=0.5 |
| XGBoost v4 (regularized) | ~0.90 | max_depth=6, gamma=0.5, reg_lambda=2.0 |
| RandomForest v4 | 0.8977 | n_estimators=1500, min_samples_leaf=5 |
| Neural Network v4 (MLP) | ~0.9088 | 192→96→3, batch 2048, early stopping |

> **V4 Winner: LightGBM v3** (val 0.9337) — upload `submission_v4_lgb_v3.csv`

### All V4 Submissions Ready

| Submission | Model | Val balanced accuracy |
|---|---|---|
| `submission_v4_lgb_v2.csv` | LightGBM v2 (regularized) | 0.9333 |
| `submission_v4_lgb_v3.csv` | LightGBM v3 (conservative) | **0.9337** |
| `submission_v4_rf.csv` | RandomForest v4 | 0.8977 |
| `submission_v4_xgb.csv` | XGBoost v4 (regularized) | ~0.90 |
| `submission_v4_nn.csv` | Neural Network v4 | ~0.9088 |

### Key Learnings

1. **XGBoost overfits** on this dataset — high val score (0.9027) but lowest public (0.85488)
2. **LightGBM generalizes best** — val and public scores closely match
3. **Feature engineering helps** — v4 val scores jumped from ~0.90 to ~0.93
4. **Regularization is critical** — more regularization = better generalization to public
5. **Class imbalance** (at-risk 85%, fit 6%, unhealthy 8%) — class_weight="balanced" essential
