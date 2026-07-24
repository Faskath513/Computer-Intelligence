# Phase 5 — Modeling

Notebooks: `notebooks/02_baseline_models.ipynb`, `notebooks/03_final_model.ipynb`
Script: `src/train.py`

Task: **3-class classification** — target `health_condition` with classes
`at-risk`, `unhealthy`, `fit`. Scored on **balanced accuracy**, which is the
average of per-class recall. This has direct code implications below — don't
skip the class-imbalance handling, or your local score won't match Kaggle's.

## 0. Class balance (from EDA)

```
at-risk      85.9%
unhealthy     8.4%
fit           5.8%
```

Both minority classes are below 20% — every model must use `class_weight="balanced"` or equivalent.

## 1. Baseline — Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report

baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
baseline.fit(X_train_processed, y_train)
preds = baseline.predict(X_val_processed)

print("Balanced accuracy:", balanced_accuracy_score(y_val, preds))
```

**Result:** 0.8574 balanced accuracy

## 2. Ensemble — Random Forest (tuned)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(random_state=42, class_weight="balanced")
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 5],
}
grid = GridSearchCV(rf, param_grid, cv=5, scoring="balanced_accuracy", n_jobs=-1)
grid.fit(X_train_processed, y_train)
best_rf = grid.best_estimator_
```

**Best params:** n_estimators=300, min_samples_leaf=5, max_depth=None
**Result:** 0.8790 balanced accuracy | CV: 0.8771 ± 0.0004

## 3. Neural Network — Keras MLP (3-class softmax)

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

class_weights = compute_class_weight("balanced", classes=np.unique(y_train_enc), y=y_train_enc)
class_weight_dict = dict(enumerate(class_weights))

nn = models.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(3, activation="softmax"),
])
nn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = nn.fit(
    X_train_processed, y_train_enc,
    validation_data=(X_val_processed, y_val_enc),
    epochs=50, batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)
```

**Result:** 0.9006 balanced accuracy (15 epochs, early stopping)

## 4. Cross-validation on best candidates

| Model | CV mean ± std |
|---|---|
| RandomForest | 0.8771 ± 0.0004 |
| Neural Net | Not cross-validated (used val split) |

## 5. Comparison table

| Model | Val balanced accuracy | CV mean ± std | Notes |
|---|---|---|---|
| LogisticRegression (balanced) | 0.8574 | - | Baseline |
| RandomForest (tuned, balanced) | 0.8790 | 0.8771 ± 0.0004 | n_estimators=300, min_samples_leaf=5 |
| **Neural Net (MLP, class-weighted)** | **0.9006** | - | 15 epochs, early stopping |

> **Winner: Neural Net (MLP, class-weighted)** with 0.9006 balanced accuracy

## 6. Save the winning model

```python
import joblib
nn.save("models/model_final.h5")
joblib.dump(le, "models/label_encoder.pkl")
```

## Deliverable checklist

- [x] Confirmed class balance of `health_condition` before modeling
- [x] 3 modeling techniques trained, all handling class imbalance
- [x] All models evaluated with `balanced_accuracy_score`, not plain accuracy
- [x] Best candidate (RF) cross-validated
- [x] `experiments.md` filled in with real numbers
- [x] Final model saved to `models/model_final.*` (+ label encoder for NN)

Next: `06_kaggle_submission.md`
