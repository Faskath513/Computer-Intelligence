# Phase 5 — Modeling

Notebooks: `notebooks/02_baseline_models.ipynb`, `notebooks/03_final_model.ipynb`
Script: `src/train.py`

Task: **3-class classification** — target `health_condition` with classes
`at-risk`, `unhealthy`, `fit`. Scored on **balanced accuracy**, which is the
average of per-class recall. This has direct code implications below — don't
skip the class-imbalance handling, or your local score won't match Kaggle's.

Build at least two techniques so you have a real comparison. Suggested combo:

| Model | Why |
|---|---|
| Logistic Regression | Fast baseline, sanity check |
| Random Forest or XGBoost | Strong default for tabular data |
| Neural Network (Keras MLP, softmax) | Matches "Deep Learning" emphasis of the module |

## 0. Check class balance first (do this before anything else)

```python
y_train.value_counts(normalize=True)
```
If one class dominates, plain accuracy would look good while balanced
accuracy (Kaggle's actual metric) stays low — this is exactly why every
model below uses `class_weight="balanced"` or an equivalent.

## 1. Baseline model

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report

baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
baseline.fit(X_train_processed, y_train)
preds = baseline.predict(X_val_processed)

print("Balanced accuracy:", balanced_accuracy_score(y_val, preds))
print(classification_report(y_val, preds))
```

## 2. Ensemble model

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

If using XGBoost/LightGBM instead, they don't have a `class_weight="balanced"`
shortcut — pass `sample_weight` computed manually, or use
`sklearn.utils.class_weight.compute_sample_weight("balanced", y_train)` and
pass that into `.fit(..., sample_weight=weights)`.

## 3. Neural network (multiclass — 3 output units, softmax)

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Encode string labels -> integers (0=at-risk, 1=fit, 2=unhealthy, etc. — check .classes_)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)

# Balanced accuracy equivalent for Keras: pass class weights into .fit()
class_weights = compute_class_weight("balanced", classes=np.unique(y_train_enc), y=y_train_enc)
class_weight_dict = dict(enumerate(class_weights))

nn = models.Sequential([
    layers.Input(shape=(X_train_processed.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax")   # 3 classes: at-risk / unhealthy / fit
])
nn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = nn.fit(
    X_train_processed, y_train_enc,
    validation_data=(X_val_processed, y_val_enc),
    epochs=50, batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# Evaluate with the ACTUAL competition metric (Keras doesn't track balanced accuracy natively)
from sklearn.metrics import balanced_accuracy_score
nn_preds = np.argmax(nn.predict(X_val_processed), axis=1)
print("NN Balanced accuracy:", balanced_accuracy_score(y_val_enc, nn_preds))
```

Keep `le` (the `LabelEncoder`) saved alongside the model — you'll need it to
decode predictions back to `at-risk`/`unhealthy`/`fit` strings at submission
time (see Phase 6).

## 4. Cross-validation on your best candidate(s)

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(best_rf, X_train_processed, y_train, cv=5, scoring="balanced_accuracy")
scores.mean(), scores.std()
```

## 5. Comparison table — keep this updated as you go

Create `experiments.md` in project root:

```markdown
| Model | Val balanced accuracy | CV mean ± std | Notes |
|---|---|---|---|
| LogisticRegression (balanced) | 0.71 | - | baseline |
| RandomForest (tuned, balanced) | 0.84 | 0.83 ± 0.02 | best_params: ... |
| Neural Net (MLP, class-weighted) | 0.82 | - | 50 epochs, early stopping at 23 |
```

## 6. Save the winning model

```python
import joblib
joblib.dump(best_rf, "models/model_final.pkl")
joblib.dump(le, "models/label_encoder.pkl")   # only needed if the winner is the NN
# or for Keras:
nn.save("models/model_final.h5")
```

## Deliverable checklist

- [ ] Confirmed class balance of `health_condition` before modeling
- [ ] ≥2 modeling techniques trained, all handling class imbalance
      (`class_weight="balanced"` or equivalent)
- [ ] All models evaluated with `balanced_accuracy_score`, not plain accuracy
- [ ] Best candidate(s) cross-validated, not just single-split
- [ ] `experiments.md` filled in with real numbers
- [ ] Final model saved to `models/model_final.*` (+ label encoder if using the NN)

Next: `06_kaggle_submission.md`