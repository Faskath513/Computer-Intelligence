# Phase 5 — Modeling

Notebooks: `notebooks/02_baseline_models.ipynb`, `notebooks/03_final_model.ipynb`
Script: `src/train.py`

Build at least two techniques so you have a real comparison. Suggested combo:

| Model | Why |
|---|---|
| Logistic/Linear Regression | Fast baseline, sanity check |
| Random Forest or XGBoost | Strong default for tabular data |
| Neural Network (Keras MLP) | Matches "Deep Learning" emphasis of the module |

## 1. Baseline model

```python
from sklearn.linear_model import LogisticRegression   # or LinearRegression
from sklearn.metrics import classification_report, mean_squared_error

baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train_processed, y_train)
preds = baseline.predict(X_val_processed)
print(classification_report(y_val, preds))   # or RMSE for regression
```

## 2. Ensemble model

```python
from sklearn.ensemble import RandomForestClassifier   # or Regressor
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 5],
}
grid = GridSearchCV(rf, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
grid.fit(X_train_processed, y_train)
best_rf = grid.best_estimator_
```

## 3. Neural network

```python
import tensorflow as tf
from tensorflow.keras import layers, models

nn = models.Sequential([
    layers.Input(shape=(X_train_processed.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")   # adjust for regression / multiclass
])
nn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = nn.fit(
    X_train_processed, y_train,
    validation_data=(X_val_processed, y_val),
    epochs=50, batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)
```

## 4. Cross-validation on your best candidate(s)

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(best_rf, X_train_processed, y_train, cv=5, scoring="f1_macro")
scores.mean(), scores.std()
```

## 5. Comparison table — keep this updated as you go

Create `experiments.md` in project root:

```markdown
| Model | Val metric | CV mean ± std | Notes |
|---|---|---|---|
| LogisticRegression | 0.71 | - | baseline |
| RandomForest (tuned) | 0.84 | 0.83 ± 0.02 | best_params: ... |
| Neural Net (MLP) | 0.82 | - | 50 epochs, early stopping at 23 |
```

## 6. Save the winning model

```python
import joblib
joblib.dump(best_rf, "models/model_final.pkl")
# or for Keras:
nn.save("models/model_final.h5")
```

## Deliverable checklist

- [ ] ≥2 modeling techniques trained and evaluated on the same val split
- [ ] Best candidate(s) cross-validated, not just single-split
- [ ] `experiments.md` filled in with real numbers
- [ ] Final model saved to `models/model_final.*`

Next: `06_kaggle_submission.md`
