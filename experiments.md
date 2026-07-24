# Model Comparison -- Playground Series S6E7

| Model | Val balanced accuracy | CV mean +/- std | Notes |
|---|---|---|---|
| LogisticRegression (balanced) | 0.8574 | - | Baseline |
| RandomForest (tuned, balanced) | 0.8790 | 0.8771 +/- 0.0004 | n_estimators=300, min_samples_leaf=5 |
| Neural Net (MLP, class-weighted) | **0.9006** | - | 15 epochs, early stopping |

> **Winner: Neural Net (MLP, class-weighted)**
