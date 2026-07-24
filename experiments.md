# Model Comparison -- Playground Series S6E7

| Model | Val balanced accuracy | CV mean +/- std | Notes |
|---|---|---|---|
| LogisticRegression (balanced) | 0.5937 | - | Baseline model |
| RandomForest (tuned, balanced) | **0.6734** | 0.6583 +/- 0.0220 | WINNER -- n_estimators=300, min_samples_leaf=5, class_weight=balanced |
| Neural Net (Keras MLP, class-weighted) | 0.6221 | - | 19 epochs, early stopping, 3-class softmax |

> RF wins on balanced accuracy. Final submission uses models/model_final.pkl (RandomForest).
