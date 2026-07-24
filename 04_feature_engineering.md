# Phase 4 — Feature Engineering & Preprocessing Pipeline

Files: `src/data_prep.py`, `src/features.py`

Everything here must be wrapped in reusable functions / an sklearn `Pipeline` —
you'll call this exact same code at Kaggle-submission time AND inside the app,
so no notebook-only logic.

## 1. `src/data_prep.py` — loading & splitting

```python
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(raw_dir="data/raw"):
    train = pd.read_csv(f"{raw_dir}/train.csv")
    test = pd.read_csv(f"{raw_dir}/test.csv")
    return train, test

def split_data(train, target_col, test_size=0.2, stratify=False, random_state=42):
    X = train.drop(columns=[target_col])
    y = train[target_col]
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, stratify=strat, random_state=random_state)
```

## 2. `src/features.py` — the preprocessing pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])
    return preprocessor
```

Apply your EDA findings here specifically:
- [ ] High-cardinality categorical → swap `OneHotEncoder` for target/frequency encoding on that column
- [ ] Skewed target → apply `np.log1p` before training, `np.expm1` on predictions
- [ ] Correlated/redundant features → drop before building the `ColumnTransformer`
- [ ] Outliers → clip or leave, based on your EDA decision — document which

## 3. Fit on train, apply to val/test consistently

```python
preprocessor = build_preprocessor(numeric_cols, categorical_cols)
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)     # NEVER fit on val/test
```

## 4. Persist the fitted pipeline

```python
import joblib
joblib.dump(preprocessor, "models/preprocessor.pkl")
```
This is what the app will load later — it must be the exact fitted object,
not a re-built one, so train/inference stay consistent.

## Deliverable checklist

- [ ] `data_prep.py` and `features.py` importable with no notebook dependency
- [ ] Preprocessor fit only on training data, applied (not re-fit) to val/test
- [ ] `models/preprocessor.pkl` saved
- [ ] Quick unit check: shape of `X_train_processed` matches expected feature count

Next: `05_modeling.md`
