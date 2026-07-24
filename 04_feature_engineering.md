# Phase 4 — Feature Engineering & Preprocessing Pipeline

Files: `src/data_prep.py`, `src/features.py`, `src/save_feature_meta.py`

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
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ], remainder="drop")
    return preprocessor
```

### EDA-driven decisions applied here:

- **Missing values (up to 12%):** Median imputation for numerics, mode imputation for categoricals — handles the 6 features with missing data.
- **No high-cardinality columns:** All categoricals have 2-3 unique values → OneHotEncoder is appropriate (no target/frequency encoding needed).
- **No extreme outliers:** All numeric features have reasonable distributions → StandardScaler applied without clipping.
- **No correlated features to drop:** No extremely high pairwise correlations → all 7 numeric features retained.

## 3. Fit on train, apply to val/test consistently

```python
preprocessor = build_preprocessor(numeric_cols, categorical_cols)
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)     # NEVER fit on val/test
```

After fitting: **7 numeric + 14 one-hot encoded categorical = 21 features**.

## 4. Persist the fitted pipeline

```python
import joblib
joblib.dump(preprocessor, "models/preprocessor.pkl")
```
This is what the app loads — it must be the exact fitted object, not a re-built one.

## 5. Feature metadata for the Streamlit app

`src/save_feature_meta.py` generates `models/feature_meta.json` containing:
- List of numeric and categorical feature names
- Min/max/mean/step for each numeric feature (for Streamlit number_input widgets)
- Unique values for each categorical feature (for Streamlit selectbox widgets)

## Deliverable checklist

- [x] `data_prep.py` and `features.py` importable with no notebook dependency
- [x] Preprocessor fit only on training data, applied (not re-fit) to val/test
- [x] `models/preprocessor.pkl` saved
- [x] Quick unit check: shape of `X_train_processed` matches expected feature count (21 columns)

Next: `05_modeling.md`
