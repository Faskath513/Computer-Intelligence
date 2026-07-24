"""
Run this ONCE after training to save feature metadata for the Streamlit app.
Usage: python src/save_feature_meta.py
"""
import json
import pandas as pd
from data_prep import load_data, get_feature_cols, get_column_types


def build_feature_meta(raw_dir="data/raw", out_path="models/feature_meta.json"):
    train_df, _ = load_data(raw_dir)
    X = get_feature_cols(train_df)

    numeric_cols, categorical_cols = get_column_types(X)

    numeric_meta = {}
    for col in numeric_cols:
        numeric_meta[col] = {
            "min":  round(float(X[col].min()), 2),
            "max":  round(float(X[col].max()), 2),
            "mean": round(float(X[col].mean()), 2),
            "step": 1.0 if X[col].dtype in ["int64", "int32"] else 0.1,
        }

    categorical_opts = {}
    for col in categorical_cols:
        categorical_opts[col] = sorted(X[col].dropna().unique().tolist())

    meta = {
        "numeric":      numeric_cols,
        "numeric_meta": numeric_meta,
        "categorical":  categorical_opts,
    }

    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Feature metadata saved → {out_path}")
    print(f"  Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"  Categorical features ({len(categorical_cols)}): {list(categorical_opts.keys())}")
    return meta


if __name__ == "__main__":
    build_feature_meta()
