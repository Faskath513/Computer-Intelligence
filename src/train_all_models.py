"""
Train 3 strong models + ensemble for Playground Series S5E1
Metric: MAPE (regression)
Models: Random Forest, XGBoost, LightGBM, Ensemble
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
import lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
MODELS = os.path.join(BASE, "models")
SUBMISSIONS = os.path.join(BASE, "submissions")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(SUBMISSIONS, exist_ok=True)

TARGET = "health_condition"
ID_COL = "id"


def load_and_split():
    print("Loading data...")
    train = pd.read_csv(os.path.join(RAW, "train.csv"))
    test = pd.read_csv(os.path.join(RAW, "test.csv"))
    print(f"Train: {train.shape}, Test: {test.shape}")

    ids = test[ID_COL].copy()
    X = train.drop(columns=[ID_COL, TARGET])
    y = train[TARGET]
    X_test = test.drop(columns=[ID_COL])

    return X, y, X_test, ids


def build_preprocessor(X):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ], remainder="drop")
    return preprocessor


def train_random_forest(X_proc, y, X_val_proc, y_val, sample_weight_train):
    print("\n" + "=" * 60)
    print("MODEL 1: RANDOM FOREST (Aggressively Tuned)")
    print("=" * 60)
    t0 = time.time()

    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_proc, y, sample_weight=sample_weight_train)

    preds = rf.predict(X_val_proc)
    ba = balanced_accuracy_score(y_val, preds)
    print(f"Balanced Accuracy: {ba:.4f}")
    print(f"OOB Score: {rf.oob_score_:.4f}")
    print(classification_report(y_val, preds))
    print(f"Time: {time.time() - t0:.1f}s")

    return rf, ba


def train_xgboost(X_proc, y, X_val_proc, y_val, le, sample_weight_train):
    print("\n" + "=" * 60)
    print("MODEL 2: XGBOOST (Gradient Boosting)")
    print("=" * 60)
    t0 = time.time()

    y_enc = le.transform(y)
    y_val_enc = le.transform(y_val)

    xgb_model = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1,
        objective="multi:softprob",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=100,
    )

    xgb_model.fit(
        X_proc, y_enc,
        sample_weight=sample_weight_train,
        eval_set=[(X_val_proc, y_val_enc)],
        verbose=100,
    )

    preds_enc = xgb_model.predict(X_val_proc)
    preds = le.inverse_transform(preds_enc)
    ba = balanced_accuracy_score(y_val, preds)
    print(f"Balanced Accuracy: {ba:.4f}")
    print(f"Best iteration: {xgb_model.best_iteration}")
    print(classification_report(y_val, preds))
    print(f"Time: {time.time() - t0:.1f}s")

    return xgb_model, ba


def train_lightgbm(X_proc, y, X_val_proc, y_val, le, sample_weight_train):
    print("\n" + "=" * 60)
    print("MODEL 3: LIGHTGBM (Gradient Boosting)")
    print("=" * 60)
    t0 = time.time()

    y_enc = le.transform(y)
    y_val_enc = le.transform(y_val)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=3000,
        max_depth=10,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        num_leaves=63,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        objective="multiclass",
        num_class=len(le.classes_),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    lgb_model.fit(
        X_proc, y_enc,
        eval_set=[(X_val_proc, y_val_enc)],
        callbacks=[
            lgb.early_stopping(200, verbose=True),
            lgb.log_evaluation(200),
        ],
    )

    preds_enc = lgb_model.predict(X_val_proc)
    preds = le.inverse_transform(preds_enc)
    ba = balanced_accuracy_score(y_val, preds)
    print(f"Balanced Accuracy: {ba:.4f}")
    print(f"Best iteration: {lgb_model.best_iteration_}")
    print(classification_report(y_val, preds))
    print(f"Time: {time.time() - t0:.1f}s")

    return lgb_model, ba


def cross_validate_best(models_dict, X_proc, y, cv=5):
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION OF BEST MODEL")
    print("=" * 60)

    best_name = max(models_dict, key=models_dict.get)
    print(f"Best so far: {best_name} ({models_dict[best_name]:.4f})")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    if best_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=1000, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    elif best_name == "XGBoost":
        model = xgb.XGBClassifier(
            n_estimators=1500, max_depth=8, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", use_label_encoder=False,
            eval_metric="mlogloss", random_state=42, n_jobs=-1
        )
    else:
        le_cv = LabelEncoder()
        y_cv = le_cv.fit_transform(y)
        model = lgb.LGBMClassifier(
            n_estimators=2000, max_depth=10, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8, num_leaves=63,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
        )

    if best_name == "LightGBM":
        le_cv = LabelEncoder()
        y_cv = le_cv.fit_transform(y)
        scores = cross_val_score(model, X_proc, y_cv, cv=skf, scoring="balanced_accuracy", n_jobs=-1)
    else:
        scores = cross_val_score(model, X_proc, y, cv=skf, scoring="balanced_accuracy", n_jobs=-1)

    print(f"CV Balanced Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
    print(f"Per-fold: {[f'{s:.4f}' for s in scores]}")
    return scores.mean(), scores.std()


def generate_submission(best_model_name, preprocessor, X_test, y, ids, le=None):
    print("\n" + "=" * 60)
    print("GENERATING SUBMISSION")
    print("=" * 60)

    X_test_proc = preprocessor.transform(X_test)

    if best_model_name == "RandomForest":
        model_path = os.path.join(MODELS, "model_rf_v3.pkl")
        model = joblib.load(model_path)
        preds = model.predict(X_test_proc)
    elif best_model_name == "XGBoost":
        model_path = os.path.join(MODELS, "model_xgb_v3.json")
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        preds_enc = model.predict(X_test_proc)
        preds = le.inverse_transform(preds_enc)
    else:
        model_path = os.path.join(MODELS, "model_lgb_v3.txt")
        model = lgb.LGBMClassifier()
        model = lgb.Booster(model_file=model_path)
        preds_enc = np.argmax(model.predict(X_test_proc), axis=1)
        preds = le.inverse_transform(preds_enc)

    sub = pd.DataFrame({"id": ids, "health_condition": preds})
    out_path = os.path.join(SUBMISSIONS, f"submission_v3_{best_model_name.lower()}.csv")
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(sub["health_condition"].value_counts())
    return sub


def main():
    from sklearn.model_selection import train_test_split

    X, y, X_test, ids = load_and_split()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    print(f"Classes: {le.classes_} ({n_classes} classes)")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_tr.shape}, Val: {X_val.shape}")

    preprocessor = build_preprocessor(X)
    preprocessor.fit(X_tr)
    joblib.dump(preprocessor, os.path.join(MODELS, "preprocessor_v3.pkl"))
    print("Preprocessor saved")

    X_tr_proc = preprocessor.transform(X_tr)
    X_val_proc = preprocessor.transform(X_val)

    classes = np.unique(y_tr)
    cw = compute_class_weight("balanced", classes=classes, y=y_tr)
    weight_dict = dict(zip(classes, cw))
    sw_train = y_tr.map(weight_dict).values

    results = {}

    # --- Model 1: Random Forest ---
    rf, rf_ba = train_random_forest(X_tr_proc, y_tr, X_val_proc, y_val, sw_train)
    results["RandomForest"] = rf_ba
    joblib.dump(rf, os.path.join(MODELS, "model_rf_v3.pkl"))

    # --- Model 2: XGBoost ---
    xgb_m, xgb_ba = train_xgboost(X_tr_proc, y_tr, X_val_proc, y_val, le, sw_train)
    results["XGBoost"] = xgb_ba
    xgb_m.save_model(os.path.join(MODELS, "model_xgb_v3.json"))

    # --- Model 3: LightGBM ---
    lgb_m, lgb_ba = train_lightgbm(X_tr_proc, y_tr, X_val_proc, y_val, le, sw_train)
    results["LightGBM"] = lgb_ba
    lgb_m.booster_.save_model(os.path.join(MODELS, "model_lgb_v3.txt"))

    # --- Summary ---
    print("\n" + "=" * 60)
    print("ALL RESULTS")
    print("=" * 60)
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:20s}: {score:.4f}")

    best_name = max(results, key=results.get)
    print(f"\nBEST MODEL: {best_name} ({results[best_name]:.4f})")

    # --- Cross-validate best ---
    # We need full data CV
    print(f"\nRunning 5-fold CV on best model ({best_name})...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_all_proc = preprocessor.transform(X)

    y_all_enc = le.transform(y)

    if best_name == "LightGBM":
        cv_model = lgb.LGBMClassifier(
            n_estimators=3000, max_depth=10, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8, num_leaves=63,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
        )
        cv_scores = cross_val_score(cv_model, X_all_proc, y_all_enc, cv=skf, scoring="balanced_accuracy", n_jobs=-1)
    elif best_name == "XGBoost":
        cv_model = xgb.XGBClassifier(
            n_estimators=2000, max_depth=8, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1,
            objective="multi:softprob", use_label_encoder=False,
            eval_metric="mlogloss", random_state=42, n_jobs=-1
        )
        cv_scores = cross_val_score(cv_model, X_all_proc, y_all_enc, cv=skf, scoring="balanced_accuracy", n_jobs=-1)
    else:
        cv_model = RandomForestClassifier(
            n_estimators=1000, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        cv_scores = cross_val_score(cv_model, X_all_proc, y, cv=skf, scoring="balanced_accuracy", n_jobs=-1)

    print(f"CV Balanced Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # --- Generate submission ---
    generate_submission(best_name, preprocessor, X_test, y, ids, le)

    # --- Also generate submissions for all 3 models ---
    print("\nGenerating submissions for ALL models...")
    X_test_proc = preprocessor.transform(X_test)

    for name, model_file, model_type in [
        ("rf", "model_rf_v3.pkl", "sklearn"),
        ("xgb", "model_xgb_v3.json", "xgb"),
        ("lgb", "model_lgb_v3.txt", "lgb"),
    ]:
        fpath = os.path.join(MODELS, model_file)
        if not os.path.exists(fpath):
            continue
        if model_type == "sklearn":
            m = joblib.load(fpath)
            preds = m.predict(X_test_proc)
        elif model_type == "xgb":
            m = xgb.XGBClassifier()
            m.load_model(fpath)
            preds = le.inverse_transform(m.predict(X_test_proc))
        else:
            m = lgb.Booster(model_file=fpath)
            preds = le.inverse_transform(np.argmax(m.predict(X_test_proc), axis=1))

        sub = pd.DataFrame({"id": ids, "health_condition": preds})
        sub.to_csv(os.path.join(SUBMISSIONS, f"submission_v3_{name}.csv"), index=False)
        print(f"  submission_v3_{name}.csv saved")

    # --- Save results ---
    results_df = pd.DataFrame([
        {"Model": k, "Val balanced_accuracy": v} for k, v in sorted(results.items(), key=lambda x: -x[1])
    ])
    results_df.to_csv(os.path.join(MODELS, "results_v3.csv"), index=False)
    print("\nAll done!")


if __name__ == "__main__":
    main()
