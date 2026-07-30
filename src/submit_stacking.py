"""Stacking Ensemble - LightGBM + RF + XGBoost"""
import os, numpy as np, pandas as pd, joblib, warnings, time
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
MODELS = os.path.join(BASE, "models")
SUBMISSIONS = os.path.join(BASE, "submissions")

TARGET = "health_condition"


def add_features(df):
    df = df.copy()
    if "sleep_duration" in df.columns:
        df["sleep_deprived"] = (df["sleep_duration"] < 6).astype(int)
        df["good_sleep"] = (df["sleep_duration"] >= 7).astype(int)
        df["oversleep"] = (df["sleep_duration"] > 8.5).astype(int)
    if "heart_rate" in df.columns:
        df["hr_low"] = (df["heart_rate"] < 60).astype(int)
        df["hr_normal"] = ((df["heart_rate"] >= 60) & (df["heart_rate"] <= 100)).astype(int)
        df["hr_high"] = (df["heart_rate"] > 100).astype(int)
    if "bmi" in df.columns:
        df["bmi_underweight"] = (df["bmi"] < 18.5).astype(int)
        df["bmi_normal"] = ((df["bmi"] >= 18.5) & (df["bmi"] < 25)).astype(int)
        df["bmi_overweight"] = ((df["bmi"] >= 25) & (df["bmi"] < 30)).astype(int)
        df["bmi_obese"] = (df["bmi"] >= 30).astype(int)
    if "step_count" in df.columns:
        df["low_steps"] = (df["step_count"] < 5000).astype(int)
        df["high_steps"] = (df["step_count"] > 10000).astype(int)
    if "exercise_duration" in df.columns:
        df["low_exercise"] = (df["exercise_duration"] < 20).astype(int)
        df["high_exercise"] = (df["exercise_duration"] > 60).astype(int)
    if "water_intake" in df.columns:
        df["low_water"] = (df["water_intake"] < 1.5).astype(int)
        df["good_water"] = (df["water_intake"] > 2.5).astype(int)
    if "calorie_expenditure" in df.columns:
        df["low_calorie"] = (df["calorie_expenditure"] < 1500).astype(int)
        df["high_calorie"] = (df["calorie_expenditure"] > 2800).astype(int)
    if all(c in df.columns for c in ["sleep_duration", "exercise_duration"]):
        df["sleep_exercise_ratio"] = df["sleep_duration"] / (df["exercise_duration"] / 60 + 0.1)
        df["sleep_x_exercise"] = df["sleep_duration"] * df["exercise_duration"]
    if all(c in df.columns for c in ["bmi", "exercise_duration"]):
        df["bmi_x_exercise"] = df["bmi"] * df["exercise_duration"]
    if all(c in df.columns for c in ["bmi", "sleep_duration"]):
        df["bmi_x_sleep"] = df["bmi"] * df["sleep_duration"]
    if all(c in df.columns for c in ["heart_rate", "bmi"]):
        df["hr_bmi_ratio"] = df["heart_rate"] / (df["bmi"] + 0.1)
    if all(c in df.columns for c in ["calorie_expenditure", "water_intake"]):
        df["calorie_water_ratio"] = df["calorie_expenditure"] / (df["water_intake"] * 1000 + 0.1)
    if all(c in df.columns for c in ["step_count", "calorie_expenditure"]):
        df["step_calorie_ratio"] = df["step_count"] / (df["calorie_expenditure"] + 0.1)
    stress_map = {"low": 0, "medium": 1, "high": 2}
    quality_map = {"poor": 0, "average": 1, "good": 2}
    activity_map = {"sedentary": 0, "moderate": 1, "active": 2}
    smoking_map = {"no": 0, "occasional": 1, "yes": 2}
    if "stress_level" in df.columns:
        df["stress_num"] = df["stress_level"].map(stress_map).astype(float)
    if "sleep_quality" in df.columns:
        df["quality_num"] = df["sleep_quality"].map(quality_map).astype(float)
    if "physical_activity_level" in df.columns:
        df["activity_num"] = df["physical_activity_level"].map(activity_map).astype(float)
    if "smoking_alcohol" in df.columns:
        df["smoking_num"] = df["smoking_alcohol"].map(smoking_map).astype(float)
    risk_cols = [c for c in ["stress_num", "quality_num", "smoking_num"] if c in df.columns]
    if risk_cols:
        df["risk_score"] = df[risk_cols].mean(axis=1)
    health_cols = [c for c in ["activity_num", "quality_num"] if c in df.columns]
    if health_cols:
        df["health_score"] = df[health_cols].mean(axis=1)
    return df


# Load
train = pd.read_csv(os.path.join(RAW, "train.csv"))
test = pd.read_csv(os.path.join(RAW, "test.csv"))
ids = test["id"].copy()
y = train[TARGET]
train_feat = train.drop(columns=["id", TARGET])
test_feat = test.drop(columns=["id"])

combined = pd.concat([train_feat, test_feat], axis=0, ignore_index=True)
combined = add_features(combined)
X = combined.iloc[:len(train_feat)].reset_index(drop=True)
X_test = combined.iloc[len(train_feat):].reset_index(drop=True)
y = y.reset_index(drop=True)

le = LabelEncoder()
y_enc = le.fit_transform(y)

# Split for val
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = joblib.load(os.path.join(MODELS, "preprocessor_v4.pkl"))
X_tr_proc = preprocessor.transform(X_tr)
X_val_proc = preprocessor.transform(X_val)
X_all_proc = preprocessor.transform(X)
X_test_proc = preprocessor.transform(X_test)

print(f"Features: {X_tr_proc.shape[1]}")

# Stacking ensemble
print("\nStacking Ensemble (LightGBM + RF + XGBoost -> LogisticRegression)...")
t0 = time.time()

estimators = [
    ("lgb", lgb.LGBMClassifier(
        n_estimators=3000, max_depth=7, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
        num_leaves=50, reg_alpha=0.5, reg_lambda=2.0,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)),
    ("rf", RandomForestClassifier(
        n_estimators=800, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1)),
    ("xgb", xgb.XGBClassifier(
        n_estimators=1200, max_depth=6, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
        gamma=0.5, reg_alpha=0.5, reg_lambda=2.0,
        objective="multi:softprob", eval_metric="mlogloss",
        random_state=42, n_jobs=-1, verbose=-1)),
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(
        max_iter=1000, multi_class="multinomial", class_weight="balanced"),
    cv=3, n_jobs=-1,
)
stack.fit(X_tr_proc, y_tr)
preds = stack.predict(X_val_proc)
ba = balanced_accuracy_score(y_val, preds)
print(f"Stacking Val Accuracy: {ba:.4f}")
print(classification_report(y_val, preds))
print(f"Time: {time.time()-t0:.0f}s")

# Retrain on full data
print("\nRetraining stacking on full data...")
stack_full = StackingClassifier(
    estimators=[
        ("lgb", lgb.LGBMClassifier(
            n_estimators=3000, max_depth=7, learning_rate=0.01,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
            num_leaves=50, reg_alpha=0.5, reg_lambda=2.0,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)),
        ("rf", RandomForestClassifier(
            n_estimators=800, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1)),
        ("xgb", xgb.XGBClassifier(
            n_estimators=1200, max_depth=6, learning_rate=0.02,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
            gamma=0.5, reg_alpha=0.5, reg_lambda=2.0,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=42, n_jobs=-1, verbose=-1)),
    ],
    final_estimator=LogisticRegression(
        max_iter=1000, multi_class="multinomial", class_weight="balanced"),
    cv=3, n_jobs=-1,
)
stack_full.fit(X_all_proc, y)
p = stack_full.predict(X_test_proc)
sub = pd.DataFrame({"id": ids, "health_condition": p})
sub.to_csv(os.path.join(SUBMISSIONS, "submission_v4_stacking.csv"), index=False)
print(f"submission_v4_stacking.csv saved ({len(sub)} rows)")
print(p)

# Save stacking model
joblib.dump(stack_full, os.path.join(MODELS, "model_stacking_v4.pkl"))
print("Stacking model saved")
