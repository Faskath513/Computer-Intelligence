"""Retrain LightGBM V4 on full data and save for the app"""
import os, numpy as np, pandas as pd, joblib, warnings, time
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
MODELS = os.path.join(BASE, "models")


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


train = pd.read_csv(os.path.join(RAW, "train.csv"))
y = train["health_condition"]
train_feat = train.drop(columns=["id", "health_condition"])

X = add_features(train_feat)
le = LabelEncoder()
y_enc = le.fit_transform(y)

preprocessor = joblib.load(os.path.join(MODELS, "preprocessor_v4.pkl"))
X_proc = preprocessor.transform(X)

print(f"X_proc shape: {X_proc.shape}")  # Should be (690088, 56)

m = lgb.LGBMClassifier(
    n_estimators=5000, max_depth=6, learning_rate=0.005,
    subsample=0.65, colsample_bytree=0.65, min_child_weight=15,
    num_leaves=40, reg_alpha=1.0, reg_lambda=3.0,
    class_weight="balanced", min_gain_to_split=0.2, max_bin=200,
    random_state=42, n_jobs=-1, verbose=-1,
)
m.fit(X_proc, y_enc)
print(f"Trained. Num iterations: {m.n_estimators_}")

m.booster_.save_model(os.path.join(MODELS, "model_lgb_v4_app.txt"))
joblib.dump(le, os.path.join(MODELS, "label_encoder.pkl"))
print("Saved model_lgb_v4_app.txt and label_encoder.pkl")

# Verify
booster = lgb.Booster(model_file=os.path.join(MODELS, "model_lgb_v4_app.txt"))
pred = booster.predict(X_proc[:1])
print(f"Verify: input={X_proc[:1].shape}, output={pred.shape}, pred={pred}")
