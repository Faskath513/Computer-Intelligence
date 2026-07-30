import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Student Health Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

NUM_COLS = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
            "step_count", "exercise_duration", "water_intake"]
CAT_COLS = ["diet_type", "stress_level", "sleep_quality",
            "physical_activity_level", "smoking_alcohol", "gender"]


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


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor_v4.pkl"))

    import lightgbm as lgb
    lgb_path = os.path.join(MODEL_DIR, "model_lgb_v4_app.txt")
    sklearn_path = os.path.join(MODEL_DIR, "model_rf_v3.pkl")
    if os.path.exists(lgb_path):
        model = lgb.Booster(model_file=lgb_path)
        model_type = "lgb_booster"
    elif os.path.exists(sklearn_path):
        model = joblib.load(sklearn_path)
        model_type = "sklearn"
    else:
        st.error("No trained model found in models/. Run training first.")
        st.stop()

    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    le = joblib.load(le_path) if os.path.exists(le_path) else None

    with open(os.path.join(MODEL_DIR, "feature_meta.json")) as f:
        feature_meta = json.load(f)

    return preprocessor, model, le, model_type, feature_meta


def predict(preprocessor, model, le, model_type, input_df):
    input_df = add_features(input_df)
    X = preprocessor.transform(input_df)

    if model_type == "lgb_booster":
        raw = model.predict(X)
        pred_enc = np.argmax(raw, axis=1)[0]
        proba = raw[0]
        if le is not None:
            pred = le.inverse_transform([pred_enc])[0]
            classes = le.classes_
        else:
            pred = str(pred_enc)
            classes = [str(i) for i in range(raw.shape[1])]
    elif model_type == "sklearn":
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
        classes = model.classes_
    else:
        raw = model.predict(X)
        if len(raw.shape) > 1 and raw.shape[1] > 1:
            pred_enc = np.argmax(raw, axis=1)[0]
            pred = le.inverse_transform([pred_enc])[0]
            proba = raw[0]
            classes = le.classes_
        else:
            pred = raw[0]
            proba = None
            classes = []

    return pred, proba, classes


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Student Health Risk Predictor")
st.caption("Playground Series S6E7 -- Predicting Student Health Risk")
st.write("Enter the student's details below to predict their health condition.")

preprocessor, model, le, model_type, feature_meta = load_artifacts()

with st.form("input_form"):
    st.subheader("Student Profile")
    inputs = {}

    col1, col2 = st.columns(2)
    numeric_features = feature_meta.get("numeric", [])
    categorical_features = feature_meta.get("categorical", {})

    for i, feat in enumerate(numeric_features):
        meta = feature_meta["numeric_meta"].get(feat, {})
        col = col1 if i % 2 == 0 else col2
        inputs[feat] = col.number_input(
            feat.replace("_", " ").title(),
            min_value=float(meta.get("min", 0)),
            max_value=float(meta.get("max", 999)),
            value=float(meta.get("mean", 0)),
            step=float(meta.get("step", 1.0)),
        )

    for feat, options in categorical_features.items():
        inputs[feat] = st.selectbox(feat.replace("_", " ").title(), options=options)

    submitted = st.form_submit_button("Predict Health Condition", type="primary")

if submitted:
    input_df = pd.DataFrame([inputs])
    pred, proba, classes = predict(preprocessor, model, le, model_type, input_df)

    colour_map = {
        "fit": "green",
        "unhealthy": "orange",
        "at-risk": "red",
    }
    colour = colour_map.get(pred, "blue")

    st.markdown("---")
    st.subheader("Prediction")
    st.markdown(
        f"<h2 style='color:{colour};'>{pred.upper()}</h2>",
        unsafe_allow_html=True,
    )

    if proba is not None:
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({
            "Class": classes,
            "Probability": [f"{p:.1%}" for p in proba],
        })
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

    with st.expander("About this model"):
        st.write("- **Model:** LightGBM (V4, regularized)")
        st.write("- **Features:** 44 original + engineered features")
        st.write("- **Metric:** Balanced accuracy (average per-class recall)")
        st.write("- **Val balanced accuracy:** 0.9337")
        st.write("- **Classes:** at-risk (85%), fit (6%), unhealthy (8%)")
        st.write("- **Techniques:** class_weight='balanced', feature engineering, regularization")
        st.write("- **Competition:** Kaggle Playground Series S6E7, July 2026")
