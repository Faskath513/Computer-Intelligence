import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(
    page_title="Student Health Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FEATURE_META_PATH = os.path.join(MODEL_DIR, "feature_meta.json")


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))

    # Try sklearn model first, fall back to Keras
    sklearn_path = os.path.join(MODEL_DIR, "model_final.pkl")
    keras_path   = os.path.join(MODEL_DIR, "model_final.h5")
    le_path      = os.path.join(MODEL_DIR, "label_encoder.pkl")

    if os.path.exists(sklearn_path):
        model = joblib.load(sklearn_path)
        le = None
        model_type = "sklearn"
    elif os.path.exists(keras_path):
        import tensorflow as tf
        model = tf.keras.models.load_model(keras_path)
        le = joblib.load(le_path)
        model_type = "keras"
    else:
        st.error("No trained model found in models/. Run the training notebooks first.")
        st.stop()

    with open(FEATURE_META_PATH) as f:
        feature_meta = json.load(f)

    return preprocessor, model, le, model_type, feature_meta


def predict(preprocessor, model, le, model_type, input_df):
    X = preprocessor.transform(input_df)
    if model_type == "sklearn":
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
        classes = model.classes_
    else:
        raw = model.predict(X)
        pred_enc = np.argmax(raw, axis=1)[0]
        pred = le.inverse_transform([pred_enc])[0]
        proba = raw[0]
        classes = le.classes_
    return pred, proba, classes


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🩺 Student Health Risk Predictor")
st.caption("Playground Series S6E7 — Predicting Student Health Risk")
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
        st.write(
            f"- **Model type:** {model_type} ({type(model).__name__ if model_type == 'sklearn' else 'Keras MLP'})"
        )
        st.write("- **Metric:** Balanced accuracy (average per-class recall)")
        st.write("- **Classes:** at-risk, unhealthy, fit")
        st.write("- **Training:** class_weight='balanced' to handle class imbalance")
        st.write("- **Competition:** Kaggle Playground Series S6E7, July 2026")
