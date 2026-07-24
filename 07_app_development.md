# Phase 7 — App Development (Streamlit Web App)

File: `app/app.py`

## 1. Minimum viable app

```python
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="<Your Model> Predictor", layout="centered")

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("models/preprocessor.pkl")
    model = joblib.load("models/model_final.pkl")   # or load_model(...) for Keras
    return preprocessor, model

preprocessor, model = load_artifacts()

st.title("<Competition Name> — Prediction App")
st.write("Enter the feature values below to get a prediction from the trained model.")

# --- Build input form: one widget per feature used at training time ---
with st.form("input_form"):
    feature_1 = st.number_input("Feature 1", value=0.0)
    feature_2 = st.selectbox("Feature 2 (category)", options=["A", "B", "C"])
    # ... repeat for every feature the model expects
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "feature_1": feature_1,
        "feature_2": feature_2,
        # match all columns the preprocessor expects, in the right names
    }])

    X_processed = preprocessor.transform(input_df)
    prediction = model.predict(X_processed)

    st.subheader("Prediction")
    st.write(prediction[0])

    # Optional, if classifier supports it:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_processed)
        st.write("Confidence:", proba.max())
```

## 2. Nice-to-haves (add if time allows, in this order of value)

- [ ] Show where the input sits relative to the training distribution (a small
      histogram with a marker line) — makes the "practical demonstration" land better
- [ ] Batch mode: let the user upload a CSV of multiple rows instead of one manual entry
- [ ] Basic input validation (ranges, required fields) with `st.error(...)` messages
- [ ] A short "About this model" expander summarizing which technique is used and why

## 3. `app/requirements.txt`

```
streamlit
pandas
scikit-learn
joblib
tensorflow   # only if using a Keras model
```

## 4. Run it

```bash
cd project-root
streamlit run app/app.py
```

## Alternative: FastAPI (only if you specifically want an "API service" deliverable)

```python
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/model_final.pkl")

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    X = preprocessor.transform(df)
    pred = model.predict(X)
    return {"prediction": pred.tolist()[0]}

@app.get("/health")
def health():
    return {"status": "ok"}
```
```bash
uvicorn app.app:app --reload
```

## Deliverable checklist

- [ ] App loads the saved model + preprocessor (not retrained on the fly)
- [ ] Form/input covers every feature the pipeline expects
- [ ] Predict button returns a sensible result for normal input
- [ ] Runs from a clean `pip install -r app/requirements.txt` + `streamlit run`

Next: `08_integration_testing.md`
