# Phase 7 — App Development (Streamlit Web App)

File: `app/app.py`

## 1. Current App (V4)

The app loads:
- `models/preprocessor_v4.pkl` — V4 preprocessor (56 features)
- `models/model_lgb_v3.txt` — LightGBM V4 model
- `models/label_encoder.pkl` — label encoder
- `models/feature_meta.json` — feature metadata for input widgets

### Feature Engineering in App

The app applies the same `add_features()` function used in training:
- 31 engineered features computed from 13 original inputs
- BMI categories, sleep/exercise flags, interaction features, risk scores
- Categorical encodings (stress_num, quality_num, activity_num, smoking_num)

### How to Run

```bash
cd project-root
streamlit run app/app.py
```

### App Features

- **Input form** with 7 numeric inputs + 6 categorical inputs
- **Predict button** returns color-coded prediction (green=fit, orange=unhealthy, red=at-risk)
- **Class probabilities** shown for all 3 classes
- **About expander** with model info (LightGBM V4, val 0.9337)

## 2. App Architecture

```
User Input (13 features)
    ↓
add_features() → 31 new features (44 total raw)
    ↓
preprocessor_v4.pkl → SimpleImputer + OneHotEncoder (56 processed features)
    ↓
LightGBM Booster → 3-class prediction
    ↓
LabelEncoder → at-risk / fit / unhealthy
    ↓
Display with probabilities
```

## 3. `app/requirements.txt`

```
streamlit
pandas
numpy
scikit-learn
joblib
lightgbm
xgboost
```

## 4. Deliverable checklist

- [x] App loads saved model + preprocessor (not retrained on the fly)
- [x] Form/input covers every feature the pipeline expects
- [x] Feature engineering matches training pipeline exactly
- [x] Predict button returns sensible results for normal input
- [x] Runs from `pip install -r app/requirements.txt` + `streamlit run`
- [x] Color-coded predictions with class probabilities

Next: `08_integration_testing.md`
