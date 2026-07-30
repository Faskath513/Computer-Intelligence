# Phase 7 — App Development (Streamlit Web App)

File: `app/app.py`

## 1. Current App (S5E1)

The app loads:
- `models/s5e1/1_lgb_s5e1.pkl` — LightGBM S5E1 model
- `data/raw/train_s5e1.csv` — training data for visualization
- `data/raw/test_s5e1.csv` — test data for predictions
- `data/processed/test_s5e1_fe.parquet` — precomputed features for prediction

### Features
- **Line chart** — total daily sales by country over time
- **Box plots** — sales distribution by country and product
- **Seasonality plot** — monthly average sales by country
- **Time series explorer** — filter by country, store, product to see individual series
- **Forecast overlay** — historical data + model predictions for selected series
- **Submission download** — one-click CSV download

### How to Run

```bash
cd project-root
streamlit run app/app.py
```

### App Architecture

```
Load train.csv + test.csv
    ↓
Visualizations: line charts, box plots, seasonality
    ↓
User selects country / store / product
    ↓
Load processed features → LightGBM model → predictions
    ↓
Overlay predictions on historical data
    ↓
Download submission CSV
```

## 2. `app/requirements.txt`

```
streamlit
pandas
numpy
scikit-learn
joblib
lightgbm
xgboost
plotly
```

## 3. Deliverable checklist

- [x] App loads saved model (not retrained on the fly)
- [x] Interactive time series explorer with filtering
- [x] Historical data visualizations (line, box, seasonality)
- [x] Forecast overlay on selected time series
- [x] One-click submission CSV download
- [x] Runs from `streamlit run app/app.py`

Next: `08_integration_testing.md`