# Phase 4 — Feature Engineering & Preprocessing Pipeline

Script: `src/features_s5e1.py`

## Global Forecasting Approach

Train a single model across all 90 time series simultaneously. This allows
the model to learn patterns from high-volume series and apply them to
low-volume series (especially Holographic Goose which has no training data).

## Features Engineered (41 total)

### Date Features (10)
- year, month, dayofyear, quarter, dayofweek, weekend
- sin_month, cos_month (cyclic encoding)
- sin_dayofyear, cos_dayofyear (cyclic encoding)

### Trend Features (2)
- days_since_start, elapsed_years

### Yearly Lag Features (3)
- lag_yearly_365, lag_yearly_730, lag_yearly_1095
- Always point to training data even for test rows

### Series-Level Statistics (5)
- overall_mean, overall_std, overall_median, overall_min, overall_max
- Computed per (country, store, product) from all training data

### Monthly Historical Averages (3)
- monthly_mean, monthly_std, monthly_median
- Computed per (country, store, product, month)

### Recent Statistics (5)
- recent_mean, recent_std, recent_median (from the last training year)
- last90_mean, last90_std, last90_median, last90_min, last90_max

### Year-over-Year Features (2)
- yearly_mean, yearly_change (from aggregated yearly data)

### Categorical Encodings (3)
- country_encoded, store_encoded, product_encoded (LabelEncoder)

## Preprocessing

- **No scaling** needed — tree-based models handle raw values natively
- **NaN handling** — Holographic Goose series has NaN targets; fill missing lags/stats with training median
- **Rows with NaN target** (Holographic Goose training data) dropped before training

## Key Decisions from EDA

- **Wide value range (5 to 5939)** → model needs to learn multiplicative patterns, not additive
- **Holographic Goose cold start** → global model + target encodings let it borrow from other products
- **Yearly lags (365/730/1095)** → capture same-day-last-year patterns without leaking future data
- **Monthly stats** → capture intra-year seasonality robustly

## Deliverable checklist

- [x] `features_s5e1.py` importable with no notebook dependency
- [x] Feature engineering applied identically to train + test
- [x] Processed data saved to `data/processed/train_s5e1_fe.parquet`
- [x] All NaN values handled (filled with training median)
- [x] 41 features after engineering

Next: `05_modeling.md`