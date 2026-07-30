# Phase 3 — Exploratory Data Analysis

Script: `notebooks/01_eda_s5e1.py`

Goal: understand the data well enough that every later modeling decision can
be justified by something you found here.

## Dataset Summary

- **Competition:** Kaggle Playground Series S5E1 — Forecasting Sticker Sales
- **Training rows:** 230,130 | **Test rows:** 98,550
- **Target:** `num_sold` (continuous)
- **Metric:** Mean Absolute Percentage Error (MAPE)
- **90 time series:** 6 countries × 3 stores × 5 products

## Features

| Feature | Type | Description |
|---|---|---|
| date | datetime | Daily from 2010-01-01 to 2016-12-31 (train), 2017-2019 (test) |
| country | categorical | Canada, Finland, Italy, Kenya, Norway, Singapore |
| store | categorical | Discount Stickers, Stickers for Less, Premium Sticker Mart |
| product | categorical | Holographic Goose, Kaggle, Kaggle Tiers, Kerneler, Kerneler Dark Mode |
| num_sold | numeric | Target — daily sticker sales (5 to 5939) |

## Key EDA Findings

1. **90 balanced time series** — each of the 90 (country, store, product) combinations has 2557 rows
2. **Holographic Goose has no training sales** — all 8871 rows for this product have NaN num_sold, must be predicted from other series patterns
3. **Right-skewed distribution** — num_sold ranges 5-5939, mean 753, median 605, skew 1.42
4. **Clear seasonality** — sales peak in Dec-Jan and spring (Mar-May), lowest in Sep-Oct
5. **Huge variation by country** — Norway Premium Sticker Mart sells ~3198 Kaggle stickers avg, Kenya sells ~5-18 (extreme range requires MAPE-aware modeling)
6. **Downward trend** — average sales declining from 832 (2011) to 678 (2016)

## Deliverable checklist

- [x] EDA script runs top-to-bottom without errors
- [x] 6 concrete findings documented above
- [x] These findings drove feature engineering decisions in Phase 4

Next: `04_feature_engineering.md`
