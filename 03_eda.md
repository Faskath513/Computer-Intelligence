# Phase 3 — Exploratory Data Analysis

Notebook: `notebooks/01_eda.ipynb`

Goal: understand the data well enough that every later modeling decision can
be justified by something you found here.

## Dataset Summary

- **Competition:** Kaggle Playground Series S6E7 — Predicting Student Health Risk
- **Training rows:** 690,088 | **Test rows:** 295,753
- **Target:** `health_condition` (at-risk / unhealthy / fit)
- **Metric:** Balanced accuracy

## Features

| Feature | Type | Missing % | Description |
|---|---|---|---|
| sleep_duration | numeric | ~11% | Hours of sleep per night |
| heart_rate | numeric | ~3.5% | Resting heart rate (bpm) |
| bmi | numeric | ~3.4% | Body mass index |
| calorie_expenditure | numeric | ~0% | Daily calories burned |
| step_count | numeric | ~0% | Daily step count |
| exercise_duration | numeric | ~0% | Minutes of exercise |
| water_intake | numeric | ~0% | Litres of water per day |
| diet_type | categorical | ~0% | veg / non-veg / balanced |
| stress_level | categorical | ~12% | high / medium / low |
| sleep_quality | categorical | ~8.5% | good / average / poor |
| physical_activity_level | categorical | ~0% | active / moderate / sedentary |
| smoking_alcohol | categorical | ~0% | yes / occasional / no |
| gender | categorical | ~0% | male / female / other |

## Key EDA Findings

1. **Severe class imbalance:** at-risk=85.9%, unhealthy=8.4%, fit=5.8%. Both minority classes below 20% — `class_weight='balanced'` is essential.
2. **Significant missing values:** stress_level (12%), sleep_duration (11%), sleep_quality (8.5%), heart_rate (3.5%), bmi (3.4%). Median/mode imputation in the preprocessing pipeline handles this.
3. **All numeric features have reasonable distributions** with no extreme outliers — StandardScaler is appropriate.
4. **All categorical features have low cardinality** (2-3 unique values each) — OneHotEncoder is appropriate.
5. **No extremely high pairwise correlations** among numeric features — all features retained.

## Deliverable checklist

- [x] Notebook runs top-to-bottom without errors (run `01_eda.ipynb` in Jupyter)
- [x] 5 concrete findings documented above
- [x] These findings drove feature engineering decisions in Phase 4

Next: `04_feature_engineering.md`
