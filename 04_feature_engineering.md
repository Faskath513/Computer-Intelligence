# Phase 4 — Feature Engineering & Preprocessing Pipeline

Files: `src/data_prep.py`, `src/features.py`, `src/train_v4.py`

## V1-V3 Pipeline (Original)

7 numeric features + 6 categorical features → StandardScaler + OneHotEncoder → 25 features.

## V4 Pipeline (Current — with Feature Engineering)

Original 13 features → **31 engineered features** → SimpleImputer only (no scaler for tree models) → 56 features.

### Engineered Features (31 new)

| Feature | Type | Logic |
|---|---|---|
| sleep_deprived | binary | sleep_duration < 6 |
| good_sleep | binary | sleep_duration >= 7 |
| oversleep | binary | sleep_duration > 8.5 |
| hr_low | binary | heart_rate < 60 |
| hr_normal | binary | 60 <= heart_rate <= 100 |
| hr_high | binary | heart_rate > 100 |
| bmi_underweight | binary | bmi < 18.5 |
| bmi_normal | binary | 18.5 <= bmi < 25 |
| bmi_overweight | binary | 25 <= bmi < 30 |
| bmi_obese | binary | bmi >= 30 |
| low_steps | binary | step_count < 5000 |
| high_steps | binary | step_count > 10000 |
| low_exercise | binary | exercise_duration < 20 |
| high_exercise | binary | exercise_duration > 60 |
| low_water | binary | water_intake < 1.5 |
| good_water | binary | water_intake > 2.5 |
| low_calorie | binary | calorie_expenditure < 1500 |
| high_calorie | binary | calorie_expenditure > 2800 |
| sleep_exercise_ratio | numeric | sleep_duration / (exercise/60 + 0.1) |
| sleep_x_exercise | numeric | sleep_duration * exercise_duration |
| bmi_x_exercise | numeric | bmi * exercise_duration |
| bmi_x_sleep | numeric | bmi * sleep_duration |
| hr_bmi_ratio | numeric | heart_rate / (bmi + 0.1) |
| calorie_water_ratio | numeric | calorie / (water*1000 + 0.1) |
| step_calorie_ratio | numeric | steps / (calorie + 0.1) |
| stress_num | numeric | stress_level mapped to 0/1/2 |
| quality_num | numeric | sleep_quality mapped to 0/1/2 |
| activity_num | numeric | physical_activity_level mapped to 0/1/2 |
| smoking_num | numeric | smoking_alcohol mapped to 0/1/2 |
| risk_score | numeric | mean of stress_num, quality_num, smoking_num |
| health_score | numeric | mean of activity_num, quality_num |

### Preprocessing Changes (V4)

- **No StandardScaler** — tree-based models (LightGBM, RF, XGBoost) don't need it
- **SimpleImputer only** — median for numerics, most_frequent for categoricals
- **OneHotEncoder** for categoricals (handle_unknown="ignore")

### Key Decisions from EDA

- **12% missing in stress_level**, 11% in sleep_duration → imputation essential
- **85.9% at-risk, 8.4% unhealthy, 5.8% fit** → class_weight="balanced" required
- **All categoricals low cardinality** (2-3 values) → OneHotEncoder appropriate
- **No high correlations** → all features retained

## Deliverable checklist

- [x] `data_prep.py` and `features.py` importable with no notebook dependency
- [x] Preprocessor fit only on training data, applied (not re-fit) to val/test
- [x] `models/preprocessor_v4.pkl` saved (V4 with feature engineering)
- [x] Feature engineering function (`add_features`) shared between training + app
- [x] 56 features after preprocessing (13 original + 31 engineered + OHE)

Next: `05_modeling.md`
