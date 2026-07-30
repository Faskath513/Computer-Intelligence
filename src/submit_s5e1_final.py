import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
import os

train = pd.read_parquet('data/processed/train_s5e1_fe.parquet')
test = pd.read_parquet('data/processed/test_s5e1_fe.parquet')
sample = pd.read_csv('data/raw/sample_submission_s5e1.csv')

feature_cols = [c for c in train.columns if c not in ['id', 'date', 'num_sold',
    'country', 'store', 'product']]

X = train[feature_cols].copy()
y = train['num_sold'].copy()

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

os.makedirs('models/s5e1', exist_ok=True)

print("=== LIGHTGBM (full data) ===")
lgb_model = lgb.LGBMRegressor(
    objective='regression',
    metric='mape',
    boosting_type='gbdt',
    num_leaves=127,
    max_depth=12,
    learning_rate=0.01,
    n_estimators=2000,
    reg_alpha=1.0,
    reg_lambda=1.0,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X, y)
joblib.dump(lgb_model, 'models/s5e1/lgb_s5e1_full.pkl')
pred_lgb = lgb_model.predict(test[feature_cols])
pred_lgb = np.maximum(pred_lgb, 0)

print(f"  MAPE on train: {mape(y, lgb_model.predict(X)):.4f}%")

print("=== XGBoost (full data) ===")
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='mape',
    max_depth=8,
    learning_rate=0.01,
    n_estimators=2000,
    reg_alpha=1.0,
    reg_lambda=2.0,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X, y, verbose=False)
joblib.dump(xgb_model, 'models/s5e1/xgb_s5e1_full.pkl')
pred_xgb = xgb_model.predict(test[feature_cols])
pred_xgb = np.maximum(pred_xgb, 0)

print(f"  MAPE on train: {mape(y, xgb_model.predict(X)):.4f}%")

print("=== ENSEMBLE (avg of LGB + XGB) ===")
pred_ensemble = (pred_lgb + pred_xgb) / 2

for name, pred in [('lgb', pred_lgb), ('xgb', pred_xgb), ('ensemble', pred_ensemble)]:
    sub = sample.copy()
    sub['num_sold'] = pred
    sub.to_csv(f'submissions/submission_s5e1_{name}.csv', index=False)
    print(f"  Saved submissions/submission_s5e1_{name}.csv")
    print(f"  Range: {pred.min():.1f} - {pred.max():.1f}, Mean: {pred.mean():.1f}")
