import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os

train = pd.read_parquet('data/processed/train_s5e1_fe.parquet')
test = pd.read_parquet('data/processed/test_s5e1_fe.parquet')
sample = pd.read_csv('data/raw/sample_submission_s5e1.csv')

feature_cols = [c for c in train.columns if c not in ['id', 'date', 'num_sold',
    'country', 'store', 'product']]
categoricals = ['country_encoded', 'store_encoded', 'product_encoded']

X = train[feature_cols].copy()
y = train['num_sold'].copy()

train_cutoff = pd.to_datetime('2015-12-31')
val_mask = train['date'] > train_cutoff
X_train, y_train = X[~val_mask], y[~val_mask]
X_val, y_val = X[val_mask], y[val_mask]

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
print(f"Train date range: {train['date'].min()} to {train['date'].max()}")
print(f"Val date range: {train[val_mask]['date'].min()} to {train[val_mask]['date'].max()}")

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

models = {}

print("\n=== LIGHTGBM ===")
lgb_params = {
    'objective': 'regression',
    'metric': 'mape',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'max_depth': 12,
    'learning_rate': 0.01,
    'n_estimators': 3000,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.callback.early_stopping(100), lgb.callback.log_evaluation(100)]
)
y_pred_lgb = lgb_model.predict(X_val)
val_mape = mape(y_val, y_pred_lgb)
print(f"LightGBM Val MAPE: {val_mape:.4f}%")
models['lgb'] = (lgb_model, val_mape)

print("\n=== XGBOOST ===")
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mape',
    'max_depth': 8,
    'learning_rate': 0.01,
    'n_estimators': 3000,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'auto'
}
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)
y_pred_xgb = xgb_model.predict(X_val)
val_mape_xgb = mape(y_val, y_pred_xgb)
print(f"XGBoost Val MAPE: {val_mape_xgb:.4f}%")
models['xgb'] = (xgb_model, val_mape_xgb)

best_model_name = min(models, key=lambda k: models[k][1])
best_model = models[best_model_name][0]
best_mape = models[best_model_name][1]
print(f"\n=== BEST: {best_model_name} (Val MAPE: {best_mape:.4f}%) ===")

os.makedirs('models/s5e1', exist_ok=True)
joblib.dump(best_model, f'models/s5e1/{best_model_name}_s5e1.pkl')
print(f"Saved model to models/s5e1/{best_model_name}_s5e1.pkl")

print("\n=== PREDICTING TEST ===")
X_test = test[feature_cols].copy()
test_pred = best_model.predict(X_test)
test_pred = np.maximum(test_pred, 0)

sub = sample.copy()
sub['num_sold'] = test_pred
sub.to_csv('submissions/submission_s5e1.csv', index=False)
print(f"Submission saved: submissions/submission_s5e1.csv")
print(f"Predictions: min={test_pred.min():.2f}, max={test_pred.max():.2f}, mean={test_pred.mean():.2f}")
print(f"Sample submission head:\n{sub.head()}")
