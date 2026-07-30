import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time

train = pd.read_parquet('data/processed/train_s5e1_fe.parquet')
test = pd.read_parquet('data/processed/test_s5e1_fe.parquet')
sample = pd.read_csv('data/raw/sample_submission_s5e1.csv')

feature_cols = [c for c in train.columns if c not in ['id', 'date', 'num_sold',
    'country', 'store', 'product']]

X = train[feature_cols].copy()
y = train['num_sold'].copy()

train_cutoff = pd.to_datetime('2015-12-31')
val_mask = train['date'] > train_cutoff
X_train, y_train = X[~val_mask], y[~val_mask]
X_val, y_val = X[val_mask], y[val_mask]

print(f"Train: {len(X_train)} rows ({train[~val_mask]['date'].min().date()} to {train[~val_mask]['date'].max().date()})")
print(f"Val:   {len(X_val)} rows ({train[val_mask]['date'].min().date()} to {train[val_mask]['date'].max().date()})")

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

os.makedirs('models/s5e1', exist_ok=True)
results = []

print("=" * 60)
print("MODEL 1/5: LightGBM")
print("=" * 60)
start = time.time()
lgb_model = lgb.LGBMRegressor(
    objective='regression', metric='mape', boosting_type='gbdt',
    num_leaves=127, max_depth=12, learning_rate=0.01, n_estimators=3000,
    reg_alpha=1.0, reg_lambda=1.0, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.callback.early_stopping(100)])
pred_lgb = lgb_model.predict(X_val)
val_mape_lgb = mape(y_val, pred_lgb)
train_mape_lgb = mape(y_train, lgb_model.predict(X_train))
elapsed = time.time() - start
print(f"  Train MAPE: {train_mape_lgb:.4f}% | Val MAPE: {val_mape_lgb:.4f}% | Time: {elapsed:.1f}s")
results.append(('LightGBM', train_mape_lgb, val_mape_lgb, elapsed))
joblib.dump(lgb_model, 'models/s5e1/1_lgb_s5e1.pkl')

print("=" * 60)
print("MODEL 2/5: XGBoost")
print("=" * 60)
start = time.time()
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', eval_metric='mape',
    max_depth=8, learning_rate=0.01, n_estimators=3000,
    reg_alpha=1.0, reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
pred_xgb = xgb_model.predict(X_val)
val_mape_xgb = mape(y_val, pred_xgb)
train_mape_xgb = mape(y_train, xgb_model.predict(X_train))
elapsed = time.time() - start
print(f"  Train MAPE: {train_mape_xgb:.4f}% | Val MAPE: {val_mape_xgb:.4f}% | Time: {elapsed:.1f}s")
results.append(('XGBoost', train_mape_xgb, val_mape_xgb, elapsed))
joblib.dump(xgb_model, 'models/s5e1/2_xgb_s5e1.pkl')

print("=" * 60)
print("MODEL 3/5: Random Forest")
print("=" * 60)
start = time.time()
rf_model = RandomForestRegressor(
    n_estimators=500, max_depth=20, min_samples_leaf=5,
    min_samples_split=10, n_jobs=-1, random_state=42, verbose=0
)
rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_val)
val_mape_rf = mape(y_val, pred_rf)
train_mape_rf = mape(y_train, rf_model.predict(X_train))
elapsed = time.time() - start
print(f"  Train MAPE: {train_mape_rf:.4f}% | Val MAPE: {val_mape_rf:.4f}% | Time: {elapsed:.1f}s")
results.append(('RandomForest', train_mape_rf, val_mape_rf, elapsed))
joblib.dump(rf_model, 'models/s5e1/3_rf_s5e1.pkl')

print("=" * 60)
print("MODEL 4/5: HistGradientBoosting")
print("=" * 60)
start = time.time()
hgb_model = HistGradientBoostingRegressor(
    max_iter=1000, learning_rate=0.05, max_depth=8,
    min_samples_leaf=20, random_state=42, verbose=0
)
hgb_model.fit(X_train, y_train)
pred_hgb = hgb_model.predict(X_val)
val_mape_hgb = mape(y_val, pred_hgb)
train_mape_hgb = mape(y_train, hgb_model.predict(X_train))
elapsed = time.time() - start
print(f"  Train MAPE: {train_mape_hgb:.4f}% | Val MAPE: {val_mape_hgb:.4f}% | Time: {elapsed:.1f}s")
results.append(('HistGradientBoost', train_mape_hgb, val_mape_hgb, elapsed))
joblib.dump(hgb_model, 'models/s5e1/4_hgb_s5e1.pkl')

print("=" * 60)
print("MODEL 5/5: MLP Neural Network")
print("=" * 60)
start = time.time()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
nn_model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64), activation='relu',
    solver='adam', learning_rate='adaptive', learning_rate_init=0.001,
    max_iter=500, batch_size=2048, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=20,
    random_state=42, verbose=False
)
nn_model.fit(X_train_scaled, y_train)
pred_nn = nn_model.predict(X_val_scaled)
val_mape_nn = mape(y_val, pred_nn)
train_mape_nn = mape(y_train, nn_model.predict(X_train_scaled))
elapsed = time.time() - start
print(f"  Train MAPE: {train_mape_nn:.4f}% | Val MAPE: {val_mape_nn:.4f}% | Time: {elapsed:.1f}s")
results.append(('MLP Neural Net', train_mape_nn, val_mape_nn, elapsed))
joblib.dump(scaler, 'models/s5e1/5_scaler_s5e1.pkl')
joblib.dump(nn_model, 'models/s5e1/5_nn_s5e1.pkl')

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
results_df = pd.DataFrame(results, columns=['Model', 'Train MAPE (%)', 'Val MAPE (%)', 'Time (s)'])
results_df = results_df.sort_values('Val MAPE (%)').reset_index(drop=True)
results_df['Rank'] = range(1, len(results_df) + 1)
print(results_df.to_string(index=False))

best_idx = results_df['Val MAPE (%)'].idxmin()
best_name = results_df.loc[best_idx, 'Model']
best_mape = results_df.loc[best_idx, 'Val MAPE (%)']
print(f"\n*** BEST MODEL: {best_name} (Val MAPE: {best_mape:.4f}%) ***")

results_df.to_csv('models/s5e1/model_comparison.csv', index=False)

X_test = test[feature_cols].copy()
test_scaled = scaler.transform(X_test)

models_and_preds = [
    ('lgb', lgb_model, X_test, False),
    ('xgb', xgb_model, X_test, False),
    ('rf', rf_model, X_test, False),
    ('hgb', hgb_model, X_test, False),
    ('nn', nn_model, test_scaled, False),
]

actual_best_model = {
    'lgb': lgb_model, 'xgb': xgb_model, 'rf': rf_model,
    'hgb': hgb_model, 'nn': nn_model
}
best_model_key = {'LightGBM':'lgb','XGBoost':'xgb','RandomForest':'rf',
                  'HistGradientBoost':'hgb','MLP Neural Net':'nn'}[best_name]
best_model_obj = actual_best_model[best_model_key]

print("\nGenerating submission CSVs...")
ensemble_preds = []
for name_key, model_obj, x_data, is_scaled in models_and_preds:
    pred = model_obj.predict(x_data)
    pred = np.maximum(pred, 0)
    ensemble_preds.append(pred)
    sub = sample.copy()
    sub['num_sold'] = pred
    sub.to_csv(f'submissions/submission_s5e1_{name_key}.csv', index=False)
    print(f"  submissions/submission_s5e1_{name_key}.csv  |  min={pred.min():.1f}  max={pred.max():.1f}  mean={pred.mean():.1f}")

ensemble_pred = np.mean(ensemble_preds, axis=0)
sub = sample.copy()
sub['num_sold'] = ensemble_pred
sub.to_csv('submissions/submission_s5e1_ensemble.csv', index=False)
print(f"  submissions/submission_s5e1_ensemble.csv  |  min={ensemble_pred.min():.1f}  max={ensemble_pred.max():.1f}  mean={ensemble_pred.mean():.1f}")

best_pred = best_model_obj.predict(X_test if best_model_key != 'nn' else scaler.transform(X_test))
best_pred = np.maximum(best_pred, 0)
sub = sample.copy()
sub['num_sold'] = best_pred
sub.to_csv(f'submissions/submission_s5e1_best_{best_model_key}.csv', index=False)
print(f"  submissions/submission_s5e1_best_{best_model_key}.csv  |  BEST MODEL ({best_name})")

print("\n" + "=" * 60)
print("ALL DONE — 5 models + 1 ensemble + 1 best")
print("=" * 60)
