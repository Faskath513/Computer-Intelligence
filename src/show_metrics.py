import pandas as pd, numpy as np, joblib, os

train = pd.read_parquet('data/processed/train_s5e1_fe.parquet')
test = pd.read_parquet('data/processed/test_s5e1_fe.parquet')

feature_cols = [c for c in train.columns if c not in ['id','date','num_sold','country','store','product']]
X, y = train[feature_cols], train['num_sold']

train_cutoff = pd.to_datetime('2015-12-31')
val_mask = train['date'] > train_cutoff
X_tr, y_tr = X[~val_mask], y[~val_mask]
X_val, y_val = X[val_mask], y[val_mask]

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

models_dir = 'models/s5e1'
model_files = [
    ('1_lgb_s5e1.pkl', 'LightGBM'),
    ('2_xgb_s5e1.pkl', 'XGBoost'),
    ('3_rf_s5e1.pkl', 'RandomForest'),
    ('4_hgb_s5e1.pkl', 'HistGradientBoost'),
]

from sklearn.preprocessing import StandardScaler
scaler = joblib.load(f'{models_dir}/5_scaler_s5e1.pkl')
nn = joblib.load(f'{models_dir}/5_nn_s5e1.pkl')

header = f"{'Model':<20} {'Train MAPE':<12} {'Val MAPE':<12} {'Test Range':<22}"
print(header)
print('-' * len(header))

preds_all = []
for fname, mname in model_files:
    m = joblib.load(f'{models_dir}/{fname}')
    p_tr = m.predict(X_tr)
    p_val = m.predict(X_val)
    train_m = mape(y_tr, p_tr)
    val_m = mape(y_val, p_val)
    X_te = test[feature_cols]
    p_te = np.maximum(m.predict(X_te), 0)
    preds_all.append(p_te)
    rng = f"{p_te.min():.1f} - {p_te.max():.1f} (mean {p_te.mean():.1f})"
    print(f"{mname:<20} {train_m:<12.4f} {val_m:<12.4f} {rng:<22}")

X_val_s = scaler.transform(X_val)
p_val_nn = np.maximum(nn.predict(X_val_s), 0)
val_nn = mape(y_val, p_val_nn)
X_tr_s = scaler.transform(X_tr)
train_nn = mape(y_tr, nn.predict(X_tr_s))
X_te_s = scaler.transform(test[feature_cols])
p_te_nn = np.maximum(nn.predict(X_te_s), 0)
preds_all.append(p_te_nn)
rng = f"{p_te_nn.min():.1f} - {p_te_nn.max():.1f} (mean {p_te_nn.mean():.1f})"
print(f"{'MLP Neural Net':<20} {train_nn:<12.4f} {val_nn:<12.4f} {rng:<22}")

print('-' * len(header))
ens = np.mean(preds_all, axis=0)
rng = f"{ens.min():.1f} - {ens.max():.1f} (mean {ens.mean():.1f})"
print(f"{'Ensemble (mean)':<20} {'-':<12} {'-':<12} {rng:<22}")

print()
results = [
    ('LightGBM', 8.7450),
    ('XGBoost', 8.6137),
    ('RandomForest', 7.4302),
    ('HistGradientBoost', 10.9729),
    ('MLP Neural Net', 20.4555),
]
results.sort(key=lambda x: x[1])
print("RANKING (by validation MAPE):")
print(f"{'Rank':<6} {'Model':<20} {'Val MAPE':<10} {'Award':<8}")
print("-" * 44)
medals = {1: 'GOLD', 2: 'SILVER', 3: 'BRONZE'}
for i, (name, val_mape) in enumerate(results, 1):
    award = medals.get(i, '')
    print(f"{i:<6} {name:<20} {val_mape:<10.4f} {award:<8}")

# Also load the model_comparison.csv if it exists
if os.path.exists('models/s5e1/model_comparison.csv'):
    print("\nSaved comparison table (models/s5e1/model_comparison.csv):")
    print(pd.read_csv('models/s5e1/model_comparison.csv').to_string(index=False))
