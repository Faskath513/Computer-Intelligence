"""Generate submission CSVs for all 3 final models"""
import os, numpy as np, pandas as pd, joblib, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
MODELS = os.path.join(BASE, "models")
SUBMISSIONS = os.path.join(BASE, "submissions")

train = pd.read_csv(os.path.join(RAW, "train.csv"))
test = pd.read_csv(os.path.join(RAW, "test.csv"))
ids = test["id"].copy()
X = train.drop(columns=["id", "health_condition"])
y = train["health_condition"]
X_test = test.drop(columns=["id"])

le = LabelEncoder()
y_enc = le.fit_transform(y)

preprocessor = joblib.load(os.path.join(MODELS, "preprocessor_v3.pkl"))
X_all_proc = preprocessor.transform(X)
X_test_proc = preprocessor.transform(X_test)

# ── 1. Random Forest (retrain on full data for best submission) ──
print("Training RandomForest on full data...")
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=1000, min_samples_leaf=3,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_all_proc, y)
preds_rf = rf.predict(X_test_proc)
sub_rf = pd.DataFrame({"id": ids, "health_condition": preds_rf})
sub_rf.to_csv(os.path.join(SUBMISSIONS, "submission_v3_rf_final.csv"), index=False)
print(f"  RF submission saved ({len(sub_rf)} rows)")
print(f"  {preds_rf}")

# ── 2. XGBoost (retrain on full data) ──
print("\nTraining XGBoost on full data...")
xgb_m = xgb.XGBClassifier(
    n_estimators=2500, max_depth=8, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    gamma=0.1, reg_alpha=0.1,
    objective="multi:softprob", eval_metric="mlogloss",
    random_state=42, n_jobs=-1
)
xgb_m.fit(X_all_proc, y_enc)
preds_xgb_enc = xgb_m.predict(X_test_proc)
preds_xgb = le.inverse_transform(preds_xgb_enc)
sub_xgb = pd.DataFrame({"id": ids, "health_condition": preds_xgb})
sub_xgb.to_csv(os.path.join(SUBMISSIONS, "submission_v3_xgb_final.csv"), index=False)
print(f"  XGBoost submission saved ({len(sub_xgb)} rows)")
print(preds_xgb)

# ── 3. LightGBM (retrain on full data) ──
print("\nTraining LightGBM on full data...")
lgb_m = lgb.LGBMClassifier(
    n_estimators=3000, max_depth=10, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8, num_leaves=63,
    class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
)
lgb_m.fit(X_all_proc, y_enc)
preds_lgb_enc = lgb_m.predict(X_test_proc)
preds_lgb = le.inverse_transform(preds_lgb_enc)
sub_lgb = pd.DataFrame({"id": ids, "health_condition": preds_lgb})
sub_lgb.to_csv(os.path.join(SUBMISSIONS, "submission_v3_lgb_final.csv"), index=False)
print(f"  LightGBM submission saved ({len(sub_lgb)} rows)")
print(preds_lgb)

print("\n=== ALL 3 SUBMISSIONS READY ===")
for f in sorted(os.listdir(SUBMISSIONS)):
    if f.startswith("submission_v3") and f.endswith("final.csv"):
        fpath = os.path.join(SUBMISSIONS, f)
        size = os.path.getsize(fpath)
        print(f"  {f}  ({size/1024:.0f} KB)")
