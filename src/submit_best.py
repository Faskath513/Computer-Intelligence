"""Retrain XGBoost on full data and generate submission"""
import os, numpy as np, pandas as pd, joblib, warnings, time
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

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

print(f"X_all_proc: {X_all_proc.shape}, X_test_proc: {X_test_proc.shape}")

# Retrain XGBoost on full training data
print("Retraining XGBoost on FULL training data...")
t0 = time.time()
final_xgb = xgb.XGBClassifier(
    n_estimators=2500, max_depth=8, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    gamma=0.1, reg_alpha=0.1,
    objective="multi:softprob", eval_metric="mlogloss",
    random_state=42, n_jobs=-1
)
final_xgb.fit(X_all_proc, y_enc)
print(f"Training done in {time.time()-t0:.0f}s")

# Generate predictions
preds_enc = final_xgb.predict(X_test_proc)
preds = le.inverse_transform(preds_enc)

sub = pd.DataFrame({"id": ids, "health_condition": preds})
out_path = os.path.join(SUBMISSIONS, "submission_v3_xgb_full.csv")
sub.to_csv(out_path, index=False)
print(f"Submission saved -> {out_path} ({len(sub)} rows)")
print(sub["health_condition"].value_counts())

# Also save the model
final_xgb.save_model(os.path.join(MODELS, "model_xgb_full.json"))
print("Model saved")
