"""Quick CV + submission for all 3 models"""
import os, numpy as np, pandas as pd, joblib, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# XGBoost CV
print("5-fold CV on XGBoost...")
cv_model = xgb.XGBClassifier(
    n_estimators=2000, max_depth=8, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    gamma=0.1, reg_alpha=0.1,
    objective="multi:softprob", eval_metric="mlogloss",
    random_state=42, n_jobs=-1
)
cv_scores = cross_val_score(cv_model, X_all_proc, y_enc, cv=skf, scoring="balanced_accuracy", n_jobs=-1)
print(f"XGBoost CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# LightGBM CV
print("\n5-fold CV on LightGBM...")
cv_lgb = lgb.LGBMClassifier(
    n_estimators=3000, max_depth=10, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8, num_leaves=63,
    class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
)
cv_scores_lgb = cross_val_score(cv_lgb, X_all_proc, y_enc, cv=skf, scoring="balanced_accuracy", n_jobs=-1)
print(f"LightGBM CV: {cv_scores_lgb.mean():.4f} +/- {cv_scores_lgb.std():.4f}")

# RF CV
print("\n5-fold CV on RandomForest...")
cv_rf = RandomForestClassifier(n_estimators=1000, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)
cv_scores_rf = cross_val_score(cv_rf, X_all_proc, y, cv=skf, scoring="balanced_accuracy", n_jobs=-1)
print(f"RF CV: {cv_scores_rf.mean():.4f} +/- {cv_scores_rf.std():.4f}")

# Retrain XGBoost on full data for best submission
print("\nRetraining XGBoost on FULL data...")
final_xgb = xgb.XGBClassifier(
    n_estimators=2000, max_depth=8, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    gamma=0.1, reg_alpha=0.1,
    objective="multi:softprob", eval_metric="mlogloss",
    random_state=42, n_jobs=-1
)
final_xgb.fit(X_all_proc, y_enc)
preds_enc = final_xgb.predict(X_test_proc)
preds = le.inverse_transform(preds_enc)

sub = pd.DataFrame({"id": ids, "health_condition": preds})
sub.to_csv(os.path.join(SUBMISSIONS, "submission_v3_xgb_full.csv"), index=False)
print(f"Submission saved! ({len(sub)} rows)")
print(sub["health_condition"].value_counts())

print("\n=== FINAL SUMMARY ===")
print(f"  XGBoost   val=0.9027  CV={cv_scores.mean():.4f}")
print(f"  LightGBM  val=0.8987  CV={cv_scores_lgb.mean():.4f}")
print(f"  RF        val=0.8762  CV={cv_scores_rf.mean():.4f}")
print(f"  WINNER: XGBoost")
