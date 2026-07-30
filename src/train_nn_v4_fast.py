"""5th Model: Neural Network V4 — faster training"""
import os, numpy as np, pandas as pd, joblib, warnings, time
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
MODELS = os.path.join(BASE, "models")
SUBMISSIONS = os.path.join(BASE, "submissions")


def add_features(df):
    df = df.copy()
    if "sleep_duration" in df.columns:
        df["sleep_deprived"] = (df["sleep_duration"] < 6).astype(int); df["good_sleep"] = (df["sleep_duration"] >= 7).astype(int); df["oversleep"] = (df["sleep_duration"] > 8.5).astype(int)
    if "heart_rate" in df.columns:
        df["hr_low"] = (df["heart_rate"] < 60).astype(int); df["hr_normal"] = ((df["heart_rate"] >= 60) & (df["heart_rate"] <= 100)).astype(int); df["hr_high"] = (df["heart_rate"] > 100).astype(int)
    if "bmi" in df.columns:
        df["bmi_underweight"] = (df["bmi"] < 18.5).astype(int); df["bmi_normal"] = ((df["bmi"] >= 18.5) & (df["bmi"] < 25)).astype(int); df["bmi_overweight"] = ((df["bmi"] >= 25) & (df["bmi"] < 30)).astype(int); df["bmi_obese"] = (df["bmi"] >= 30).astype(int)
    if "step_count" in df.columns:
        df["low_steps"] = (df["step_count"] < 5000).astype(int); df["high_steps"] = (df["step_count"] > 10000).astype(int)
    if "exercise_duration" in df.columns:
        df["low_exercise"] = (df["exercise_duration"] < 20).astype(int); df["high_exercise"] = (df["exercise_duration"] > 60).astype(int)
    if "water_intake" in df.columns:
        df["low_water"] = (df["water_intake"] < 1.5).astype(int); df["good_water"] = (df["water_intake"] > 2.5).astype(int)
    if "calorie_expenditure" in df.columns:
        df["low_calorie"] = (df["calorie_expenditure"] < 1500).astype(int); df["high_calorie"] = (df["calorie_expenditure"] > 2800).astype(int)
    if all(c in df.columns for c in ["sleep_duration", "exercise_duration"]):
        df["sleep_exercise_ratio"] = df["sleep_duration"] / (df["exercise_duration"] / 60 + 0.1); df["sleep_x_exercise"] = df["sleep_duration"] * df["exercise_duration"]
    if all(c in df.columns for c in ["bmi", "exercise_duration"]):
        df["bmi_x_exercise"] = df["bmi"] * df["exercise_duration"]
    if all(c in df.columns for c in ["bmi", "sleep_duration"]):
        df["bmi_x_sleep"] = df["bmi"] * df["sleep_duration"]
    if all(c in df.columns for c in ["heart_rate", "bmi"]):
        df["hr_bmi_ratio"] = df["heart_rate"] / (df["bmi"] + 0.1)
    if all(c in df.columns for c in ["calorie_expenditure", "water_intake"]):
        df["calorie_water_ratio"] = df["calorie_expenditure"] / (df["water_intake"] * 1000 + 0.1)
    if all(c in df.columns for c in ["step_count", "calorie_expenditure"]):
        df["step_calorie_ratio"] = df["step_count"] / (df["calorie_expenditure"] + 0.1)
    stress_map = {"low": 0, "medium": 1, "high": 2}; quality_map = {"poor": 0, "average": 1, "good": 2}; activity_map = {"sedentary": 0, "moderate": 1, "active": 2}; smoking_map = {"no": 0, "occasional": 1, "yes": 2}
    if "stress_level" in df.columns: df["stress_num"] = df["stress_level"].map(stress_map).astype(float)
    if "sleep_quality" in df.columns: df["quality_num"] = df["sleep_quality"].map(quality_map).astype(float)
    if "physical_activity_level" in df.columns: df["activity_num"] = df["physical_activity_level"].map(activity_map).astype(float)
    if "smoking_alcohol" in df.columns: df["smoking_num"] = df["smoking_alcohol"].map(smoking_map).astype(float)
    risk_cols = [c for c in ["stress_num", "quality_num", "smoking_num"] if c in df.columns]
    if risk_cols: df["risk_score"] = df[risk_cols].mean(axis=1)
    health_cols = [c for c in ["activity_num", "quality_num"] if c in df.columns]
    if health_cols: df["health_score"] = df[health_cols].mean(axis=1)
    return df


train = pd.read_csv(os.path.join(RAW, "train.csv"))
test = pd.read_csv(os.path.join(RAW, "test.csv"))
ids = test["id"].copy()
y = train["health_condition"]
train_feat = train.drop(columns=["id", "health_condition"]); test_feat = test.drop(columns=["id"])
combined = pd.concat([train_feat, test_feat], axis=0, ignore_index=True)
combined = add_features(combined)
X = combined.iloc[:len(train_feat)].reset_index(drop=True); X_test = combined.iloc[len(train_feat):].reset_index(drop=True)
y = y.reset_index(drop=True)

le = LabelEncoder(); y_enc = le.fit_transform(y); n_classes = len(le.classes_)
preprocessor = joblib.load(os.path.join(MODELS, "preprocessor_v4.pkl"))
X_proc = preprocessor.transform(X); X_test_proc = preprocessor.transform(X_test)
scaler = StandardScaler(); X_s = scaler.fit_transform(X_proc); X_test_s = scaler.transform(X_test_proc)

from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X_s, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(n_classes), y=y_tr)))

tf.random.set_seed(42)
nn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_tr.shape[1],)),
    tf.keras.layers.Dense(192, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(96, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(n_classes, activation="softmax"),
])
nn.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Training NN V4 (50 epochs max)...")
t0 = time.time()
history = nn.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=50, batch_size=2048,
                 class_weight=cw, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                 verbose=1)
print(f"Training time: {time.time()-t0:.0f}s, epochs: {len(history.history['loss'])}")

from sklearn.metrics import balanced_accuracy_score
preds_val = np.argmax(nn.predict(X_val, verbose=0), axis=1)
ba = balanced_accuracy_score(y_val, preds_val)
print(f"NN V4 Balanced Accuracy: {ba:.4f}")

nn.save(os.path.join(MODELS, "model_nn_v4.h5"))
joblib.dump(scaler, os.path.join(MODELS, "scaler_v4.pkl"))
preds_test = np.argmax(nn.predict(X_test_s, verbose=0), axis=1)
preds = le.inverse_transform(preds_test)
pd.DataFrame({"id": ids, "health_condition": preds}).to_csv(os.path.join(SUBMISSIONS, "submission_v4_nn.csv"), index=False)
print(f"submission_v4_nn.csv saved ({len(preds)} rows)")
