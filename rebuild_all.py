import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns', 50)

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'data', 'raw')
MODELS = os.path.join(BASE, 'models')
PROCESSED = os.path.join(BASE, 'data', 'processed')
SUBMISSIONS = os.path.join(BASE, 'submissions')
os.makedirs(MODELS, exist_ok=True)
os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(SUBMISSIONS, exist_ok=True)

# ============================================================
# PHASE 3: EDA
# ============================================================
print("=" * 60)
print("PHASE 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

train = pd.read_csv(os.path.join(RAW, 'train.csv'))
test = pd.read_csv(os.path.join(RAW, 'test.csv'))
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print()

# Drop id
id_test = test['id'].copy()

# Target distribution
print("=== TARGET DISTRIBUTION ===")
print(train['health_condition'].value_counts())
print()
print(train['health_condition'].value_counts(normalize=True).round(4))
print()

fig, ax = plt.subplots(figsize=(6, 4))
train['health_condition'].value_counts().plot(kind='bar', ax=ax, color=['#e74c3c', '#f39c12', '#2ecc71'])
ax.set_title('Target Distribution — health_condition')
ax.set_xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(MODELS, 'target_distribution.png'), dpi=100)
plt.close()
print("Saved target_distribution.png")

# Missing values
print("\n=== MISSING VALUES ===")
missing = train.drop(columns=['id', 'health_condition']).isnull().sum()
missing_pct = (missing / len(train) * 100).round(2).sort_values(ascending=False)
print(missing_pct)
print()

# Numeric stats
num_cols = train.select_dtypes(include='number').columns.drop('id').tolist()
print(f"Numeric columns ({len(num_cols)}): {num_cols}")
print(train[num_cols].describe().round(3))
print()

# Histograms
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    train[col].hist(bins=50, ax=axes[i], color='steelblue', edgecolor='white')
    axes[i].set_title(col)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Numeric Feature Distributions')
plt.tight_layout()
plt.savefig(os.path.join(MODELS, 'numeric_histograms.png'), dpi=100)
plt.close()
print("Saved numeric_histograms.png")

# Correlation heatmap
corr = train[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', center=0, annot=True, fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(MODELS, 'correlation_heatmap.png'), dpi=100)
plt.close()
print("Saved correlation_heatmap.png")

# Find high correlations
high_corr = []
for i in range(len(num_cols)):
    for j in range(i + 1, len(num_cols)):
        if abs(corr.iloc[i, j]) > 0.7:
            high_corr.append((num_cols[i], num_cols[j], corr.iloc[i, j]))
if high_corr:
    print("\nHigh correlations (>0.7):")
    for a, b, v in high_corr:
        print(f"  {a} <-> {b}: {v:.3f}")
else:
    print("\nNo correlations > 0.7 found")

# Boxplots by target
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    train.boxplot(column=col, by='health_condition', ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel('')
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Numeric Features by Target Class')
plt.tight_layout()
plt.savefig(os.path.join(MODELS, 'features_vs_target.png'), dpi=100)
plt.close()
print("Saved features_vs_target.png")

# Categorical features
cat_cols = train.select_dtypes(include='object').columns.drop('health_condition').tolist()
print(f"\nCategorical columns ({len(cat_cols)}): {cat_cols}")
for c in cat_cols:
    print(f"  {c}: {train[c].nunique()} unique -> {train[c].unique()[:10]}")

# Outlier detection
Q1 = train[num_cols].quantile(0.25)
Q3 = train[num_cols].quantile(0.75)
IQR = Q3 - Q1
outlier_count = ((train[num_cols] < (Q1 - 1.5 * IQR)) | (train[num_cols] > (Q3 + 1.5 * IQR))).sum()
print("\nOutlier counts:")
print(outlier_count.sort_values(ascending=False))

# ============================================================
# PHASE 4: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: FEATURE ENGINEERING")
print("=" * 60)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import json

feature_cols = [c for c in train.columns if c not in ['id', 'health_condition']]
numeric_features = train[feature_cols].select_dtypes(include='number').columns.tolist()
categorical_features = train[feature_cols].select_dtypes(include='object').columns.tolist()

print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# Build preprocessor
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# Fit on training data
X_train_full = train[feature_cols]
y_train_full = train['health_condition']

preprocessor.fit(X_train_full)
print("Preprocessor fitted on training data")

# Save preprocessor
joblib.dump(preprocessor, os.path.join(MODELS, 'preprocessor.pkl'))
print("Saved preprocessor.pkl")

# Build feature_meta.json for the app
feature_meta = {
    "numeric": numeric_features,
    "categorical": {},
    "numeric_meta": {}
}

for feat in numeric_features:
    feature_meta["numeric_meta"][feat] = {
        "min": float(train[feat].min()),
        "max": float(train[feat].max()),
        "mean": float(train[feat].mean()),
        "step": float((train[feat].max() - train[feat].min()) / 100)
    }

for feat in categorical_features:
    options = train[feat].dropna().unique().tolist()
    feature_meta["categorical"][feat] = options

with open(os.path.join(MODELS, 'feature_meta.json'), 'w') as f:
    json.dump(feature_meta, f, indent=2)
print("Saved feature_meta.json")

# Transform for modeling
X_proc = preprocessor.transform(X_train_full)
# Get feature names after transformation
cat_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features).tolist()
all_feature_names = numeric_features + cat_feature_names
print(f"Processed feature count: {X_proc.shape[1]}")

# ============================================================
# PHASE 5: MODELING
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: MODEL TRAINING & EVALUATION")
print("=" * 60)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_proc, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)
print(f"Train: {X_tr.shape}, Val: {X_val.shape}")

# Compute class weights
classes = np.unique(y_train_full)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train_full)
weight_dict = dict(zip(classes, class_weights))
print(f"Class weights: {weight_dict}")

results = []

# --- Model 1: Logistic Regression ---
print("\n--- Logistic Regression ---")
lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
lr.fit(X_tr, y_tr)
lr_preds = lr.predict(X_val)
lr_ba = balanced_accuracy_score(y_val, lr_preds)
print(f"Balanced Accuracy: {lr_ba:.4f}")
print(classification_report(y_val, lr_preds))
results.append({'Model': 'LogisticRegression (balanced)', 'Val balanced_acc': lr_ba, 'CV mean': '-', 'CV std': '-', 'Notes': 'Baseline'})

# Save LR
joblib.dump(lr, os.path.join(MODELS, 'model_lr.pkl'))

# --- Model 2: Random Forest ---
print("--- Random Forest ---")
rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=5,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_tr, y_tr)
rf_preds = rf.predict(X_val)
rf_ba = balanced_accuracy_score(y_val, rf_preds)
print(f"Balanced Accuracy: {rf_ba:.4f}")
print(classification_report(y_val, rf_preds))

# CV
print("Running 3-fold CV...")
rf_cv = cross_val_score(rf, X_proc, y_train_full, cv=3, scoring='balanced_accuracy', n_jobs=-1)
print(f"CV: {rf_cv.mean():.4f} +/- {rf_cv.std():.4f}")
results.append({'Model': 'RandomForest (tuned, balanced)', 'Val balanced_acc': rf_ba, 'CV mean': f'{rf_cv.mean():.4f}', 'CV std': f'{rf_cv.std():.4f}', 'Notes': 'n_estimators=300, min_samples_leaf=5'})

# Feature importance
feat_imp = pd.Series(rf.feature_importances_, index=all_feature_names).sort_values(ascending=False)
print("\nTop 10 features:")
print(feat_imp.head(10))

joblib.dump(rf, os.path.join(MODELS, 'model_rf.pkl'))
print("Saved model_rf.pkl")

# --- Model 3: Neural Network ---
print("\n--- Neural Network (MLP) ---")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_tr_enc = le.fit_transform(y_tr)
y_val_enc = le.transform(y_val)
y_full_enc = le.fit_transform(y_train_full)

input_dim = X_tr.shape[1]
n_classes = len(le.classes_)

nn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])

nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Class weights for Keras
class_weight_arr = compute_class_weight('balanced', classes=np.arange(n_classes), y=y_tr_enc)
class_weight_keras = dict(enumerate(class_weight_arr))

early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')

history = nn_model.fit(
    X_tr, y_tr_enc,
    validation_data=(X_val, y_val_enc),
    epochs=50,
    batch_size=512,
    class_weight=class_weight_keras,
    callbacks=[early_stop],
    verbose=1
)

nn_val_preds = np.argmax(nn_model.predict(X_val, verbose=0), axis=1)
nn_ba = balanced_accuracy_score(y_val_enc, nn_val_preds)
print(f"Balanced Accuracy: {nn_ba:.4f}")
print(classification_report(y_val_enc, nn_val_preds, target_names=le.classes_))
n_epochs = len(history.history['loss'])
results.append({'Model': 'Neural Net (MLP, class-weighted)', 'Val balanced_acc': nn_ba, 'CV mean': '-', 'CV std': '-', 'Notes': f'{n_epochs} epochs, early stopping'})

# Save NN
nn_model.save(os.path.join(MODELS, 'model_nn.h5'))
joblib.dump(le, os.path.join(MODELS, 'label_encoder.pkl'))
print("Saved model_nn.h5 and label_encoder.pkl")

# ============================================================
# COMPARE & PICK WINNER
# ============================================================
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results)
print(results_df.to_markdown(index=False))

# Pick winner
best_idx = results_df['Val balanced_acc'].idxmax()
winner = results_df.loc[best_idx, 'Model']
print(f"\nWINNER: {winner}")

# Save experiments.md
with open(os.path.join(BASE, 'experiments.md'), 'w') as f:
    f.write("# Model Comparison -- Playground Series S6E7\n\n")
    f.write("| Model | Val balanced accuracy | CV mean +/- std | Notes |\n")
    f.write("|---|---|---|---|\n")
    for _, row in results_df.iterrows():
        cv = f"{row['CV mean']} +/- {row['CV std']}" if row['CV mean'] != '-' else '-'
        bold = '**' if row.name == best_idx else ''
        f.write(f"| {row['Model']} | {bold}{row['Val balanced_acc']:.4f}{bold} | {cv} | {row['Notes']} |\n")
    f.write(f"\n> **Winner: {winner}**\n")
print("Saved experiments.md")

# ============================================================
# GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("GENERATING KAGGLE SUBMISSION")
print("=" * 60)

# Use the winning model for submission
if 'RandomForest' in winner:
    model = joblib.load(os.path.join(MODELS, 'model_rf.pkl'))
    X_test_proc = preprocessor.transform(test[feature_cols])
    preds = model.predict(X_test_proc)
    model_type = 'sklearn'
elif 'Logistic' in winner:
    model = joblib.load(os.path.join(MODELS, 'model_lr.pkl'))
    X_test_proc = preprocessor.transform(test[feature_cols])
    preds = model.predict(X_test_proc)
    model_type = 'sklearn'
else:
    model = tf.keras.models.load_model(os.path.join(MODELS, 'model_nn.h5'))
    le_sub = joblib.load(os.path.join(MODELS, 'label_encoder.pkl'))
    X_test_proc = preprocessor.transform(test[feature_cols])
    preds_enc = np.argmax(model.predict(X_test_proc, verbose=0), axis=1)
    preds = le_sub.inverse_transform(preds_enc)
    model_type = 'keras'

submission = pd.DataFrame({
    'id': id_test,
    'health_condition': preds
})
submission.to_csv(os.path.join(SUBMISSIONS, 'submission_v2_rf_final.csv'), index=False)
print(f"Submission saved ({len(submission)} rows)")
print(submission['health_condition'].value_counts())

# Also save model_final
if model_type == 'sklearn':
    joblib.dump(model, os.path.join(MODELS, 'model_final.pkl'))
else:
    model.save(os.path.join(MODELS, 'model_final.h5'))

print("\n" + "=" * 60)
print("ALL PHASES COMPLETE!")
print("=" * 60)
