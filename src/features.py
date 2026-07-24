from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ], remainder="drop")

    return preprocessor


def fit_and_save_preprocessor(X_train, numeric_cols, categorical_cols,
                               save_path="models/preprocessor.pkl"):
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, save_path)
    print(f"Preprocessor saved -> {save_path}")
    return preprocessor


def load_preprocessor(path="models/preprocessor.pkl"):
    return joblib.load(path)
