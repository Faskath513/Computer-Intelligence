import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "health_condition"
ID_COL = "id"


def load_data(raw_dir="data/raw"):
    train = pd.read_csv(f"{raw_dir}/train.csv")
    test = pd.read_csv(f"{raw_dir}/test.csv")
    return train, test


def get_feature_cols(df, target_col=TARGET_COL, id_col=ID_COL):
    drop = [c for c in [target_col, id_col] if c in df.columns]
    return df.drop(columns=drop)


def split_data(train, target_col=TARGET_COL, test_size=0.2, random_state=42):
    X = get_feature_cols(train, target_col=target_col)
    y = train[target_col]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def get_column_types(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    return numeric_cols, categorical_cols
