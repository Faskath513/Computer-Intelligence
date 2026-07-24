import numpy as np
import pandas as pd
import joblib


def predict_sklearn(test_csv="data/raw/test.csv",
                    preprocessor_path="models/preprocessor.pkl",
                    model_path="models/model_final.pkl",
                    output_path="submissions/submission_v1.csv"):

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)

    test = pd.read_csv(test_csv)
    test_ids = test["id"]
    X_test = test.drop(columns=["id"])

    X_test_processed = preprocessor.transform(X_test)
    preds = model.predict(X_test_processed)

    submission = pd.DataFrame({
        "id": test_ids,
        "health_condition": preds,
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission saved → {output_path}")
    print(submission["health_condition"].value_counts())
    return submission


def predict_keras(test_csv="data/raw/test.csv",
                  preprocessor_path="models/preprocessor.pkl",
                  model_path="models/model_final.h5",
                  label_encoder_path="models/label_encoder.pkl",
                  output_path="submissions/submission_v1.csv"):
    import tensorflow as tf

    preprocessor = joblib.load(preprocessor_path)
    model = tf.keras.models.load_model(model_path)
    le = joblib.load(label_encoder_path)

    test = pd.read_csv(test_csv)
    test_ids = test["id"]
    X_test = test.drop(columns=["id"])

    X_test_processed = preprocessor.transform(X_test)
    raw_preds = model.predict(X_test_processed)
    preds_enc = np.argmax(raw_preds, axis=1)
    preds = le.inverse_transform(preds_enc)

    submission = pd.DataFrame({
        "id": test_ids,
        "health_condition": preds,
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission saved → {output_path}")
    print(submission["health_condition"].value_counts())
    return submission


if __name__ == "__main__":
    # Change this call to predict_keras if NN is the winner
    predict_sklearn()
