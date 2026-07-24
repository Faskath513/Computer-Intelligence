import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate_sklearn(model, X_val, y_val, model_name="Model"):
    preds = model.predict(X_val)
    bal_acc = balanced_accuracy_score(y_val, preds)
    print(f"\n── {model_name} ──")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(classification_report(y_val, preds))
    return bal_acc, preds


def evaluate_keras(nn, le, X_val, y_val_enc, model_name="Neural Net"):
    raw_preds = nn.predict(X_val)
    preds_enc = np.argmax(raw_preds, axis=1)
    bal_acc = balanced_accuracy_score(y_val_enc, preds_enc)
    print(f"\n── {model_name} ──")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(classification_report(y_val_enc, preds_enc, target_names=le.classes_))
    return bal_acc, preds_enc


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"models/{title.replace(' ', '_').lower()}.png", dpi=100)
    plt.show()


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("models/training_history.png", dpi=100)
    plt.show()
