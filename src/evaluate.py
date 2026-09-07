"""
evaluate.py
------------
Loads the trained model and evaluates it on the held-out TEST set
(data never seen during training or validation) using:
Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.

Saves the confusion matrix as an image in outputs/plots/.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import config
from data_loader import build_dataset


def load_test_data():
    """
    Loads the test split (saved earlier by data_loader.py's split_data())
    so we evaluate on the exact same held-out images every time.
    """
    test_df = pd.read_csv(os.path.join(config.SPLITS_DIR, "test.csv"))
    return test_df


def get_predictions(model, test_ds):
    """
    Runs the model on the test dataset and collects:
    - true labels
    - predicted class (argmax of probabilities)
    - full probability distributions (needed for ROC-AUC)
    """
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred_probs.extend(probs)
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)  # predicted class = highest probability

    return y_true, y_pred, y_pred_probs


def compute_metrics(y_true, y_pred, y_pred_probs):
    """
    Computes all required metrics and prints them clearly.
    """
    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # 'weighted' averaging accounts for class imbalance by weighting
    # each class's score by how many true instances it has
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")

    # ROC-AUC for multi-class: 'ovr' = one-vs-rest, treats each class as
    # a separate binary problem (this class vs all others) and averages
    roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="weighted")
    print(f"ROC-AUC (weighted, one-vs-rest): {roc_auc:.4f}")

    print("\n" + "=" * 60)
    print("PER-CLASS DETAILED REPORT")
    print("=" * 60)
    report = classification_report(
        y_true, y_pred, target_names=config.CLASS_NAMES, zero_division=0
    )
    print(report)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


def plot_confusion_matrix(y_true, y_pred):
    """
    Generates and saves a confusion matrix heatmap image.
    Rows = true labels, Columns = predicted labels.
    The diagonal represents correct predictions; anything off-diagonal
    shows exactly which classes get confused with which.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Diabetic Retinopathy Classification")
    plt.tight_layout()

    save_path = os.path.join(config.PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()


def main():
    print("Loading test data...")
    test_df = load_test_data()
    print(f"Test set size: {len(test_df)}")

    print("\nBuilding test dataset pipeline...")
    # shuffle=False: order doesn't matter for evaluation, and keeping it
    # consistent makes debugging easier if needed
    test_ds = build_dataset(test_df, shuffle=False)

    print("\nLoading trained model...")
    model_path = os.path.join(config.SAVED_MODELS_DIR, "final_model.keras")
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from: {model_path}")

    print("\nRunning predictions on test set...")
    y_true, y_pred, y_pred_probs = get_predictions(model, test_ds)

    metrics = compute_metrics(y_true, y_pred, y_pred_probs)

    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_true, y_pred)

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()