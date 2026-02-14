import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import deepmatcher as dm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, log_loss
)

from matplotlib.ticker import MaxNLocator


def _get_true_and_scores(model: dm.MatchingModel, dataset):
    """
    Holt y_true (Labels) und y_score (match_score probabilities) aligned über id.
    """
    pred_df = model.run_prediction(dataset, output_attributes=False)
    gold_df = pd.read_csv(dataset.path)
    id_col = dataset.id_field
    label_col = dataset.label_field

    gold_df[id_col] = gold_df[id_col].astype(str)

    pred_df = pred_df.copy()
    pred_df.index = pred_df.index.astype(str)

    pred_df = pred_df.drop(columns=["label"], errors="ignore")

    merged = gold_df[[id_col, label_col]].set_index(id_col).join(pred_df, how="inner")

    y_true = merged[label_col].astype(int).to_numpy()
    y_score = merged["match_score"].astype(float).to_numpy()
    return y_true, y_score


def evaluate_metrics_and_errors(model: dm.MatchingModel, dataset, threshold=0.5) -> dict:
    """
    Berechnet Accuracy / Precision / Recall / F1
    + Error-Rate (= 1-Accuracy)
    + BCE/LogLoss (über match_score probabilities).
    """
    y_true, y_score = _get_true_and_scores(model, dataset)
    y_pred = (y_score >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    err_rate = 1.0 - acc

    # LogLoss ist numerisch stabiler mit clipping
    y_score_clip = np.clip(y_score, 1e-7, 1 - 1e-7)
    bce = log_loss(y_true, y_score_clip)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "error_rate": err_rate,
        "bce": bce,
    }


def plot_metrics_subplots(epochs, train_metrics, test_metrics, out_path, title):
    """
    4 Subplots:
      1) Accuracy (Train/Test)
      2) Precision (Train/Test)
      3) Recall (Train/Test)
      4) F1 (Train/Test)

    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(title)

    def _int_epoch_ticks(ax):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(list(epochs))
        ax.set_xlim(min(epochs), max(epochs))
        ax.grid(True, alpha=0.2)

    # 1) Accuracy
    ax = axes[0, 0]
    ax.plot(epochs, train_metrics["accuracy"], label="Train Acc")
    ax.plot(epochs, test_metrics["accuracy"], label="Test Acc")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    _int_epoch_ticks(ax)

    # 2) Precision
    ax = axes[0, 1]
    ax.plot(epochs, train_metrics["precision"], label="Train Prec")
    ax.plot(epochs, test_metrics["precision"], label="Test Prec")
    ax.set_title("Precision")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    _int_epoch_ticks(ax)

    # 3) Recall
    ax = axes[1, 0]
    ax.plot(epochs, train_metrics["recall"], label="Train Recall")
    ax.plot(epochs, test_metrics["recall"], label="Test Recall")
    ax.set_title("Recall")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    _int_epoch_ticks(ax)

    # 4) F1
    ax = axes[1, 1]
    ax.plot(epochs, train_metrics["f1"], label="Train F1")
    ax.plot(epochs, test_metrics["f1"], label="Test F1")
    ax.set_title("F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    _int_epoch_ticks(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close()


# Optional: wenn du weiterhin eine einzelne BCE-Plot-Funktion willst (fixe Benennung)
def plot_learningcurve_bce(epochs, train_bce, test_bce, out_path, title):
    plt.figure()
    plt.plot(epochs, train_bce, label="Train BCE")
    plt.plot(epochs, test_bce, label="Test BCE")
    plt.xlabel("Epoch")
    plt.ylabel("BCE")
    plt.title(title)
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().set_xticks(list(epochs))
    plt.gca().set_xlim(min(epochs), max(epochs))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    print("eval_while_training loaded")
