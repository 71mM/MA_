#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eval Protocol
- Dateien werden DIREKT im Script definiert
- Berechnet Accuracy, Precision, Recall, F1
- Optional: ROC, PR, Confusion Matrix
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# ============================================================
# 🔥 HIER DEINE PFADE EINTRAGEN
# ============================================================

BASE_PATH = Path(
    "../Meta_Matcher/scripts/Meta_Matcher/scripts/predictions/"
)

FILES = [
    "abt-buy_textual_test_predictions.csv",
    "Amazon-Google_structured_test_predictions.csv",
    "Itunes-Amazon_test_predictions.csv",
    "Walmart-Amazon_dirty_test_predictions.csv",
    "dblp_scholar_exp_data_test_predictions.csv"


]
OUTPUT_DIR = Path("dataframes/Meta_Matcher")
MAKE_PLOTS = True


# ============================================================
# Metric Calculation
# ============================================================

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# ============================================================
# Plot Functions
# ============================================================

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_roc(y_true, y_score, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    return roc_auc


def plot_pr(y_true, y_score, title, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(recall, precision, label=f"AP={ap:.4f}")

    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    return ap


# ============================================================
# Evaluation
# ============================================================

def evaluate():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    all_true = []
    all_pred = []
    all_score = []

    for file in FILES:

        file_path = BASE_PATH / file

        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} nicht gefunden")

        print(f"Evaluating: {file_path}")

        df = pd.read_csv(file_path)

        y_true = df["y_true"]
        y_pred = df["pred_label"]

        metrics = compute_metrics(y_true, y_pred)
        metrics["file"] = file
        metrics["n"] = len(df)

        # Score-basierte Metriken
        if "match_prob" in df.columns:
            y_score = df["match_prob"]
            all_score.append(y_score)

            if MAKE_PLOTS:
                metrics["roc_auc"] = plot_roc(
                    y_true, y_score,
                    f"ROC - {file}",
                    OUTPUT_DIR / f"roc_{file}.png"
                )
                metrics["avg_precision"] = plot_pr(
                    y_true, y_score,
                    f"PR - {file}",
                    OUTPUT_DIR / f"pr_{file}.png"
                )

        if MAKE_PLOTS:
            plot_confusion_matrix(
                y_true, y_pred,
                f"Confusion Matrix - {file}",
                OUTPUT_DIR / f"cm_{file}.png"
            )

        results.append(metrics)
        all_true.append(y_true)
        all_pred.append(y_pred)

    # =====================================================
    # TOTAL (über alle 5 Dateien)
    # =====================================================

    y_true_all = pd.concat(all_true)
    y_pred_all = pd.concat(all_pred)

    total_metrics = compute_metrics(y_true_all, y_pred_all)
    total_metrics["file"] = "TOTAL"
    total_metrics["n"] = len(y_true_all)

    if all_score:
        y_score_all = pd.concat(all_score)

        if MAKE_PLOTS:
            total_metrics["roc_auc"] = plot_roc(
                y_true_all, y_score_all,
                "ROC - TOTAL",
                OUTPUT_DIR / "roc_TOTAL.png"
            )
            total_metrics["avg_precision"] = plot_pr(
                y_true_all, y_score_all,
                "PR - TOTAL",
                OUTPUT_DIR / "pr_TOTAL.png"
            )

    results.append(total_metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "metrics_report.csv", index=False)

    print("\n===== FINAL RESULTS =====\n")
    print(results_df)


if __name__ == "__main__":
    evaluate()
