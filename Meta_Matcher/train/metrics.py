import numpy as np
from typing import Dict

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        order = np.argsort(y_prob)
        y_sorted = y_true[order]
        n_pos = (y_sorted == 1).sum()
        n_neg = (y_sorted == 0).sum()
        ranks = np.arange(1, len(y_sorted) + 1)
        sum_ranks_pos = ranks[y_sorted == 1].sum()
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg + 1e-12)

    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}
