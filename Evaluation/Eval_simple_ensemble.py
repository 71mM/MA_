from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Konfiguration
# -----------------------------
CSV_PATHS = [
    "../Evaluation/dataframes/big_scores_ab__test_full.csv",
    "../Evaluation/dataframes/big_scores_ag__test_full.csv",
    "../Evaluation/dataframes/big_scores_ia__test_full.csv",
    "../Evaluation/dataframes/big_scores_wm__test_full.csv",
    "../Evaluation/dataframes/big_scores_dplb__test_full.csv"
]

# Output-Verhalten:
# - True: erzeugt neue Dateien <stem>_with_ensembles.csv
# - False: überschreibt die Eingabedatei (in-place)
WRITE_NEW_FILES = True

# Ensemble-Spaltennamen
MAJ_COL = "ensemble_majority"
AVG_COL = "ensemble_average"

# Threshold für Binarisierung / Average
THRESH = 0.5


# -----------------------------
# Helper: Spaltenerkennung
# -----------------------------
def find_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    """Findet eine Spalte anhand Regex-Patterns (erst fullmatch, dann contains)."""
    cols = list(df.columns)
    for p in patterns:
        for c in cols:
            if re.fullmatch(p, str(c).strip(), flags=re.IGNORECASE):
                return c
    for p in patterns:
        for c in cols:
            if re.search(p, str(c), flags=re.IGNORECASE):
                return c
    return None


def find_model_cols(df: pd.DataFrame) -> List[str]:
    """
    Versucht SIF/Hybrid/RNN/Attention Spalten zu finden.
    Fallback: nimmt die ersten 4 "numerischen" Spalten (mit hoher numeric quote),
    falls Namen nicht passen.
    """
    name_map: Dict[str, str] = {}
    patterns = {
        "sif": [r"sif", r"sif[_\s-]*score", r"sif[_\s-]*pred"],
        "hybrid": [r"hybrid", r"hybrid[_\s-]*score", r"hybrid[_\s-]*pred"],
        "rnn": [r"rnn", r"rnn[_\s-]*score", r"rnn[_\s-]*pred"],
        "attention": [r"attention", r"attn", r"attention[_\s-]*score", r"attention[_\s-]*pred"],
    }

    for key, pats in patterns.items():
        col = find_col(df, pats)
        if col is not None:
            name_map[key] = col

    if len(name_map) < 4:
        # Fallback: 4 numerische Spalten suchen (IDs/Labels rausfiltern)
        excluded = {
            c for c in df.columns
            if re.search(r"(label|target|ground|true|gt|y_true|class|id|uuid|file|name)", str(c), re.I)
        }
        numeric_cols = []
        for c in df.columns:
            if c in excluded:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().mean() > 0.8:  # "meistens numerisch"
                numeric_cols.append(c)

        chosen = list(dict.fromkeys(list(name_map.values()) + numeric_cols))[:4]
        if len(chosen) >= 4:
            for k in ["sif", "hybrid", "rnn", "attention"]:
                if k not in name_map:
                    for cand in chosen:
                        if cand not in name_map.values():
                            name_map[k] = cand
                            break

    cols = [name_map.get(k) for k in ["sif", "hybrid", "rnn", "attention"] if name_map.get(k) is not None]
    return cols


def find_label_col(df: pd.DataFrame) -> Optional[str]:
    """Sucht eine Ground-Truth Spalte."""
    return find_col(df, [r"label", r"target", r"y_true", r"ground[_\s-]*truth", r"gt", r"class"])


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Binary classification metrics: accuracy/precision/recall/f1 + confusion."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# -----------------------------
# Plot
# -----------------------------
def plot_method(summary_df: pd.DataFrame, method_name: str, out_png: str) -> Optional[str]:
    sub = summary_df[summary_df["method"] == method_name].copy()
    if sub.empty:
        return None

    files = list(sub["file"])
    metrics = ["accuracy", "precision", "recall", "f1"]

    x = np.arange(len(files))
    width = 0.2

    plt.figure(figsize=(10, 4))
    for i, met in enumerate(metrics):
        plt.bar(x + (i - 1.5) * width, sub[met].values, width, label=met)

    plt.xticks(x, files, rotation=0)
    plt.ylim(0, 1)
    plt.title(f"Ensemble metrics: {method_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return out_png


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    per_file_metrics_paths = []
    updated_paths = []
    summary_rows = []

    for in_p in CSV_PATHS:
        df = pd.read_csv(in_p)
        model_cols = find_model_cols(df)
        if len(model_cols) < 4:
            raise ValueError(
                f"Konnte in '{in_p}' nicht 4 Modellspalten finden.\n"
                f"Gefunden: {model_cols}\nAlle Spalten: {list(df.columns)}"
            )

        label_col = find_label_col(df)

        # Modelle als numeric lesen
        m = df[model_cols].apply(pd.to_numeric, errors="coerce")

        # Ensemble 1: Majority Vote (erst pro Modell binarize, dann >=2 -> 1)
        bin_m = (m >= THRESH).astype(int)
        maj = (bin_m.sum(axis=1) >= 2).astype(int)  # tie 2:2 -> 1

        # Ensemble 2: Average-Threshold
        avg = (m.mean(axis=1) >= THRESH).astype(int)

        # Ensemble-Spalten schreiben/überschreiben
        df[MAJ_COL] = maj
        df[AVG_COL] = avg

        base = Path(in_p).stem
        out_csv = str(Path(in_p).with_name(f"{base}_with_ensembles.csv")) if WRITE_NEW_FILES else in_p
        df.to_csv(out_csv, index=False)
        updated_paths.append(out_csv)

        # Metrics berechnen (falls Label existiert)
        metrics_rows = []
        if label_col is not None:
            y_true = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
            mmaj = compute_metrics(y_true, maj.to_numpy())
            mavg = compute_metrics(y_true, avg.to_numpy())

            metrics_rows.append({"file": base, "method": "majority_vote", **mmaj})
            metrics_rows.append({"file": base, "method": "average_threshold", **mavg})
            summary_rows.extend(metrics_rows)
        else:
            print(f"[WARN] Keine Label-Spalte in '{in_p}' gefunden -> Metrics werden übersprungen.")

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_csv = str(Path(in_p).with_name(f"{base}_ensemble_metrics.csv"))
        metrics_df.to_csv(metrics_csv, index=False)
        per_file_metrics_paths.append(metrics_csv)

    # Summary
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = "ensemble_metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Plots
    plot_method(summary_df, "majority_vote", "ensemble_majority_metrics.png")
    plot_method(summary_df, "average_threshold", "ensemble_average_metrics.png")

    print("Fertig.")
    print("CSVs mit Ensembles:", updated_paths)
    print("Per-File Metrics CSVs:", per_file_metrics_paths)
    print("Summary:", summary_csv)
    print("PNGs: ensemble_majority_metrics.png, ensemble_average_metrics.png")


if __name__ == "__main__":
    main()