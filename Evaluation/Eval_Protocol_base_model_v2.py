from pathlib import Path
import re
import numpy as np
import pandas as pd
import deepmatcher as dm

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, log_loss
)
import matplotlib.pyplot as plt

# ============================================================
# Konfiguration
# ============================================================
MODEL_DIR = Path(r"../Base-Models/Base-Model Checkpoints")

VIS_DIR = Path(r"../Evaluation/Vis")
VIS_DIR.mkdir(parents=True, exist_ok=True)

DF_DIR = Path(r"../Evaluation/dataframes")
DF_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIRS = {
    "ab": Path(r"../data/abt-buy_textual/"),
    "ag": Path(r"../data/Amazon-Google_structured/"),
    "ia": Path(r"../data/Itunes-Amazon/"),
    "wm": Path(r"../data/Walmart-Amazon_dirty/"),
    "dplb": Path(r"../data/dblp_scholar_exp_data"),
}

MODEL_TYPES = ["sif", "rnn", "attention", "hybrid"]
DATASETS = ["ab", "ag", "ia", "wm", "dplb"]

THRESHOLD = 0.5

SPLITS = {
    "train_base": "train_base.csv",
    "validation_full": "validation_full.csv",
    "test_full": "test_full.csv",
}

CKPT_SUFFIX_EPOCH_REGEX = re.compile(r"_ep(\d+)\.pth$", re.IGNORECASE)

# ============================================================
# Split als DeepMatcher-Dataset laden (nur Prediction)
# ============================================================
def load_split_as_dataset(ds_key: str, csv_file: str, cache: bool = False):
    if ds_key not in DATA_DIRS:
        raise KeyError(f"Unbekanntes Dataset '{ds_key}'. Erlaubt: {list(DATA_DIRS.keys())}")

    path = DATA_DIRS[ds_key] / csv_file
    if not path.exists():
        raise FileNotFoundError(f"[{ds_key}] Split-Datei fehlt: {path}")

    return dm.data.process(
        path=str(DATA_DIRS[ds_key]),
        test=csv_file,
        cache=cache
    )

# ============================================================
# y_true & match_score aligned
# ============================================================
def get_true_and_scores(model: dm.MatchingModel, dataset):
    pred_df = model.run_prediction(dataset, output_attributes=False)
    gold_df = pd.read_csv(dataset.path)

    id_col = dataset.id_field
    label_col = dataset.label_field

    gold_df[id_col] = gold_df[id_col].astype(str)

    pred_df = pred_df.copy()
    pred_df.index = pred_df.index.astype(str)
    pred_df = pred_df.drop(columns=["label"], errors="ignore")

    merged = gold_df[[id_col, label_col]].set_index(id_col).join(pred_df, how="inner")

    if "match_score" not in merged.columns:
        raise KeyError("'match_score' fehlt nach Join. Prüfe Model-Prediction-Output.")

    y_true = merged[label_col].astype(int).to_numpy()
    y_score = merged["match_score"].astype(float).to_numpy()
    return y_true, y_score

# ============================================================
# Metriken: BCE, F1, Precision, Recall
# ============================================================
def compute_metrics(y_true, y_score, threshold: float = THRESHOLD):
    y_pred = (y_score >= threshold).astype(int)

    # numerische Stabilität für log_loss
    eps = 1e-15
    y_score_clipped = np.clip(y_score, eps, 1 - eps)

    return {
        "bce": float(log_loss(y_true, y_score_clipped)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

# ============================================================
# Checkpoints finden nach Naming Convention
# ============================================================
def epoch_ckpt_path(ds_key: str, model_type: str, epoch: int) -> Path:
    return MODEL_DIR / f"Model_{model_type}_{ds_key}_ep{epoch}.pth"

def best_ckpt_path(ds_key: str, model_type: str) -> Path:
    return MODEL_DIR / f"Model_{model_type}_{ds_key}_best.pth"

def list_epoch_checkpoints(ds_key: str, model_type: str) -> dict:
    """
    Returns: {epoch: Path} sortiert
    """
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"MODEL_DIR existiert nicht: {MODEL_DIR}")

    epoch_map = {}
    for p in MODEL_DIR.glob(f"Model_{model_type}_{ds_key}_ep*.pth"):
        m = CKPT_SUFFIX_EPOCH_REGEX.search(p.name)
        if not m:
            continue
        epoch = int(m.group(1))
        epoch_map[epoch] = p

    return dict(sorted(epoch_map.items(), key=lambda kv: kv[0]))

def load_model(model_type: str, ckpt: Path) -> dm.MatchingModel:
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {ckpt}")
    model = dm.MatchingModel(attr_summarizer=model_type)
    model.load_state(str(ckpt))
    return model

# ============================================================
# Learning Curves (aus Predictions) berechnen
# ============================================================
def compute_learning_curves_for_dataset_model(ds_key: str, model_type: str) -> pd.DataFrame:
    epoch_ckpts = list_epoch_checkpoints(ds_key, model_type)

    # Fallback: wenn keine epoch ckpts vorhanden -> best als "epoch=0"
    if not epoch_ckpts:
        best = best_ckpt_path(ds_key, model_type)
        if not best.exists():
            print(f"[WARN] Keine Checkpoints gefunden für {ds_key}/{model_type}")
            return pd.DataFrame(columns=["dataset", "model", "epoch", "split", "bce", "f1", "precision", "recall"])
        print(f"[WARN] Keine Epoch-Checkpoints für {ds_key}/{model_type}. Nutze best als epoch=0.")
        epoch_ckpts = {0: best}

    # Splits laden (einmal)
    split_datasets = {
        split_name: load_split_as_dataset(ds_key, csv_file, cache=False)
        for split_name, csv_file in SPLITS.items()
    }

    records = []
    for ep, ckpt in epoch_ckpts.items():
        model = load_model(model_type, ckpt)

        for split_name, ds in split_datasets.items():
            y_true, y_score = get_true_and_scores(model, ds)
            met = compute_metrics(y_true, y_score, threshold=THRESHOLD)
            records.append({
                "dataset": ds_key,
                "model": model_type,
                "epoch": int(ep),
                "split": split_name,
                **met
            })

    return pd.DataFrame(records).sort_values(["dataset", "model", "epoch", "split"])

# ============================================================
# Plot: Panel pro (dataset, model) mit 4 Subplots:
# BCE, F1, Precision, Recall jeweils train/val/test
# ============================================================
def plot_metric_panel_for_dataset_model(curves_long: pd.DataFrame, ds_key: str, model_type: str, out_path: Path):
    sub = curves_long[(curves_long["dataset"] == ds_key) & (curves_long["model"] == model_type)].copy()
    if sub.empty:
        print(f"[WARN] Keine Daten für Plot {ds_key}/{model_type}")
        return

    min_ep = int(sub["epoch"].min())
    max_ep = int(sub["epoch"].max())
    if min_ep == max_ep:
        min_ep -= 1
        max_ep += 1

    def piv(metric: str) -> pd.DataFrame:
        return sub.pivot_table(index="epoch", columns="split", values=metric, aggfunc="mean").sort_index()

    metrics = [
        ("bce", "BCE Loss"),
        ("f1", "F1-Score"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), dpi=160, sharex=True)
    axes = axes.ravel()

    for ax, (mkey, title) in zip(axes, metrics):
        p = piv(mkey)

        ax.set_xlim(min_ep, max_ep)
        ax.grid(True, alpha=0.2)

        # BCE ist nicht auf [0,1] beschränkt
        if mkey != "bce":
            ax.set_ylim(0, 1)

        for split in ["train_base", "validation_full", "test_full"]:
            if split in p.columns:
                ax.plot(p.index.to_numpy(), p[split].to_numpy(), label=split)

        ax.set_title(title)
        ax.set_xlabel("Epoch")

    # Legend nur einmal
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"{ds_key}/{model_type} — Curves (BCE/F1/Precision/Recall)", y=0.995,
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Bestes (model, epoch) pro Dataset nach validation_full F1
# Confusion Matrix auf test_full (ein Plot pro Dataset)
# ============================================================
def best_model_epoch_per_dataset(curves_long: pd.DataFrame) -> pd.DataFrame:
    val = curves_long[curves_long["split"] == "validation_full"].copy()
    if val.empty:
        raise ValueError("Keine validation_full Werte in curves_long gefunden.")

    idx = val.groupby("dataset")["f1"].idxmax()
    best = val.loc[idx, ["dataset", "model", "epoch", "f1"]].copy()
    best = best.rename(columns={"model": "best_model", "epoch": "best_epoch", "f1": "best_val_f1"}).sort_values(["dataset"])
    best["best_epoch"] = best["best_epoch"].astype(int)
    return best.reset_index(drop=True)

def plot_confusion_matrix_best_model_per_dataset(curves_long: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    best_tbl = best_model_epoch_per_dataset(curves_long)

    # test_full datasets einmal pro ds laden
    test_ds_cache = {ds: load_split_as_dataset(ds, SPLITS["test_full"], cache=False) for ds in DATASETS}

    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in best_tbl.iterrows():
        ds = row["dataset"]
        mt = row["best_model"]
        ep = int(row["best_epoch"])
        bv = float(row["best_val_f1"])

        ckpt = epoch_ckpt_path(ds, mt, ep)
        if not ckpt.exists():
            # fallback: falls ep=0 (wenn nur best genutzt wurde)
            if ep == 0:
                ckpt = best_ckpt_path(ds, mt)
            if not ckpt.exists():
                print(f"[WARN] Kein CKPT gefunden für best {ds}: {mt} ep{ep}")
                continue

        model = load_model(mt, ckpt)
        test_ds = test_ds_cache[ds]

        y_true, y_score = get_true_and_scores(model, test_ds)
        y_pred = (y_score >= THRESHOLD).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # [[TN, FP],[FN, TP]]

        fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=180)
        im = ax.imshow(cm, vmin=0, vmax=int(cm.max()) if cm.size else 1)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred No-Match", "Pred Match"])
        ax.set_yticklabels(["True No-Match", "True Match"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(int(v)), ha="center", va="center")

        ax.set_title(f"{ds} — Best: {mt} ep{ep} | \nConfusion Matrix on test_full")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        out_path = out_dir / f"confusion_matrix_best__{ds}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    return best_tbl

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    all_curves = []

    for mt in MODEL_TYPES:
        for ds in DATASETS:
            print(f"[INFO] Evaluate from checkpoints: dataset={ds}, model={mt}")
            df = compute_learning_curves_for_dataset_model(ds, mt)
            all_curves.append(df)

    curves_long = pd.concat(all_curves, ignore_index=True)

    # CSV: long (alle Metriken je epoch/split)
    curves_long.to_csv(DF_DIR / "learning_curves_long.csv", index=False)

    # Optional: wide CSVs je (ds, mt, metric)
    for ds in DATASETS:
        for mt in MODEL_TYPES:
            sub = curves_long[(curves_long["dataset"] == ds) & (curves_long["model"] == mt)]
            if sub.empty:
                continue
            for metric in ["bce", "f1", "precision", "recall"]:
                wide = sub.pivot_table(index="epoch", columns="split", values=metric, aggfunc="mean").sort_index()
                wide.to_csv(DF_DIR / f"learning_curve_{metric}__{ds}__{mt}.csv")

    # Plots: Panel pro (ds, mt)
    for ds in DATASETS:
        for mt in MODEL_TYPES:
            plot_metric_panel_for_dataset_model(
                curves_long=curves_long,
                ds_key=ds,
                model_type=mt,
                out_path=VIS_DIR / "metric_panels" / f"panel__{ds}__{mt}"
            )

    # Confusion Matrix: bestes Modell pro Dataset (ein Plot pro Dataset)
    best_tbl = plot_confusion_matrix_best_model_per_dataset(
        curves_long=curves_long,
        out_dir=VIS_DIR / "confusion_matrices_best_model_per_dataset"
    )
    best_tbl.to_csv(DF_DIR / "best_model_epoch_per_dataset.csv", index=False)

    print(f"\n[OK] learning_curves_long.csv: {(DF_DIR / 'learning_curves_long.csv').resolve()}")
    print(f"[OK] metric panels dir: {(VIS_DIR / 'metric_panels').resolve()}")
    print(f"[OK] confusion matrices dir: {(VIS_DIR / 'confusion_matrices_best_model_per_dataset').resolve()}")
    print(f"[OK] best_model_epoch_per_dataset.csv: {(DF_DIR / 'best_model_epoch_per_dataset.csv').resolve()}")
