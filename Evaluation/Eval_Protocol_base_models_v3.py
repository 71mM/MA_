from pathlib import Path
import numpy as np
import pandas as pd
import deepmatcher as dm

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ============================================================
# Konfiguration (wie zuvor – Pfade bleiben gültig)
# ============================================================
MODEL_DIR = Path(r"../Base-Models/Base-Model Checkpoints")

VIS_DIR = Path(r"../Evaluation/Vis/confusion_matrices_best_checkpoints")
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

MODEL_TYPES = ["attention", "sif", "rnn", "hybrid"]
DATASETS = ["ab", "ag", "ia", "wm", "dplb"]

THRESHOLD = 0.5
TEST_SPLIT = "test_full.csv"

# ============================================================
# DeepMatcher Split laden (nur Prediction)
# ============================================================
def load_test_as_dataset(ds_key: str, csv_file: str = TEST_SPLIT, cache: bool = False):
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
# Checkpoint Pfad + Model Loader
# ============================================================
def best_ckpt_path(ds_key: str, model_type: str) -> Path:
    return MODEL_DIR / f"Model_{model_type}_{ds_key}_best.pth"

def load_model(model_type: str, ckpt: Path) -> dm.MatchingModel:
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {ckpt}")
    model = dm.MatchingModel(attr_summarizer=model_type)
    model.load_state(str(ckpt))
    return model

# ============================================================
# Confusion Matrix Plot Helper
# ============================================================
def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=180)
    im = ax.imshow(cm, vmin=0, vmax=int(cm.max()) if cm.size else 1)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Non-Match", "Pred Match"])
    ax.set_yticklabels(["True Non-Match", "True Match"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center")

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Test-Datasets einmal laden
    test_ds_cache = {ds: load_test_as_dataset(ds, TEST_SPLIT, cache=False) for ds in DATASETS}

    records = []

    for model_type in MODEL_TYPES:
        for ds in DATASETS:
            ckpt = best_ckpt_path(ds, model_type)
            if not ckpt.exists():
                print(f"[WARN] Fehlt: {ckpt.name}")
                continue

            print(f"[INFO] Confusion Matrix: {ckpt.name}")
            model = load_model(model_type, ckpt)
            test_ds = test_ds_cache[ds]

            y_true, y_score = get_true_and_scores(model, test_ds)
            y_pred = (y_score >= THRESHOLD).astype(int)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])


            records.append({
                "dataset": ds,
                "model": model_type,
                "checkpoint": ckpt.name,
                "TN": int(cm[0, 0]),
                "FP": int(cm[0, 1]),
                "FN": int(cm[1, 0]),
                "TP": int(cm[1, 1]),
            })

            plot_confusion_matrix(
                cm=cm,
                title=f"{ds} — {model_type} — test_full (thr={THRESHOLD})",
                out_path=VIS_DIR / f"cm__{ds}__{model_type}__best.png"
            )

    # Summary CSV
    out_csv = DF_DIR / "confusion_matrices_best_checkpoints_test_full.csv"
    pd.DataFrame(records).sort_values(["dataset", "model"]).to_csv(out_csv, index=False)

    print(f"\n[OK] Confusion-Matrix-Plots: {VIS_DIR.resolve()}")
    print(f"[OK] Summary CSV: {out_csv.resolve()}")
