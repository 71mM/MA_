from pathlib import Path
import pandas as pd
import deepmatcher as dm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

# -------------------------
# Konfiguration
# -------------------------
MODEL_DIR = Path(r"../Base-Models/Base-Model Checkpoints")

# Output-Ordner
VIS_DIR = Path(r"../Evaluation/Vis")
VIS_DIR.mkdir(parents=True, exist_ok=True)

DF_DIR = Path(r"../Evaluation/dataframes")
DF_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIRS = {
    "ab": Path(r"../data/abt-buy_textual/"),
    "ag": Path(r"../data/Amazon-Google_structured/"),
    "ia": Path(r"../data/Itunes-Amazon/"),
    "wm": Path(r"../data/Walmart-Amazon_dirty/"),
    "dblp": Path(r"../data/dblp_scholar_exp_data"),
}

MODEL_TYPES = ["sif", "rnn", "attention", "hybrid"]
DATASETS = ["ab", "ag", "ia", "wm", "dblp"]

THRESHOLD = 0.5  # Match-Score Schwelle für Klassifikation


BIG_SCORE_SPLITS = {
    "test_full": "test_full.csv",
    "train_full": "train_full.csv",
    "train_base": "train_base.csv",
    "train_meta": "train_meta.csv",
    "validation_full": "validation_full.csv"
}

# -------------------------
# Daten laden (Testset für Metriken)
# -------------------------
def load_testset(dm, dataset_key: str, cache: bool = False, test_file: str = "test_full.csv"):
    if dataset_key not in DATA_DIRS:
        raise KeyError(f"Unbekanntes Dataset '{dataset_key}'. Erlaubt: {list(DATA_DIRS.keys())}")

    return dm.data.process(
        path=str(DATA_DIRS[dataset_key]),
        test=test_file,
        cache=cache
    )


test_sets = {k: load_testset(dm, k, cache=False, test_file="test_full.csv") for k in DATASETS}


def load_matching_model(dm, dataset_key: str, attr_summarizer: str):
    """
    Lädt ein MatchingModel für dataset_key und summarizer (model type).
    Erwarteter Checkpoint-Name:
        Model_<attr_summarizer>_<dataset_key>_best.pth
    """
    model = dm.MatchingModel(attr_summarizer=attr_summarizer)

    ckpt = MODEL_DIR / f"Model_{attr_summarizer}_{dataset_key}_best.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {ckpt}")

    model.load_state(str(ckpt))
    return model


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

    # Falls pred_df eine 'label' Spalte hat -> entfernen (sonst Join-Kollision)
    pred_df = pred_df.drop(columns=["label"], errors="ignore")

    merged = gold_df[[id_col, label_col]].set_index(id_col).join(pred_df, how="inner")

    if "match_score" not in merged.columns:
        raise KeyError("'match_score' fehlt nach Join. Prüfe Model-Prediction-Output.")

    y_true = merged[label_col].astype(int).to_numpy()
    y_score = merged["match_score"].astype(float).to_numpy()
    return y_true, y_score


# -------------------------
# Metriken
# -------------------------
def compute_metrics(y_true, y_pred):
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_table_as_images(df: pd.DataFrame, out_basepath: Path, title: str):
    """
    Speichert df als gerenderte Tabelle in PNG und SVG.
    out_basepath ohne Endung, z.B. VIS_DIR/'ab' -> erzeugt ab.png und ab.svg
    """
    df_disp = df.copy()
    for col in df_disp.columns:
        if pd.api.types.is_numeric_dtype(df_disp[col]):
            df_disp[col] = df_disp[col].map(lambda x: f"{x:.4f}")

    nrows, ncols = df_disp.shape
    fig_w = max(6, 1.6 * ncols)
    fig_h = max(2.5, 0.6 * (nrows + (1 if title else 0)))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.axis("off")

    if title:
        ax.set_title(title, pad=12, fontsize=12, fontweight="bold")

    table = ax.table(
        cellText=df_disp.values,
        colLabels=df_disp.columns.tolist(),
        rowLabels=df_disp.index.tolist(),
        loc="center",
        cellLoc="center",
        rowLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)

    fig.savefig(out_basepath.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_basepath.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Ergänzung: beliebige CSV als Dataset laden (für Big-Scores Splits)
# -------------------------
def load_any_split_as_dataset(dm, dataset_key: str, csv_file: str, cache: bool = False):
    """
    Lädt eine beliebige CSV aus dem Dataset-Ordner als DeepMatcher-Dataset.
    Wir nutzen dm.data.process(..., test=<csv_file>), weil wir nur Vorhersagen machen wollen.
    """
    if dataset_key not in DATA_DIRS:
        raise KeyError(f"Unbekanntes Dataset '{dataset_key}'. Erlaubt: {list(DATA_DIRS.keys())}")

    return dm.data.process(
        path=str(DATA_DIRS[dataset_key]),
        test=csv_file,
        cache=cache
    )


# -------------------------
# Big Scores: pro Datensatz + Split
# -------------------------
def build_big_scores_df_for_dataset_split(ds_key: str, split_file: str) -> pd.DataFrame:
    """
    Erstellt pro Datensatz + Split einen großen DF:
    Spalten: id, label, sif, rnn, attention, hybrid
    wobei jede Modell-Spalte den match_score des jeweiligen Models enthält.
    """
    dataset = load_any_split_as_dataset(dm, ds_key, csv_file=split_file, cache=False)

    gold_df = pd.read_csv(dataset.path)
    id_col = dataset.id_field
    label_col = dataset.label_field

    base = gold_df[[id_col, label_col]].copy()
    base[id_col] = base[id_col].astype(str)
    base = base.rename(columns={label_col: "label"}).set_index(id_col)

    for model_type in MODEL_TYPES:
        model = load_matching_model(dm, dataset_key=ds_key, attr_summarizer=model_type)

        pred_df = model.run_prediction(dataset, output_attributes=False).copy()
        pred_df.index = pred_df.index.astype(str)

        if "match_score" not in pred_df.columns:
            raise KeyError(
                f"'match_score' fehlt in Prediction-DF für dataset={ds_key}, split={split_file}, model={model_type}"
            )

        scores = pred_df[["match_score"]].rename(columns={"match_score": model_type})
        base = base.join(scores, how="inner")

    base = base.reset_index().rename(columns={id_col: "id"})
    return base


# -------------------------
# Evaluation: alle Modelle x alle Datensätze (Metriken auf test_full)
# -------------------------
rows = []

for model_type in MODEL_TYPES:
    for ds_key in DATASETS:
        dataset = test_sets[ds_key]
        model = load_matching_model(dm, dataset_key=ds_key, attr_summarizer=model_type)

        y_true, y_score = _get_true_and_scores(model, dataset)
        y_pred = (y_score >= THRESHOLD).astype(int)

        metrics = compute_metrics(y_true, y_pred)

        rows.append({
            "model": model_type,      # nur Typ-Name
            "dataset": ds_key,        # Abkürzung
            **metrics
        })

results = pd.DataFrame(rows)

# -------------------------
# Tabellen erzeugen
# -------------------------
overall_table = (
    results
    .set_index(["model", "dataset"])
    .sort_index()
)

tables_per_dataset = {
    ds: results[results["dataset"] == ds]
        .drop(columns=["dataset"])
        .set_index("model")
        .sort_index()
    for ds in DATASETS
}

print("\n=== Gesamt-Tabelle (model x dataset) ===")
print(overall_table)

for ds, tbl in tables_per_dataset.items():
    print(f"\n=== Tabelle für Datensatz: {ds} ===")
    print(tbl)

    save_table_as_images(tbl, VIS_DIR / ds, title=f"Evaluation ({ds})")

# =========================
# Metrik-DataFrames als CSV speichern
# =========================
results.to_csv(DF_DIR / "overview.csv", index=False)
overall_table.to_csv(DF_DIR / "metrics_overall_table.csv")

for ds, tbl in tables_per_dataset.items():
    tbl.to_csv(DF_DIR / f"metrics_{ds}.csv")

# =========================
# Big-Scores für test_full + train_* Splits erzeugen und speichern
# =========================
big_scores_per_dataset = {}

for ds_key in DATASETS:
    big_scores_per_dataset[ds_key] = {}

    for split_name, split_file in BIG_SCORE_SPLITS.items():
        split_path = DATA_DIRS[ds_key] / split_file
        if not split_path.exists():
            print(f"[WARN] Split-Datei fehlt, überspringe: dataset={ds_key}, file={split_file}")
            continue

        big_df = build_big_scores_df_for_dataset_split(ds_key, split_file=split_file)
        big_scores_per_dataset[ds_key][split_name] = big_df

        out_path = DF_DIR / f"big_scores_{ds_key}__{split_name}.csv"
        big_df.to_csv(out_path, index=False)

print(f"\nAlle CSVs gespeichert unter: {DF_DIR.resolve()}")
print(f"Alle Visualisierungen gespeichert unter: {VIS_DIR.resolve()}")
