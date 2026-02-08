from pathlib import Path
import pandas as pd
import deepmatcher as dm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

# -------------------------
# Konfiguration
# -------------------------
MODEL_DIR = Path(r"../Matching Models/Matching Models")

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
}

MODEL_TYPES = ["sif", "rnn", "attention", "hybrid"]
DATASETS = ["ab", "ag", "ia", "wm"]

THRESHOLD = 0.5  # Match-Score Schwelle für Klassifikation


# -------------------------
# Daten laden
# -------------------------
def load_testset(dm, dataset_key: str, cache: bool = False, test_file: str = "test_full.csv"):
    if dataset_key not in DATA_DIRS:
        raise KeyError(f"Unbekanntes Dataset '{dataset_key}'. Erlaubt: {list(DATA_DIRS.keys())}")

    return dm.data.process(
        path=str(DATA_DIRS[dataset_key]),
        test=test_file,
        cache=cache
    )


test_sets = {k: load_testset(dm, k, cache=False) for k in DATASETS}


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
# Ergänzung: großer Score-DF pro Datensatz
# -------------------------
def build_big_scores_df_for_dataset(ds_key: str) -> pd.DataFrame:
    """
    Erstellt pro Datensatz einen großen DF:
    Spalten: id, label, sif, rnn, attention, hybrid
    wobei jede Modell-Spalte den match_score des jeweiligen Models enthält.
    """
    dataset = test_sets[ds_key]

    gold_df = pd.read_csv(dataset.path)
    id_col = dataset.id_field
    label_col = dataset.label_field

    # Basis: id + label
    base = gold_df[[id_col, label_col]].copy()
    base[id_col] = base[id_col].astype(str)
    base = base.rename(columns={label_col: "label"}).set_index(id_col)

    # Scores je Modell hinzufügen (Spaltenname = Modelltyp)
    for model_type in MODEL_TYPES:
        model = load_matching_model(dm, dataset_key=ds_key, attr_summarizer=model_type)

        pred_df = model.run_prediction(dataset, output_attributes=False).copy()
        pred_df.index = pred_df.index.astype(str)

        if "match_score" not in pred_df.columns:
            raise KeyError(
                f"'match_score' fehlt in Prediction-DF für dataset={ds_key}, model={model_type}"
            )

        scores = pred_df[["match_score"]].rename(columns={"match_score": model_type})

        # Join nach id
        base = base.join(scores, how="inner")

    # zurück zu Spaltenform
    base = base.reset_index().rename(columns={id_col: "id"})
    return base


# -------------------------
# Evaluation: alle Modelle x alle Datensätze
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
# 1) Eine Gesamt-Tabelle (MultiIndex: Modelltyp x Datensatz)
overall_table = (
    results
    .set_index(["model", "dataset"])
    .sort_index()
)

# 2) Eine Tabelle pro Datensatz (Modelle als Zeilen)
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

    # als PNG + SVG (Dateiname = Datensatzkürzel)
    save_table_as_images(tbl, VIS_DIR / ds, title=f"Evaluation ({ds})")


# =========================
# Ergänzung: DataFrames als CSV speichern
# =========================

# --- Metrik-DataFrames speichern ---
results.to_csv(DF_DIR / "metrics_results_long.csv", index=False)
overall_table.to_csv(DF_DIR / "metrics_overall_table.csv")

for ds, tbl in tables_per_dataset.items():
    tbl.to_csv(DF_DIR / f"metrics_{ds}.csv")


# --- große Score-DFs erzeugen und speichern ---
big_scores_per_dataset = {}
for ds_key in DATASETS:
    big_df = build_big_scores_df_for_dataset(ds_key)
    big_scores_per_dataset[ds_key] = big_df

    big_df.to_csv(DF_DIR / f"big_scores_{ds_key}.csv", index=False)

print(f"\nAlle CSVs gespeichert unter: {DF_DIR.resolve()}")
print(f"Alle Visualisierungen gespeichert unter: {VIS_DIR.resolve()}")

