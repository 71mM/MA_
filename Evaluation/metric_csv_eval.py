import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# HARDCODED PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "dataframes" / "overview.csv"
OUTPUT_DIR = BASE_DIR / "plots"

# =========================
# FIXE PLOT-REIHENFOLGE
# =========================
ORDER_WITH_GAPS = [
    "SIF",
    "Attention",
    "RNN",
    "hybrid",
    "__GAP__",
    "majority_vote",
    "average",
    "__GAP__",
    "meta_matcher",
]


def metric_prefers_lower(metric_name: str) -> bool:
    """Heuristik: bei Error/Loss/RMSE/MAE/MSE etc. ist 'kleiner besser'."""
    m = metric_name.lower()
    return bool(re.search(r"(loss|error|err|rmse|mae|mse|wer|cer)", m))


def _norm(s: str) -> str:
    """Normiert Strings zum Matchen (case/whitespace)."""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV nicht gefunden unter: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required = {"model", "dataset"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV muss mindestens die Spalten {sorted(required)} enthalten.\n"
            f"Gefunden: {list(df.columns)}"
        )

    id_cols = ["model", "dataset"]
    metric_cols = [c for c in df.columns if c not in id_cols]
    if not metric_cols:
        raise ValueError("Keine Metrik-Spalten gefunden.")

    df = df.copy()
    df["model"] = df["model"].astype(str)
    df["dataset"] = df["dataset"].astype(str)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Farben + Hatches pro Metrik
    base_colors = [
        "tab:blue", "tab:orange", "tab:purple", "tab:red", "tab:brown",
        "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "tab:green"
    ]
    base_hatches = ["", "//", "\\\\", "xx", "..", "oo", "--", "++", "**", "||"]

    # Normierte Order-Liste (für robustes Matching)
    order_norm = [_norm(x) if x != "__GAP__" else "__GAP__" for x in ORDER_WITH_GAPS]

    for dataset, g in df.groupby("dataset", sort=True):
        g = g.copy()
        g["model_norm"] = g["model"].map(_norm)

        # Werte je Modell aus CSV in dict packen
        row_by_model = {mn: row for mn, row in zip(g["model_norm"], g.to_dict(orient="records"))}

        # Nur Modelle berücksichtigen, die im ORDER vorkommen (ansonsten ignorieren)
        present_models_in_order = [m for m in order_norm if m != "__GAP__" and m in row_by_model]

        if not present_models_in_order:
            # nichts zu plotten
            print(f"[WARN] Dataset '{dataset}': keine der erwarteten Modellnamen gefunden. Überspringe.")
            continue

        # X-Achse inkl. Gaps aufbauen
        x_positions = []
        x_labels = []
        models_for_positions = []  # entweder model_norm oder None (für Gap)

        x = 0
        for entry in order_norm:
            if entry == "__GAP__":
                x_positions.append(x)
                x_labels.append("")          # keine Beschriftung für Lücke
                models_for_positions.append(None)
                x += 1
            else:
                if entry in row_by_model:
                    x_positions.append(x)
                    # Original-Label wie in ORDER (schöner) – dafür mapping:
                    # suche das "pretty" Label aus ORDER_WITH_GAPS
                    pretty = ORDER_WITH_GAPS[order_norm.index(entry)]
                    x_labels.append(pretty)
                    models_for_positions.append(entry)
                    x += 1
                else:
                    # Modell aus ORDER fehlt in CSV -> überspringen (kein Slot, keine Lücke)
                    # (Falls du lieber einen leeren Slot willst: hier stattdessen Gap erzeugen)
                    continue

        x_positions = np.array(x_positions, dtype=float)

        # Welche Positionen sind "echte" Modelle?
        model_pos_idx = [i for i, m in enumerate(models_for_positions) if m is not None]
        model_positions = x_positions[model_pos_idx]
        model_norms = [models_for_positions[i] for i in model_pos_idx]

        # Matrix: (n_models, n_metrics)
        vals_matrix = np.array(
            [[float(row_by_model[m][met]) for met in metric_cols] for m in model_norms],
            dtype=float,
        )

        n_models = len(model_norms)
        n_metrics = len(metric_cols)

        # Bestwerte pro Metrik (über die echten Modelle, nicht über Lücken)
        best_mask = {}
        for j, met in enumerate(metric_cols):
            col = vals_matrix[:, j]
            if metric_prefers_lower(met):
                best_val = np.nanmin(col)
                best_mask[met] = np.isclose(col, best_val, equal_nan=False)
            else:
                best_val = np.nanmax(col)
                best_mask[met] = np.isclose(col, best_val, equal_nan=False)

        # Plot
        fig, ax = plt.subplots(figsize=(max(10, 1.15 * len(x_positions)), 5.8))

        group_width = 0.80
        bar_width = group_width / max(n_metrics, 1)

        for j, met in enumerate(metric_cols):
            col = vals_matrix[:, j]
            offsets = (j - (n_metrics - 1) / 2.0) * bar_width
            xpos = model_positions + offsets

            color = base_colors[j % len(base_colors)]
            hatch = base_hatches[j % len(base_hatches)]

            bars = ax.bar(
                xpos,
                col,
                width=bar_width * 0.95,
                label=met,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=1.0,
            )

            bm = best_mask[met]  # Länge n_models
            for i, b in enumerate(bars):
                if bm[i]:
                    b.set_edgecolor("green")
                    b.set_linewidth(3.0)

        # Achse/Labels
        ax.set_title(f"Dataset: {dataset} — Metriken pro Modell")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Modelltyp / Verfahren")
        ax.set_ylabel("Wert")

        # Vertikale Trennlinien an den Gaps (optional, aber hilft)
        for i, m in enumerate(models_for_positions):
            if m is None:
                ax.axvline(x_positions[i], linestyle=":", linewidth=1.2, alpha=0.5)

        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.legend(title="Metrik", ncols=min(4, n_metrics), frameon=True)

        fig.tight_layout()

        outfile = OUTPUT_DIR / f"{dataset}_metrics_by_model.png"
        fig.savefig(outfile, dpi=200)
        plt.close(fig)

        print(f"Gespeichert: {outfile}")

    print(f"\nAlle Plots liegen in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
