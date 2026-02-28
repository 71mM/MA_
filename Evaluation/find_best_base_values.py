from pathlib import Path
import pandas as pd

METRICS = ["f1", "recall", "accuracy", "precision"]
VALUE_COLS = ["test_full", "train_base", "validation_full"]

def main():
    eval_dir = Path(__file__).resolve().parent
    df_dir = eval_dir / "dataframes"

    best_path = df_dir / "best_epoch_per_dataset_model.csv"
    best_df = pd.read_csv(best_path)

    rows_out = []

    for _, r in best_df.iterrows():
        dataset = str(r["dataset"])
        model = str(r["model"])
        best_epoch = int(r["best_epoch"])

        for metric in METRICS:

            metric_variants = [metric, metric.replace(" ", "_")]

            lc_path = None
            for mv in metric_variants:
                p = df_dir / f"learning_curve_{mv}__{dataset}__{model}.csv"
                if p.exists():
                    lc_path = p
                    break

            if lc_path is None:
                print(f"Datei nicht gefunden für {metric}, {dataset}, {model}")
                continue

            lc_df = pd.read_csv(lc_path)

            # epoch-Spalte ist explizit vorhanden
            row = lc_df[lc_df["epoch"] == best_epoch]

            if row.empty:
                print(f"Epoch {best_epoch} nicht gefunden in {lc_path.name}")
                continue

            row = row.iloc[0]

            rows_out.append({
                "metric": metric,
                "dataset": dataset,
                "model": model,
                "best_epoch": best_epoch,
                "train_base": row["train_base"],
                "validation_full": row["validation_full"],
                "test_full": row["test_full"],
            })

    out_df = pd.DataFrame(rows_out)

    out_path = eval_dir / "best_epoch_values.csv"
    out_df.to_csv(out_path, index=False)

    print(f"Fertig. Datei geschrieben nach: {out_path}")


if __name__ == "__main__":
    main()
