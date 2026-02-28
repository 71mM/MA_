import pandas as pd
from typing import List

DEFAULT_SCORE_COLS = ["sif", "rnn", "attention", "hybrid"]

def load_scores(scores_path: str, id_col: str = "id", label_col: str = "label",
                score_cols: List[str] = None) -> pd.DataFrame:
    score_cols = score_cols or DEFAULT_SCORE_COLS
    df = pd.read_csv(scores_path)

    missing = [c for c in ([id_col] + score_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"Scores-Datei fehlt Spalten: {missing}")

    keep = [id_col] + score_cols + ([label_col] if label_col in df.columns else [])
    return df[keep].copy()
