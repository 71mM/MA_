from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

def load_scores_and_labels(
    df: pd.DataFrame,
    score_cols: List[str],
    label_col: str = "label"
) -> Tuple[np.ndarray, np.ndarray]:
    scores = df[score_cols].to_numpy(dtype=np.float32)
    labels = df[label_col].to_numpy()
    return scores, labels

def infer_score_cols(df: pd.DataFrame, exclude: Optional[set] = None) -> List[str]:
    exclude = exclude or set()
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    candidates = [c for c in numeric_cols if c not in exclude]
    return candidates
