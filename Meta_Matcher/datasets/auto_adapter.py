import re
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, List

@dataclass
class AutoAdapterConfig:
    id_col: str = "id"
    label_col: str = "label"
    left_prefix: str = "left_"
    right_prefix: str = "right_"
    # Spalten, die NICHT in den Text sollen (Preis holen wir separat)
    exclude_price: bool = True
    # Ausgabe als "key: value | key: value" statt nur values
    include_keys: bool = True
    sep: str = " | "

def _infer_price_col(df: pd.DataFrame, prefix: str) -> Optional[str]:
    # findet left_price, left_Price, leftPRICE ...
    pat = re.compile(rf"^{re.escape(prefix)}price$", re.IGNORECASE)
    for c in df.columns:
        if pat.match(c):
            return c
    return None

def standardize_raw_auto(df_raw: pd.DataFrame, cfg: AutoAdapterConfig) -> pd.DataFrame:
    if cfg.id_col not in df_raw.columns:
        raise ValueError(f"id_col '{cfg.id_col}' nicht gefunden")

    out = pd.DataFrame()
    out["id"] = df_raw[cfg.id_col]

    if cfg.label_col in df_raw.columns:
        out["label"] = df_raw[cfg.label_col]

    left_cols = [c for c in df_raw.columns if c.startswith(cfg.left_prefix)]
    right_cols = [c for c in df_raw.columns if c.startswith(cfg.right_prefix)]
    if not left_cols or not right_cols:
        raise ValueError("Keine left_* / right_* Spalten gefunden")

    left_price_col = _infer_price_col(df_raw, cfg.left_prefix)
    right_price_col = _infer_price_col(df_raw, cfg.right_prefix)

    # Textspalten = alle left_* außer price
    def filter_text_cols(cols: List[str], price_col: Optional[str]) -> List[str]:
        keep = []
        for c in cols:
            if cfg.exclude_price and price_col is not None and c == price_col:
                continue
            keep.append(c)
        return sorted(keep)

    left_text_cols = filter_text_cols(left_cols, left_price_col)
    right_text_cols = filter_text_cols(right_cols, right_price_col)

    def join_row(row, cols, prefix):
        parts = []
        for c in cols:
            v = row.get(c, "")
            if pd.isna(v):
                continue
            s = str(v).strip()
            if not s:
                continue
            if cfg.include_keys:
                key = c[len(prefix):]
                parts.append(f"{key}: {s}")
            else:
                parts.append(s)
        return cfg.sep.join(parts)

    out["left_text"] = df_raw.apply(lambda r: join_row(r, left_text_cols, cfg.left_prefix), axis=1)
    out["right_text"] = df_raw.apply(lambda r: join_row(r, right_text_cols, cfg.right_prefix), axis=1)

    # prices optional
    if left_price_col is not None:
        out["left_price"] = pd.to_numeric(df_raw[left_price_col], errors="coerce").fillna(0.0)
    if right_price_col is not None:
        out["right_price"] = pd.to_numeric(df_raw[right_price_col], errors="coerce").fillna(0.0)

    return out
