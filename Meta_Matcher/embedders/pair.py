import os
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Auto adapter (built-in here to keep it self-contained)
# ------------------------------------------------------------
@dataclass
class AutoTextConfig:
    id_col: str = "id"
    label_col: str = "label"
    left_prefix: str = "left_"
    right_prefix: str = "right_"

    # Build text as "key: value | key: value" (recommended for MiniLM)
    include_keys: bool = True
    sep: str = " | "

    # exclude price columns from text (we add price separately)
    exclude_price_from_text: bool = True


def _infer_price_col(df: pd.DataFrame, prefix: str) -> Optional[str]:
    # matches left_price, left_Price, leftPRICE, ...
    pat = re.compile(rf"^{re.escape(prefix)}price$", re.IGNORECASE)
    for c in df.columns:
        if pat.match(c):
            return c
    return None


def _safe_str(x) -> str:
    if x is None or pd.isna(x):
        return ""
    return str(x).strip()


def _build_text_from_prefixed_columns(
    df: pd.DataFrame,
    prefix: str,
    cols: List[str],
    cfg: AutoTextConfig,
) -> pd.Series:
    def join_row(row):
        parts = []
        for c in cols:
            v = _safe_str(row.get(c, ""))
            if not v:
                continue
            if cfg.include_keys:
                key = c[len(prefix):]
                parts.append(f"{key}: {v}")
            else:
                parts.append(v)
        return cfg.sep.join(parts)

    return df.apply(join_row, axis=1)


def standardize_to_left_right_text(
    df: pd.DataFrame,
    cfg: AutoTextConfig = AutoTextConfig(),
) -> pd.DataFrame:
    """
    Ensures df has:
      left_text, right_text
      optional left_price/right_price
    Works for all your dataset schemas because they use left_* / right_* prefixes.
    """
    # If already standardized, keep it
    has_text = ("left_text" in df.columns and "right_text" in df.columns)
    if has_text:
        out = df.copy()
    else:
        left_cols = [c for c in df.columns if c.startswith(cfg.left_prefix)]
        right_cols = [c for c in df.columns if c.startswith(cfg.right_prefix)]
        if not left_cols or not right_cols:
            raise ValueError("No left_* / right_* columns found to build texts.")

        left_price_col = _infer_price_col(df, cfg.left_prefix)
        right_price_col = _infer_price_col(df, cfg.right_prefix)

        def filter_text_cols(cols: List[str], price_col: Optional[str]) -> List[str]:
            keep = []
            for c in cols:
                if cfg.exclude_price_from_text and price_col is not None and c == price_col:
                    continue
                keep.append(c)
            # stable ordering
            return sorted(keep)

        left_text_cols = filter_text_cols(left_cols, left_price_col)
        right_text_cols = filter_text_cols(right_cols, right_price_col)

        out = df.copy()
        out["left_text"] = _build_text_from_prefixed_columns(out, cfg.left_prefix, left_text_cols, cfg)
        out["right_text"] = _build_text_from_prefixed_columns(out, cfg.right_prefix, right_text_cols, cfg)

        # normalize prices to standard column names if present
        if left_price_col is not None and "left_price" not in out.columns:
            out["left_price"] = pd.to_numeric(out[left_price_col], errors="coerce").fillna(0.0)
        if right_price_col is not None and "right_price" not in out.columns:
            out["right_price"] = pd.to_numeric(out[right_price_col], errors="coerce").fillna(0.0)

    return out



def make_pair_embedding(eL: np.ndarray, eR: np.ndarray) -> np.ndarray:
    """
    eL, eR: [D]
    returns: [4D] = [eL, eR, |eL-eR|, eL*eR]
    """
    return np.concatenate([eL, eR, np.abs(eL - eR), eL * eR], axis=0).astype(np.float32)


def build_pair_embeddings_from_textcols(
    df: pd.DataFrame,
    embedder,
    left_col: str = "left_text",
    right_col: str = "right_text",
    add_price_feature: bool = False,
    left_price_col: str = "left_price",
    right_price_col: str = "right_price",
) -> np.ndarray:
    """
    Expects df with columns left_col/right_col containing text.
    Optionally adds |left_price-right_price| as extra feature.
    Returns: np.float32 [N, 4*D (+1)]
    """
    if left_col not in df.columns or right_col not in df.columns:
        raise ValueError(f"Missing text columns: {left_col}, {right_col}")

    left_texts = df[left_col].fillna("").astype(str).tolist()
    right_texts = df[right_col].fillna("").astype(str).tolist()

    eL = embedder.encode(left_texts).astype(np.float32)   # [N,D]
    eR = embedder.encode(right_texts).astype(np.float32)  # [N,D]

    if eL.shape != eR.shape:
        raise ValueError(f"Embedding shapes differ: {eL.shape} vs {eR.shape}")

    N, D = eL.shape
    pair = np.empty((N, 4 * D), dtype=np.float32)
    for i in range(N):
        pair[i] = make_pair_embedding(eL[i], eR[i])

    if add_price_feature and left_price_col in df.columns and right_price_col in df.columns:
        pL = pd.to_numeric(df[left_price_col], errors="coerce").fillna(0.0).to_numpy(np.float32)
        pR = pd.to_numeric(df[right_price_col], errors="coerce").fillna(0.0).to_numpy(np.float32)
        price_diff = np.abs(pL - pR).reshape(-1, 1).astype(np.float32)
        pair = np.concatenate([pair, price_diff], axis=1)

    return pair



def load_or_create_pair_embeddings(
    df: pd.DataFrame,
    embedder,
    embedder_name: str,
    split_name: str,
    cache_dir: str = "cache",
    # if your df already has left_text/right_text you can keep defaults
    left_col: str = "left_text",
    right_col: str = "right_text",
    add_price_feature: Optional[bool] = None,  # None => auto
    left_price_col: str = "left_price",
    right_price_col: str = "right_price",
    force_recompute: bool = False,
    # auto-text build options:
    auto_text_cfg: AutoTextConfig = AutoTextConfig(),
) -> np.ndarray:
    """
    Works with:
      - standardized df containing left_text/right_text
      - raw df containing any left_* / right_* schema (your 4 datasets)
    Caches embeddings to .npy for speed.

    IMPORTANT: cache key depends on embedder + split + auto_text_cfg
    """
    os.makedirs(cache_dir, exist_ok=True)

    # make cache key safe and unique w.r.t. text-building settings
    cfg_tag = f"keys{int(auto_text_cfg.include_keys)}_sep{len(auto_text_cfg.sep)}_exprice{int(auto_text_cfg.exclude_price_from_text)}"
    cache_path = os.path.join(cache_dir, f"pair_emb__{split_name}__{embedder_name}__{cfg_tag}.npy")

    if (not force_recompute) and os.path.exists(cache_path):
        return np.load(cache_path)

    # ensure left_text/right_text exist
    df2 = df
    if left_col not in df2.columns or right_col not in df2.columns:
        df2 = standardize_to_left_right_text(df2, auto_text_cfg)

    # auto price feature
    if add_price_feature is None:
        add_price_feature = (left_price_col in df2.columns and right_price_col in df2.columns)

    pair = build_pair_embeddings_from_textcols(
        df=df2,
        embedder=embedder,
        left_col=left_col,
        right_col=right_col,
        add_price_feature=add_price_feature,
        left_price_col=left_price_col,
        right_price_col=right_price_col,
    )

    np.save(cache_path, pair)
    return pair
