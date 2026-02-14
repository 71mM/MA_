#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict_abt_buy_textual.py  (EXAMPLE: abt-buy_textual / "ab")

What it does:
- You hardcode all relevant paths at the top
- Loads a trained MetaMatcher checkpoint from MODEL_CKPT_PATH
- Loads test scores CSV + test raw CSV, merges by id
- Builds pair embeddings (cached) for left_text/right_text
- Builds score features:
    - optional probs -> logits
    - optional logit aggregates (mean/std/max/min/votes_pos)
- Runs inference on test set
- Writes a CSV with:
    id, y_true (if present), match_prob, pred_label, (optional) alpha_* columns

Notes:
- Works with your MetaMatcher forward() returning dict with "logits" and optional "alpha".
- For softmax output (n_classes=2) it uses P(class=1) as match_prob.
- For sigmoid output it uses sigmoid(logit) as match_prob.

Run:
    python predict_abt_buy_textual.py
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from Meta_Matcher.config import MetaMatcherConfig
from Meta_Matcher.models import MetaMatcher  # adjust if your import path differs
from Meta_Matcher.dataset import PairScoreDataset
from Meta_Matcher.datasets.auto_adapter import AutoAdapterConfig, standardize_raw_auto
from Meta_Matcher.embedders.minilm import MiniLMEmbedder
from Meta_Matcher.embedders.glove import GloveEmbedder
from Meta_Matcher.embedders.fasttext import FastTextEmbedder
from Meta_Matcher.embedders.pair import AutoTextConfig, load_or_create_pair_embeddings
from Meta_Matcher.io.scoring_loader import load_scores, DEFAULT_SCORE_COLS


# ============================================================
# 0) HARD-CODED PATHS / SETTINGS (EDIT HERE)
# ============================================================

DATASET_NAME = "dblp_scholar_exp_data"

# --- trained model checkpoint (.pt) ---
MODEL_CKPT_PATH = "best_models_from_hpo/dblp/trial_0103/best_model.pt"

# --- test data ---
TEST_SCORES_PATH = r"../../data/dblp_scholar_exp_data/big_scores_dplb__test_full.csv"
TEST_RAW_PATH    = r"../../data/dblp_scholar_exp_data/test_full.csv"

# --- output ---
OUT_DIR = r"Meta_Matcher/scripts/predictions"
OUT_CSV = os.path.join(OUT_DIR, "dblp_scholar_exp_data_test_predictions.csv")

# --- embedding / cache ---
EMBEDDER_TYPE = "minilm"
CACHE_ROOT = r"Meta_Matcher/scripts/cache_predict"

# only needed for glove/fasttext
GLOVE_PATH   = r"../embedders/embeddings/glove/glove.840B.300d.w2v.txt"
FASTTEXT_PATH = r"../embedders/embeddings/fasttext/cc.en.300.bin"

# --- feature engineering ---
ADD_LOGIT_AGGREGATES = False


# --- inference ---
BATCH_SIZE = 256
THRESHOLD = 0.5
NUM_WORKERS = 0


# ============================================================
# 1) Helpers
# ============================================================

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def prob_to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def build_embedder(embedder_type: str, device: str,
                   glove_path: Optional[str] = None,
                   fasttext_path: Optional[str] = None):
    if embedder_type == "minilm":
        return MiniLMEmbedder(device=device), "all-MiniLM-L6-v2"
    if embedder_type == "glove":
        if not glove_path:
            raise ValueError("GLOVE_PATH fehlt für embedder_type='glove'.")
        return GloveEmbedder(glove_w2v_path=glove_path, binary=False), "glove"
    if embedder_type == "fasttext":
        if not fasttext_path:
            raise ValueError("FASTTEXT_PATH fehlt für embedder_type='fasttext'.")
        return FastTextEmbedder(path=fasttext_path, mode="native"), "fasttext"
    raise ValueError("EMBEDDER_TYPE must be: minilm | glove | fasttext")


def merge_scores_and_raw(
    scores_path: str,
    raw_path: str,
    adapter_cfg: AutoAdapterConfig,
    base_score_cols: List[str],
    add_logit_aggregates: bool,
) -> pd.DataFrame:
    # scores: id,label,<base_score_cols...>
    df_scores = load_scores(scores_path, id_col="id", label_col="label", score_cols=base_score_cols)

    # raw: id,label,left_*,right_*  -> standardized to left_text/right_text
    df_raw = pd.read_csv(raw_path)
    df_raw_std = standardize_raw_auto(df_raw, adapter_cfg).drop(columns=["label"], errors="ignore")


    if add_logit_aggregates:
        L = df_scores[base_score_cols].to_numpy(np.float32)
        df_scores["logit_mean"] = L.mean(axis=1)
        df_scores["logit_std"]  = L.std(axis=1)
        df_scores["logit_max"]  = L.max(axis=1)
        df_scores["logit_min"]  = L.min(axis=1)
        df_scores["votes_pos"]  = (L > 0).sum(axis=1).astype(np.float32)

    # merge
    df = df_scores.merge(df_raw_std, on="id", how="inner")

    if df.empty:
        raise ValueError(f"Merge ergab 0 Zeilen.\nScores: {scores_path}\nRaw: {raw_path}")
    if "left_text" not in df.columns or "right_text" not in df.columns:
        raise ValueError(f"left_text/right_text fehlen nach Standardisierung. Spalten: {list(df.columns)}")

    # label may exist (typically yes)
    return df


def load_checkpoint_model(ckpt_path: str, device: str) -> Tuple[MetaMatcher, MetaMatcherConfig, Dict[str, Any]]:
    """
    Expects your checkpoint saver (save_best_checkpoint) to store something like:
      {"cfg": cfg_dict or cfg, "model_state": state_dict, ...}
    We'll handle a few common variants defensively.

    Returns: (model, cfg, raw_ckpt)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # --- cfg ---
    cfg_obj = None
    if isinstance(ckpt, dict):
        if "cfg" in ckpt:
            cfg_obj = ckpt["cfg"]
        elif "config" in ckpt:
            cfg_obj = ckpt["config"]

    if cfg_obj is None:
        raise ValueError(f"Checkpoint at {ckpt_path} has no 'cfg'/'config' field. Keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    # cfg may be dataclass, dict, or something else
    if isinstance(cfg_obj, dict):
        cfg = MetaMatcherConfig(**cfg_obj)
    else:
        cfg = cfg_obj  # assume it's already MetaMatcherConfig

    cfg.device = device  # force current device

    # --- model ---
    model = MetaMatcher(cfg).to(device)

    # --- state dict ---
    state = None
    if isinstance(ckpt, dict):
        for k in ["model_state", "state_dict", "model", "model_state_dict"]:
            if k in ckpt:
                state = ckpt[k]
                break
    if state is None:
        # maybe checkpoint itself is a plain state_dict
        if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
        else:
            raise ValueError(f"Could not find model state in checkpoint. Keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def predict(model: MetaMatcher, cfg: MetaMatcherConfig, loader: DataLoader) -> Dict[str, Any]:
    """
    Returns dict with:
      probs: (N,)
      alpha: (N, M) or None
    """
    probs_all: List[np.ndarray] = []
    alpha_all: List[np.ndarray] = []

    for scores, emb, _labels in loader:
        scores = scores.to(cfg.device)
        emb = emb.to(cfg.device)

        out = model(scores, emb)
        logits = out["logits"]

        if cfg.output == "sigmoid":
            p = torch.sigmoid(logits).squeeze(-1)
        else:
            # softmax, assume binary -> take class 1
            p = torch.softmax(logits, dim=-1)[:, 1]

        probs_all.append(p.detach().cpu().numpy())

        a = out.get("alpha", None) if isinstance(out, dict) else None
        if a is not None:
            alpha_all.append(a.detach().cpu().numpy())

    probs = np.concatenate(probs_all, axis=0) if probs_all else np.array([], dtype=np.float32)
    alpha = np.concatenate(alpha_all, axis=0) if alpha_all else None
    return {"probs": probs, "alpha": alpha}


# ============================================================
# 2) Main
# ============================================================

def main() -> None:
    # --- sanity / dirs
    _ensure_dir(OUT_DIR)
    _ensure_dir(os.path.join(CACHE_ROOT, DATASET_NAME, EMBEDDER_TYPE))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # --- base score cols
    base_cols = list(DEFAULT_SCORE_COLS)


    # --- load model checkpoint
    model, cfg_from_ckpt, ckpt = load_checkpoint_model(MODEL_CKPT_PATH, device=device)
    print("Loaded checkpoint:", MODEL_CKPT_PATH)
    print("Checkpoint cfg.output:", cfg_from_ckpt.output, "| cfg.arch:", cfg_from_ckpt.arch)

    # --- load + merge test df
    adapter_cfg = AutoAdapterConfig(id_col="id", label_col="label")
    df_te = merge_scores_and_raw(
        TEST_SCORES_PATH,
        TEST_RAW_PATH,
        adapter_cfg,
        base_score_cols=base_cols,
        add_logit_aggregates=ADD_LOGIT_AGGREGATES,
    )

    # --- feature cols (must match what we feed into model)
    feature_cols = list(base_cols)
    if ADD_LOGIT_AGGREGATES:
        feature_cols += ["logit_mean", "logit_std", "logit_max", "logit_min", "votes_pos"]

    # --- build embeddings (cached)
    embedder, embedder_name = build_embedder(
        EMBEDDER_TYPE,
        device=device,
        glove_path=GLOVE_PATH if EMBEDDER_TYPE == "glove" else None,
        fasttext_path=FASTTEXT_PATH if EMBEDDER_TYPE == "fasttext" else None,
    )
    auto_text_cfg = AutoTextConfig(include_keys=True)

    cache_dir = os.path.join(CACHE_ROOT, DATASET_NAME, EMBEDDER_TYPE)
    emb_te = load_or_create_pair_embeddings(
        df_te,
        embedder,
        embedder_name=embedder_name,
        split_name="test_predict",
        cache_dir=cache_dir,
        auto_text_cfg=auto_text_cfg,
    )

    # --- scores + labels
    Xs_te = df_te[feature_cols].to_numpy(np.float32)

    # label might not exist in some setups
    if "label" in df_te.columns:
        y_te = df_te["label"].to_numpy(np.int64)
    else:
        y_te = np.zeros((len(df_te),), dtype=np.int64)

    ds_te = PairScoreDataset(Xs_te, emb_te, y_te)
    te_loader = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)

    # --- predict
    pred = predict(model, cfg_from_ckpt, te_loader)
    probs = pred["probs"]
    alpha = pred["alpha"]

    # --- output df
    out_df = pd.DataFrame({
        "id": df_te["id"].values,
        "match_prob": probs.astype(np.float32),
        "pred_label": (probs >= THRESHOLD).astype(np.int32),
    })

    if "label" in df_te.columns:
        out_df["y_true"] = df_te["label"].astype(np.int32).values

    # optional: write alpha columns
    if alpha is not None:
        for j in range(alpha.shape[1]):
            out_df[f"alpha_{j}"] = alpha[:, j].astype(np.float32)

    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nWrote predictions: {OUT_CSV}")


if __name__ == "__main__":
    main()
