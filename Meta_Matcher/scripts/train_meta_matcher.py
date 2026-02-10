import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split

from Meta_Matcher.io.scoring_loader import load_scores, DEFAULT_SCORE_COLS
from Meta_Matcher.datasets.auto_adapter import AutoAdapterConfig, standardize_raw_auto

from Meta_Matcher.embedders.minilm import MiniLMEmbedder
from Meta_Matcher.embedders.glove import GloveEmbedder
from Meta_Matcher.embedders.fasttext import FastTextEmbedder
from Meta_Matcher.embedders.pair import load_or_create_pair_embeddings, AutoTextConfig

from Meta_Matcher.dataset import PairScoreDataset
from Meta_Matcher.config import MetaMatcherConfig
from Meta_Matcher.train.trainer import train


# ============================================================
# 1) PROBS -> LOGITS (+ optional aggregates)
# ============================================================
def prob_to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# ============================================================
# 2) Embedders
# ============================================================
def build_embedder(embedder_type: str, device: str = None, **paths):
    print("Embedder:", embedder_type)
    if embedder_type == "minilm":
        emb = MiniLMEmbedder(device=device)
        return emb, "all-MiniLM-L6-v2"

    if embedder_type == "glove":
        glove_path = paths.get("glove_path", None)
        if not glove_path:
            raise ValueError("glove_path fehlt für embedder_type='glove'")
        emb = GloveEmbedder(glove_w2v_path=glove_path, binary=paths.get("glove_binary", False))
        return emb, "glove"

    if embedder_type == "fasttext":
        ft_path = paths.get("fasttext_path", None)
        if not ft_path:
            raise ValueError("fasttext_path fehlt für embedder_type='fasttext'")
        emb = FastTextEmbedder(path=ft_path, mode=paths.get("fasttext_mode", "native"))
        return emb, "fasttext"

    raise ValueError("embedder_type must be: minilm | glove | fasttext")


# ============================================================
# 3) Merge helper (+ logits + aggregates)
# ============================================================
def merge_scores_and_raw(
    scores_path: str,
    raw_path: str,
    adapter_cfg: AutoAdapterConfig,
    use_logits: bool,
    add_logit_aggregates: bool,
) -> pd.DataFrame:
    df_scores = load_scores(scores_path, id_col="id", label_col="label", score_cols=DEFAULT_SCORE_COLS)
    score_cols = list(DEFAULT_SCORE_COLS)

    # (A) convert probs -> logits (optional)
    if use_logits:
        logits = prob_to_logit(df_scores[score_cols].to_numpy(np.float32))
        df_scores.loc[:, score_cols] = logits

    # (B) aggregates (optional)
    if add_logit_aggregates:
        L = df_scores[score_cols].to_numpy(np.float32)
        df_scores["logit_mean"] = L.mean(axis=1)
        df_scores["logit_std"]  = L.std(axis=1)
        df_scores["logit_max"]  = L.max(axis=1)
        df_scores["logit_min"]  = L.min(axis=1)
        df_scores["votes_pos"]  = (L > 0).sum(axis=1).astype(np.float32)  # logit>0 <=> prob>0.5

    df_raw = pd.read_csv(raw_path)
    df_raw_std = standardize_raw_auto(df_raw, adapter_cfg)
    df_raw_std = df_raw_std.drop(columns=["label"], errors="ignore")

    df = df_scores.merge(df_raw_std, on="id", how="inner")

    if df.empty:
        raise ValueError(f"Merge ergab 0 Zeilen!\nScores: {scores_path}\nRaw: {raw_path}")
    if "label" not in df.columns:
        raise ValueError(f"Label fehlt nach Merge. Spalten: {list(df.columns)}")
    if "left_text" not in df.columns or "right_text" not in df.columns:
        raise ValueError(f"left_text/right_text fehlen nach Standardisierung. Spalten: {list(df.columns)}")

    return df


def main():
    # ============================================================
    # SETTINGS
    # ============================================================
    dataset_name = "ab"

    # >>> IMPORTANT: meta-train-pool is VALIDATION (to avoid in-sample base leakage)
    meta_scores_path = "../../data/abt-buy_textual/big_scores_validation_ab.csv"
    meta_raw_path    = "../../data/abt-buy_textual/validation_full.csv"

    test_scores_path = "../../data/abt-buy_textual/big_scores_test_ab.csv"
    test_raw_path    = "../../data/abt-buy_textual/test_full.csv"

    embedder_type = "minilm"  # "minilm" | "glove" | "fasttext"

    glove_path    = r"C:\embeddings\glove\glove.840B.300d.w2v.txt"
    fasttext_path = r"C:\embeddings\fasttext\cc.en.300.bin"

    USE_LOGITS = True
    ADD_LOGIT_AGGREGATES = True

    # >>> MINI-VAL SPLIT INSIDE META-POOL
    MINI_VAL_FRAC = 0.2
    SEED = 42

    cache_dir = os.path.join("cache", dataset_name, embedder_type)
    out_dir   = os.path.join("runs", f"{dataset_name}_{embedder_type}_metaPoolCVsplit")

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    adapter_cfg = AutoAdapterConfig(id_col="id", label_col="label")

    # ============================================================
    # LOAD META-POOL (validation) + TEST
    # ============================================================
    df_meta = merge_scores_and_raw(
        meta_scores_path, meta_raw_path, adapter_cfg,
        use_logits=USE_LOGITS,
        add_logit_aggregates=ADD_LOGIT_AGGREGATES,
    )

    df_te = merge_scores_and_raw(
        test_scores_path, test_raw_path, adapter_cfg,
        use_logits=USE_LOGITS,
        add_logit_aggregates=ADD_LOGIT_AGGREGATES,
    )

    # ============================================================
    # MINI-VAL SPLIT (stratified)
    # ============================================================
    idx = np.arange(len(df_meta))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=MINI_VAL_FRAC,
        random_state=SEED,
        stratify=df_meta["label"].values
    )

    df_tr = df_meta.iloc[tr_idx].reset_index(drop=True)
    df_va = df_meta.iloc[va_idx].reset_index(drop=True)

    print(f"Meta-pool split: train={len(df_tr)} mini_val={len(df_va)} test={len(df_te)}")

    # ============================================================
    # BUILD EMBEDDER
    # ============================================================
    embedder, embedder_name = build_embedder(
        embedder_type,
        device=device if embedder_type == "minilm" else None,
        glove_path=glove_path,
        fasttext_path=fasttext_path,
        fasttext_mode="native",
    )

    auto_text_cfg = AutoTextConfig(include_keys=True)

    # ============================================================
    # EMBEDDINGS (cached)
    # ============================================================
    emb_tr = load_or_create_pair_embeddings(
        df_tr, embedder, embedder_name=embedder_name, split_name="meta_train",
        cache_dir=cache_dir, auto_text_cfg=auto_text_cfg
    )
    emb_va = load_or_create_pair_embeddings(
        df_va, embedder, embedder_name=embedder_name, split_name="mini_val",
        cache_dir=cache_dir, auto_text_cfg=auto_text_cfg
    )
    emb_te = load_or_create_pair_embeddings(
        df_te, embedder, embedder_name=embedder_name, split_name="test",
        cache_dir=cache_dir, auto_text_cfg=auto_text_cfg
    )


    score_cols = list(DEFAULT_SCORE_COLS)
    if ADD_LOGIT_AGGREGATES:
        score_cols += ["logit_mean", "logit_std", "logit_max", "logit_min", "votes_pos"]

    print("Meta feature cols:", score_cols)

    Xs_tr = df_tr[score_cols].to_numpy(np.float32); y_tr = df_tr["label"].to_numpy()
    Xs_va = df_va[score_cols].to_numpy(np.float32); y_va = df_va["label"].to_numpy()
    Xs_te = df_te[score_cols].to_numpy(np.float32); y_te = df_te["label"].to_numpy()

    ds_tr = PairScoreDataset(Xs_tr, emb_tr, y_tr)
    ds_va = PairScoreDataset(Xs_va, emb_va, y_va)
    ds_te = PairScoreDataset(Xs_te, emb_te, y_te)

    # ============================================================
    # CONFIG
    # ============================================================
    n_base = len(DEFAULT_SCORE_COLS)          # 4 Mudgal models
    n_extra = len(score_cols) - n_base        # 5 aggregates if enabled

    cfg = MetaMatcherConfig(
        n_models=len(score_cols),          # total feature cols passed into PairScoreDataset
        n_base_models=n_base,              # NEW (for MetaMatcher base/extra split)
        n_extra_features=n_extra,          # NEW
        base_score_input="logit" if USE_LOGITS else "prob",  # NEW

        emb_dim=int(emb_tr.shape[1]),
        arch="lstm",
        output="sigmoid",
        n_classes=2,

        use_attention=True,
        attn_mode="gating",
        attn_hidden=64,

        mlp_hidden=1000,
        mlp_layers=2,
        dropout=0.2,
        activation="relu",

        lr=1e-4,
        weight_decay=1e-5,
        batch_size=16,
        epochs=50,

        best_metric="val_f1",
        early_stopping_patience=5,

        device=device,
        debug_checks=False,
    )

    train_loader = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # ============================================================
    # TRAIN
    # ============================================================
    result = train(cfg, train_loader, val_loader, test_loader, out_dir=out_dir)
    print("Best:", result["best"])


if __name__ == "__main__":
    main()
