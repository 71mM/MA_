import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
from Meta_Matcher.io.scoring_loader import load_scores, DEFAULT_SCORE_COLS
from Meta_Matcher.datasets.auto_adapter import AutoAdapterConfig, standardize_raw_auto

from Meta_Matcher.embedders.minilm import MiniLMEmbedder
from Meta_Matcher.embedders.glove import GloveEmbedder
from Meta_Matcher.embedders.fasttext import FastTextEmbedder
from Meta_Matcher.embedders.pair import load_or_create_pair_embeddings

from Meta_Matcher.dataset import PairScoreDataset
from Meta_Matcher.config import MetaMatcherConfig
from Meta_Matcher.train.trainer import train


def build_embedder(embedder_type: str, device: str = None, **paths):
    print(embedder_type)
    if embedder_type == "minilm":
        emb = MiniLMEmbedder(device=device)
        return emb, "all-MiniLM-L6-v2"
    if embedder_type == "glove":
        emb = GloveEmbedder(glove_w2v_path=paths["glove_path"], binary=paths.get("glove_binary", False))
        return emb, "glove"
    if embedder_type == "fasttext":
        emb = FastTextEmbedder(path=paths["fasttext_path"], mode=paths.get("fasttext_mode", "native"))
        return emb, "fasttext"
    raise ValueError("embedder_type must be: minilm | glove | fasttext")



def merge_scores_and_raw(scores_path: str, raw_path: str, adapter_cfg: AutoAdapterConfig) -> pd.DataFrame:
    # scores: id,label,sif,rnn,attention,hybrid
    df_scores = load_scores(scores_path, id_col="id", label_col="label", score_cols=DEFAULT_SCORE_COLS)

    # raw: id,label,left_*,right_*
    df_raw = pd.read_csv(raw_path)
    df_raw_std = standardize_raw_auto(df_raw, adapter_cfg)


    df_raw_std = df_raw_std.drop(columns=["label"], errors="ignore")

    df = df_scores.merge(df_raw_std, on="id", how="inner")

    if len(df) == 0:
        raise ValueError(f"Merge ergab 0 Zeilen!\nScores: {scores_path}\nRaw: {raw_path}\nPrüfe ob IDs matchen.")
    if "label" not in df.columns:
        raise ValueError(f"Label fehlt nach Merge. Spalten: {list(df.columns)}")

    for c in ["left_text", "right_text"]:
        if c not in df.columns:
            raise ValueError(f"Spalte '{c}' fehlt nach Standardisierung/Merge. Spalten: {list(df.columns)}")

    return df


def main():
    dataset_name = "ab"


    train_scores_path = "../../data/abt-buy_textual/big_scores_train_ab_binarized.csv"
    train_raw_path    = "../../data/abt-buy_textual/train_full.csv"


    val_scores_path = "../../data/abt-buy_textual/big_scores_validation_ab_binarized.csv"
    val_raw_path    = "../../data/abt-buy_textual/validation_full.csv"

    test_scores_path = "../../data/abt-buy_textual/big_scores_test_ab_binarized.csv"
    test_raw_path    = "../../data/abt-buy_textual/test_full.csv"

    embedder_type = "minilm"  # minilm | glove | fasttext
    cache_dir = f"cache/{dataset_name}/{embedder_type}"
    out_dir   = f"runs/{dataset_name}_{embedder_type}"

    glove_path = "../embedders/embeddings/glove/glove.840B.300d.w2v.txt"
    fasttext_path = "../embedders/embeddings/fasttext/cc.en.300.bin"


    adapter_cfg = AutoAdapterConfig(id_col="id", label_col="label")


    df_tr = merge_scores_and_raw(train_scores_path, train_raw_path, adapter_cfg)
    df_va = merge_scores_and_raw(val_scores_path,   val_raw_path,   adapter_cfg)
    df_te = merge_scores_and_raw(test_scores_path,  test_raw_path,  adapter_cfg)

    print("rows train/val/test:", len(df_tr), len(df_va), len(df_te))


    embedder, embedder_name = build_embedder(
        embedder_type,
        device="cuda" if torch.cuda.is_available() else "cpu",
        glove_path=glove_path,
        fasttext_path=fasttext_path,
        fasttext_mode="native",
    )



    emb_tr = load_or_create_pair_embeddings(
        df_tr, embedder, embedder_name=embedder_name, split_name="train",
        cache_dir=cache_dir, left_col="left_text", right_col="right_text"
    )
    emb_va = load_or_create_pair_embeddings(
        df_va, embedder, embedder_name=embedder_name, split_name="val",
        cache_dir=cache_dir, left_col="left_text", right_col="right_text"
    )
    emb_te = load_or_create_pair_embeddings(
        df_te, embedder, embedder_name=embedder_name, split_name="test",
        cache_dir=cache_dir, left_col="left_text", right_col="right_text"
    )


    score_cols = DEFAULT_SCORE_COLS

    Xs_tr = df_tr[score_cols].to_numpy(np.float32)
    y_tr  = df_tr["label"].to_numpy()

    Xs_va = df_va[score_cols].to_numpy(np.float32)
    y_va  = df_va["label"].to_numpy()

    Xs_te = df_te[score_cols].to_numpy(np.float32)
    y_te  = df_te["label"].to_numpy()

    # loaders
    ds_tr = PairScoreDataset(Xs_tr, emb_tr, y_tr)
    ds_va = PairScoreDataset(Xs_va, emb_va, y_va)
    ds_te = PairScoreDataset(Xs_te, emb_te, y_te)


    cfg = MetaMatcherConfig(
        n_models=len(score_cols),
        emb_dim=int(emb_tr.shape[1]),
        arch="mlp",
        use_attention=True,
        output="softmax",
        epochs=15,
        best_metric="test_f1",
    )

    train_loader = DataLoader(ds_tr, batch_size=1, shuffle=True)
    val_loader   = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False)


    result = train(cfg, train_loader, val_loader, test_loader, out_dir=out_dir)
    print("Best:", result["best"])


if __name__ == "__main__":
    main()
