import os
import json
import numpy as np
import pandas as pd
import optuna
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader

from Meta_Matcher.io.scoring_loader import load_scores, DEFAULT_SCORE_COLS
from Meta_Matcher.datasets.auto_adapter import AutoAdapterConfig, standardize_raw_auto

from Meta_Matcher.embedders.minilm import MiniLMEmbedder
from Meta_Matcher.embedders.glove import GloveEmbedder
from Meta_Matcher.embedders.fasttext import FastTextEmbedder
from Meta_Matcher.embedders.pair import load_or_create_pair_embeddings, AutoTextConfig

from Meta_Matcher.dataset import PairScoreDataset
from Meta_Matcher.config import MetaMatcherConfig
from Meta_Matcher.train.trainer import train, evaluate



DATASETS = [
    {
        "name": "abt-buy_textual",
        "train_scores": "../../data/abt-buy_textual/big_scores_train_ab.csv",
        "train_raw":    "../../data/abt-buy_textual/train_full.csv",
        "val_scores":   "../../data/abt-buy_textual/big_scores_validation_ab.csv",
        "val_raw":      "../../data/abt-buy_textual/validation_full.csv",
        "test_scores":  "../../data/abt-buy_textual/big_scores_test_ab.csv",
        "test_raw":     "../../data/abt-buy_textual/test_full.csv",
    },
    {
        "name": "amazon-google_structured",
        "train_scores": "../../data/Amazon-Google_structured/big_scores_train_ag.csv",
        "train_raw":    "../../data/Amazon-Google_structured/train_full.csv",
        "val_scores":   "../../data/Amazon-Google_structured/big_scores_validation_ag.csv",
        "val_raw":      "../../data/Amazon-Google_structured/validation_full.csv",
        "test_scores":  "../../data/Amazon-Google_structured/big_scores_test_ag.csv",
        "test_raw":     "../../data/Amazon-Google_structured/test_full.csv",
    },
{
        "name": "itunes-amazon",
        "train_scores": "../../data/Itunes-Amazon/big_scores_train_ia.csv",
        "train_raw":    "../../data/Itunes-Amazon/train_full.csv",
        "val_scores":   "../../data/Itunes-Amazon/big_scores_validation_ia.csv",
        "val_raw":      "../../data/Itunes-Amazon/validation_full.csv",
        "test_scores":  "../../data/Itunes-Amazon/big_scores_test_ia.csv",
        "test_raw":     "../../data/Itunes-Amazon/test_full.csv",
    },
{
        "name": "walmart-amazon_dirty",
        "train_scores": "../../data/Walmart-Amazon_dity/big_scores_train_wm.csv",
        "train_raw":    "../../data/Walmart-Amazon_dity/train_full.csv",
        "val_scores":   "../../data/Walmart-Amazon_dity/big_scores_validation_wm.csv",
        "val_raw":      "../../data/Walmart-Amazon_dity/validation_full.csv",
        "test_scores":  "../../data/Walmart-Amazon_dity/big_scores_test_wm.csv",
        "test_raw":     "../../data/Walmart-Amazon_dity/test_full.csv",
    },

]


# ============================================================
# 2) Embedding Setup
# ============================================================
def build_embedder(embedder_type: str, device: str, glove_path: str = None, fasttext_path: str = None):
    if embedder_type == "minilm":
        return MiniLMEmbedder(device=device), "all-MiniLM-L6-v2"
    if embedder_type == "glove":
        if not glove_path:
            raise ValueError("glove_path fehlt (für embedder_type='glove').")
        return GloveEmbedder(glove_w2v_path=glove_path, binary=False), "glove"
    if embedder_type == "fasttext":
        if not fasttext_path:
            raise ValueError("fasttext_path fehlt (für embedder_type='fasttext').")
        return FastTextEmbedder(path=fasttext_path, mode="native"), "fasttext"
    raise ValueError("embedder_type must be one of: minilm | glove | fasttext")


def merge_scores_and_raw(scores_path: str, raw_path: str, adapter_cfg: AutoAdapterConfig) -> pd.DataFrame:
    df_scores = load_scores(scores_path, id_col="id", label_col="label", score_cols=DEFAULT_SCORE_COLS)
    df_raw = pd.read_csv(raw_path)

    df_raw_std = standardize_raw_auto(df_raw, adapter_cfg)

    # label aus scores ist source-of-truth -> raw label droppen
    df_raw_std = df_raw_std.drop(columns=["label"], errors="ignore")

    df = df_scores.merge(df_raw_std, on="id", how="inner")

    if len(df) == 0:
        raise ValueError(f"Merge ergab 0 Zeilen! Prüfe IDs.\nScores: {scores_path}\nRaw: {raw_path}")

    # Sicherstellen, dass Texte existieren
    if "left_text" not in df.columns or "right_text" not in df.columns:
        raise ValueError(f"left_text/right_text fehlen nach Standardisierung. Spalten: {list(df.columns)}")

    # Label muss aus scores kommen
    if "label" not in df.columns:
        raise ValueError(f"label fehlt nach Merge. Spalten: {list(df.columns)}")

    return df


def prepare_data_for_dataset(
    dataset_cfg: Dict[str, str],
    embedder_type: str,
    embedder_device: str,
    glove_path: str = None,
    fasttext_path: str = None,
    cache_root: str = "cache_hpo",
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,  # Xs_tr, y_tr, emb_tr
    np.ndarray, np.ndarray, np.ndarray,  # Xs_va, y_va, emb_va
    np.ndarray, np.ndarray, np.ndarray,  # Xs_te, y_te, emb_te
    int,                                  # emb_dim
]:
    adapter_cfg = AutoAdapterConfig(id_col="id", label_col="label")

    df_tr = merge_scores_and_raw(dataset_cfg["train_scores"], dataset_cfg["train_raw"], adapter_cfg)
    df_va = merge_scores_and_raw(dataset_cfg["val_scores"], dataset_cfg["val_raw"], adapter_cfg)
    df_te = merge_scores_and_raw(dataset_cfg["test_scores"], dataset_cfg["test_raw"], adapter_cfg)

    embedder, embedder_name = build_embedder(embedder_type, device=embedder_device,
                                             glove_path=glove_path, fasttext_path=fasttext_path)

    # Text-Konfig für robusten Text (key: value | ...)
    auto_text_cfg = AutoTextConfig(include_keys=True)

    cache_dir = os.path.join(cache_root, dataset_cfg["name"], embedder_type)

    # Preis-Feature: automatisch in pair.py (wenn left_price/right_price existieren)
    emb_tr = load_or_create_pair_embeddings(df_tr, embedder, embedder_name, "train",
                                           cache_dir=cache_dir, auto_text_cfg=auto_text_cfg)
    emb_va = load_or_create_pair_embeddings(df_va, embedder, embedder_name, "val",
                                           cache_dir=cache_dir, auto_text_cfg=auto_text_cfg)
    emb_te = load_or_create_pair_embeddings(df_te, embedder, embedder_name, "test",
                                           cache_dir=cache_dir, auto_text_cfg=auto_text_cfg)

    score_cols = DEFAULT_SCORE_COLS
    Xs_tr = df_tr[score_cols].to_numpy(np.float32)
    y_tr = df_tr["label"].to_numpy()
    Xs_va = df_va[score_cols].to_numpy(np.float32)
    y_va = df_va["label"].to_numpy()
    Xs_te = df_te[score_cols].to_numpy(np.float32)
    y_te = df_te["label"].to_numpy()

    emb_dim = int(emb_tr.shape[1])
    return Xs_tr, y_tr, emb_tr, Xs_va, y_va, emb_va, Xs_te, y_te, emb_te, emb_dim


# ============================================================
# 3) Optuna Objective (pro Datensatz)
# ============================================================
def suggest_cfg(trial: optuna.Trial, n_models: int, emb_dim: int, device: str) -> MetaMatcherConfig:
    arch = trial.suggest_categorical("arch", ["mlp", "lstm"])  # rnn meistens schwächer als lstm
    use_attention = trial.suggest_categorical("use_attention", [True, False])

    output = trial.suggest_categorical("output", ["sigmoid", "softmax"])
    n_classes = 2

    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    cfg = MetaMatcherConfig(
        n_models=n_models,
        emb_dim=emb_dim,
        arch=arch,
        use_attention=use_attention,
        dropout=dropout,
        output=output,
        n_classes=n_classes,
        lr=lr,
        weight_decay=weight_decay,
        epochs=trial.suggest_int("epochs", 10, 80),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 48, 64, 128]),
        best_metric="val_f1",
        device=device,
    )

    if arch == "mlp":
        cfg.mlp_layers = trial.suggest_int("mlp_layers", 1, 4)
        cfg.mlp_hidden = trial.suggest_categorical("mlp_hidden", [128, 256, 512, 768])
    else:
        cfg.rnn_hidden = trial.suggest_categorical("rnn_hidden", [64, 128, 256])
        cfg.rnn_layers = trial.suggest_int("rnn_layers", 1, 2)
        cfg.bidirectional = trial.suggest_categorical("bidirectional", [False, True])

    if use_attention:
        cfg.attn_hidden = trial.suggest_categorical("attn_hidden", [32, 64, 128])

    return cfg


def run_hpo_for_dataset(
    dataset_cfg: Dict[str, str],
    embedder_type: str = "minilm",
    glove_path: str = None,
    fasttext_path: str = None,
    n_trials: int = 30,
    seed: int = 42,
    runs_root: str = "runs_hpo",
    cache_root: str = "cache_hpo",
):
    os.makedirs(runs_root, exist_ok=True)
    dataset_name = dataset_cfg["name"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Precompute/load embeddings once per dataset+embedder (cached)
    Xs_tr, y_tr, emb_tr, Xs_va, y_va, emb_va, Xs_te, y_te, emb_te, emb_dim = prepare_data_for_dataset(
        dataset_cfg=dataset_cfg,
        embedder_type=embedder_type,
        embedder_device=device,
        glove_path=glove_path,
        fasttext_path=fasttext_path,
        cache_root=cache_root,
    )

    ds_tr = PairScoreDataset(Xs_tr, emb_tr, y_tr)
    ds_va = PairScoreDataset(Xs_va, emb_va, y_va)
    ds_te = PairScoreDataset(Xs_te, emb_te, y_te)

    # Optuna storage per dataset -> fortsetzbar
    out_dir = os.path.join(runs_root, dataset_name, embedder_type)
    os.makedirs(out_dir, exist_ok=True)
    storage_path = f"sqlite:///{os.path.join(out_dir, 'study.db')}"

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=storage_path,
        study_name=f"{dataset_name}_{embedder_type}",
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_cfg(trial, n_models=Xs_tr.shape[1], emb_dim=emb_dim, device=device)

        train_loader = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

        trial_dir = os.path.join(out_dir, f"trial_{trial.number:04d}")
        result = train(cfg, train_loader, val_loader, test_loader, out_dir=trial_dir)

        # best val metrics stored in result["best"]["metrics"]
        best_val = result["best"]["metrics"]
        score = best_val.get("f1", float("nan"))
        if np.isnan(score):
            # falls softmax and F1 nicht berechnet wurde -> dann val_acc nehmen
            score = best_val.get("acc", float("-inf"))

        # log additional info
        trial.set_user_attr("best_epoch", result["best"]["epoch"])
        trial.set_user_attr("best_val_metrics", best_val)
        return float(score)

    study.optimize(objective, n_trials=n_trials)

    # Save best params summary
    best = study.best_trial
    best_params_path = os.path.join(out_dir, "best_params.json")
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "embedder_type": embedder_type,
                "best_value": best.value,
                "best_params": best.params,
                "best_user_attrs": best.user_attrs,
            },
            f,
            indent=2,
        )

    print(f"\n=== DONE: {dataset_name} ({embedder_type}) ===")
    print("Best value:", best.value)
    print("Best params:", best.params)
    print("Saved:", best_params_path)



def main():

    EMBEDDERS = ["minilm", "glove", "fasttext"]


    GLOVE_PATH = "../embedders/embeddings/glove/glove.840B.300d.w2v.txt"
    FASTTEXT_PATH = "../embedders/embeddings/fasttext/cc.en.300.bin"

    N_TRIALS_PER_DATASET = 30

    summary_rows = []

    for ds in DATASETS:
        for emb_type in EMBEDDERS:
            print(f"\n\n==============================")
            print(f"DATASET: {ds['name']} | EMBEDDER: {emb_type}")
            print(f"==============================\n")

            run_hpo_for_dataset(
                dataset_cfg=ds,
                embedder_type=emb_type,
                glove_path=GLOVE_PATH if emb_type == "glove" else None,
                fasttext_path=FASTTEXT_PATH if emb_type == "fasttext" else None,
                n_trials=N_TRIALS_PER_DATASET,
                seed=42,
                runs_root="runs_hpo",
                cache_root="cache_hpo",
            )


            out_dir = os.path.join("runs_hpo", ds["name"], emb_type)
            best_params_path = os.path.join(out_dir, "best_params.json")
            if os.path.exists(best_params_path):
                with open(best_params_path, "r", encoding="utf-8") as f:
                    bp = json.load(f)
                summary_rows.append({
                    "dataset": ds["name"],
                    "embedder": emb_type,
                    "best_value": bp.get("best_value", None),
                    "arch": bp.get("best_params", {}).get("arch", None),
                    "use_attention": bp.get("best_params", {}).get("use_attention", None),
                    "output": bp.get("best_params", {}).get("output", None),
                    "lr": bp.get("best_params", {}).get("lr", None),
                    "weight_decay": bp.get("best_params", {}).get("weight_decay", None),
                    "dropout": bp.get("best_params", {}).get("dropout", None),
                    "batch_size": bp.get("best_params", {}).get("batch_size", None),
                    "epochs": bp.get("best_params", {}).get("epochs", None),
                })


    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        os.makedirs("runs_hpo", exist_ok=True)
        df_sum.to_csv(os.path.join("runs_hpo", "summary_best_params.csv"), index=False)
        print("\n✅ Summary gespeichert in runs_hpo/summary_best_params.csv\n")


if __name__ == "__main__":
    main()
