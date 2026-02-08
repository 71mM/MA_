import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader

from Meta_Matcher.config import MetaMatcherConfig
from Meta_Matcher.dataset import PairScoreDataset
from Meta_Matcher.datasets.auto_adapter import AutoAdapterConfig, standardize_raw_auto
from Meta_Matcher.embedders.fasttext import FastTextEmbedder
from Meta_Matcher.embedders.glove import GloveEmbedder
from Meta_Matcher.embedders.minilm import MiniLMEmbedder
from Meta_Matcher.embedders.pair import AutoTextConfig, load_or_create_pair_embeddings
from Meta_Matcher.io.scoring_loader import DEFAULT_SCORE_COLS, load_scores
from Meta_Matcher.train.trainer import train


DEFAULT_DATASETS: List[Dict[str, str]] = [
    {
        "name": "abt-buy_textual",
        "train_scores": "../../data/abt-buy_textual/big_scores_train_ab.csv",
        "train_raw": "../../data/abt-buy_textual/train_full.csv",
        "val_scores": "../../data/abt-buy_textual/big_scores_validation_ab.csv",
        "val_raw": "../../data/abt-buy_textual/validation_full.csv",
        "test_scores": "../../data/abt-buy_textual/big_scores_test_ab.csv",
        "test_raw": "../../data/abt-buy_textual/test_full.csv",
    },
    {
        "name": "amazon-google_structured",
        "train_scores": "../../data/Amazon-Google_structured/big_scores_train_ag.csv",
        "train_raw": "../../data/Amazon-Google_structured/train_full.csv",
        "val_scores": "../../data/Amazon-Google_structured/big_scores_validation_ag.csv",
        "val_raw": "../../data/Amazon-Google_structured/validation_full.csv",
        "test_scores": "../../data/Amazon-Google_structured/big_scores_test_ag.csv",
        "test_raw": "../../data/Amazon-Google_structured/test_full.csv",
    },
    {
        "name": "itunes-amazon",
        "train_scores": "../../data/Itunes-Amazon/big_scores_train_ia.csv",
        "train_raw": "../../data/Itunes-Amazon/train_full.csv",
        "val_scores": "../../data/Itunes-Amazon/big_scores_validation_ia.csv",
        "val_raw": "../../data/Itunes-Amazon/validation_full.csv",
        "test_scores": "../../data/Itunes-Amazon/big_scores_test_ia.csv",
        "test_raw": "../../data/Itunes-Amazon/test_full.csv",
    },
    {
        "name": "walmart-amazon_dirty",
        "train_scores": "../../data/Walmart-Amazon_dirty/big_scores_train_wm.csv",
        "train_raw": "../../data/Walmart-Amazon_dirty/train_full.csv",
        "val_scores": "../../data/Walmart-Amazon_dirty/big_scores_validation_wm.csv",
        "val_raw": "../../data/Walmart-Amazon_dirty/validation_full.csv",
        "test_scores": "../../data/Walmart-Amazon_dirty/big_scores_test_wm.csv",
        "test_raw": "../../data/Walmart-Amazon_dirty/test_full.csv",
    },
]


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((_script_dir() / p).resolve())


def _resolve_walmart_path(path_str: str) -> str:
    candidates = [path_str]
    if "_dirty" in path_str:
        candidates.append(path_str.replace("_dirty", "_dity"))
    if "_dity" in path_str:
        candidates.append(path_str.replace("_dity", "_dirty"))

    for c in candidates:
        rp = Path(_resolve_path(c))
        if rp.exists():
            return str(rp)
    return _resolve_path(path_str)


def normalize_dataset_paths(dataset_cfg: Dict[str, str]) -> Dict[str, str]:
    out = dict(dataset_cfg)
    for key in ["train_scores", "train_raw", "val_scores", "val_raw", "test_scores", "test_raw"]:
        raw = out[key]
        out[key] = _resolve_walmart_path(raw) if "Walmart-Amazon" in raw else _resolve_path(raw)
    return out


def validate_dataset_paths(dataset_cfg: Dict[str, str]) -> None:
    missing = [k for k in ["train_scores", "train_raw", "val_scores", "val_raw", "test_scores", "test_raw"] if not Path(dataset_cfg[k]).exists()]
    if missing:
        detail = "\n".join([f"- {k}: {dataset_cfg[k]}" for k in missing])
        raise FileNotFoundError(f"Missing dataset files for {dataset_cfg['name']}:\n{detail}")


def build_embedder(embedder_type: str, device: str, glove_path: Optional[str] = None, fasttext_path: Optional[str] = None):
    if embedder_type == "minilm":
        return MiniLMEmbedder(device=device), "all-MiniLM-L6-v2"
    if embedder_type == "glove":
        if not glove_path:
            raise ValueError("glove_path fehlt für embedder_type='glove'.")
        return GloveEmbedder(glove_w2v_path=glove_path, binary=False), "glove"
    if embedder_type == "fasttext":
        if not fasttext_path:
            raise ValueError("fasttext_path fehlt für embedder_type='fasttext'.")
        return FastTextEmbedder(path=fasttext_path, mode="native"), "fasttext"
    raise ValueError("embedder_type must be one of: minilm | glove | fasttext")


def merge_scores_and_raw(scores_path: str, raw_path: str, adapter_cfg: AutoAdapterConfig) -> pd.DataFrame:
    df_scores = load_scores(scores_path, id_col="id", label_col="label", score_cols=DEFAULT_SCORE_COLS)
    df_raw = pd.read_csv(raw_path)
    df_raw_std = standardize_raw_auto(df_raw, adapter_cfg).drop(columns=["label"], errors="ignore")

    df = df_scores.merge(df_raw_std, on="id", how="inner")
    if df.empty:
        raise ValueError(f"Merge ergab 0 Zeilen.\nScores: {scores_path}\nRaw: {raw_path}")
    if "label" not in df.columns:
        raise ValueError("label fehlt nach Merge")
    if "left_text" not in df.columns or "right_text" not in df.columns:
        raise ValueError("left_text/right_text fehlen nach Standardisierung")
    return df


def prepare_data_for_dataset(
    dataset_cfg: Dict[str, str],
    embedder_type: str,
    embedder_device: str,
    glove_path: Optional[str] = None,
    fasttext_path: Optional[str] = None,
    cache_root: str = "cache_hpo",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    adapter_cfg = AutoAdapterConfig(id_col="id", label_col="label")

    df_tr = merge_scores_and_raw(dataset_cfg["train_scores"], dataset_cfg["train_raw"], adapter_cfg)
    df_va = merge_scores_and_raw(dataset_cfg["val_scores"], dataset_cfg["val_raw"], adapter_cfg)
    df_te = merge_scores_and_raw(dataset_cfg["test_scores"], dataset_cfg["test_raw"], adapter_cfg)

    embedder, embedder_name = build_embedder(
        embedder_type,
        device=embedder_device,
        glove_path=glove_path,
        fasttext_path=fasttext_path,
    )

    auto_text_cfg = AutoTextConfig(include_keys=True)
    cache_dir = os.path.join(cache_root, dataset_cfg["name"], embedder_type)

    emb_tr = load_or_create_pair_embeddings(df_tr, embedder, embedder_name, "train", cache_dir=cache_dir, auto_text_cfg=auto_text_cfg)
    emb_va = load_or_create_pair_embeddings(df_va, embedder, embedder_name, "val", cache_dir=cache_dir, auto_text_cfg=auto_text_cfg)
    emb_te = load_or_create_pair_embeddings(df_te, embedder, embedder_name, "test", cache_dir=cache_dir, auto_text_cfg=auto_text_cfg)

    score_cols = DEFAULT_SCORE_COLS
    Xs_tr, y_tr = df_tr[score_cols].to_numpy(np.float32), df_tr["label"].to_numpy(np.int64)
    Xs_va, y_va = df_va[score_cols].to_numpy(np.float32), df_va["label"].to_numpy(np.int64)
    Xs_te, y_te = df_te[score_cols].to_numpy(np.float32), df_te["label"].to_numpy(np.int64)

    emb_dim = int(emb_tr.shape[1])
    return Xs_tr, y_tr, emb_tr, Xs_va, y_va, emb_va, Xs_te, y_te, emb_te, emb_dim


def suggest_cfg(trial: optuna.Trial, n_models: int, emb_dim: int, device: str) -> MetaMatcherConfig:
    arch = trial.suggest_categorical("arch", ["mlp", "lstm"])
    output = trial.suggest_categorical("output", ["sigmoid", "softmax"])
    use_attention = trial.suggest_categorical("use_attention", [True, False])

    cfg = MetaMatcherConfig(
        n_models=n_models,
        emb_dim=emb_dim,
        arch=arch,
        use_attention=use_attention,
        output=output,
        n_classes=2,
        lr=trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        epochs=trial.suggest_int("epochs", 10, 60),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        best_metric="val_f1",
        device=device,
        scheduler="plateau",
        early_stopping_patience=5,
        grad_clip_norm=1.0,
        activation=trial.suggest_categorical("activation", ["relu", "gelu"]),
        calibration_mode=trial.suggest_categorical("calibration_mode", ["full", "diagonal"]),
        seed=42,
    )

    if arch == "mlp":
        cfg.mlp_layers = trial.suggest_int("mlp_layers", 1, 3)
        cfg.mlp_hidden = trial.suggest_categorical("mlp_hidden", [128, 256, 512])
    else:
        cfg.rnn_hidden = trial.suggest_categorical("rnn_hidden", [64, 128, 256])
        cfg.rnn_layers = trial.suggest_int("rnn_layers", 1, 2)
        cfg.bidirectional = trial.suggest_categorical("bidirectional", [False, True])

    if use_attention:
        cfg.attn_hidden = trial.suggest_categorical("attn_hidden", [32, 64, 128])
        cfg.attn_mode = trial.suggest_categorical("attn_mode", ["gating", "token"])

    return cfg


def run_hpo_for_dataset(
    dataset_cfg: Dict[str, str],
    embedder_type: str,
    glove_path: Optional[str],
    fasttext_path: Optional[str],
    n_trials: int,
    seed: int,
    runs_root: str,
    cache_root: str,
) -> Dict[str, Any]:
    dataset_name = dataset_cfg["name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xs_tr, y_tr, emb_tr, Xs_va, y_va, emb_va, Xs_te, y_te, emb_te, emb_dim = prepare_data_for_dataset(
        dataset_cfg=dataset_cfg,
        embedder_type=embedder_type,
        embedder_device=device,
        glove_path=glove_path,
        fasttext_path=fasttext_path,
        cache_root=cache_root,
    )

    ds_tr, ds_va, ds_te = PairScoreDataset(Xs_tr, emb_tr, y_tr), PairScoreDataset(Xs_va, emb_va, y_va), PairScoreDataset(Xs_te, emb_te, y_te)

    out_dir = os.path.join(runs_root, dataset_name, embedder_type)
    os.makedirs(out_dir, exist_ok=True)
    storage_path = f"sqlite:///{os.path.join(out_dir, 'study.db')}"

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
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

        best_metrics = result["best"]["metrics"]
        best_val = best_metrics.get("val", {})
        score = best_val.get("f1", float("nan"))
        if isinstance(score, float) and np.isnan(score):
            score = best_val.get("acc", float("-inf"))

        trial.set_user_attr("best_epoch", result["best"]["epoch"])
        trial.set_user_attr("best_val_metrics", best_val)
        return float(score)

    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    best_summary = {
        "dataset": dataset_name,
        "embedder_type": embedder_type,
        "best_value": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
    }

    best_params_path = os.path.join(out_dir, "best_params.json")
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    print(f"\n=== DONE: {dataset_name} ({embedder_type}) ===")
    print("Best value:", best.value)
    print("Best params:", best.params)
    print("Saved:", best_params_path)
    return best_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HPO over all datasets and embedders.")
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials per dataset/embedder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Optuna sampler")
    parser.add_argument("--runs-root", type=str, default="runs_hpo")
    parser.add_argument("--cache-root", type=str, default="cache_hpo")
    parser.add_argument("--embedders", nargs="+", default=["minilm", "glove", "fasttext"], choices=["minilm", "glove", "fasttext"])
    parser.add_argument("--dataset", type=str, default="all", help="Dataset name from DEFAULT_DATASETS or 'all'")
    parser.add_argument("--glove-path", type=str, default="../embedders/embeddings/glove/glove.840B.300d.w2v.txt")
    parser.add_argument("--fasttext-path", type=str, default="../embedders/embeddings/fasttext/cc.en.300.bin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.runs_root, exist_ok=True)

    datasets = [normalize_dataset_paths(d) for d in DEFAULT_DATASETS]
    if args.dataset != "all":
        datasets = [d for d in datasets if d["name"] == args.dataset]
        if not datasets:
            raise ValueError(f"Unknown dataset '{args.dataset}'.")

    glove_path = _resolve_path(args.glove_path)
    fasttext_path = _resolve_path(args.fasttext_path)

    summary_rows: List[Dict[str, Any]] = []

    for ds in datasets:
        validate_dataset_paths(ds)
        for emb_type in args.embedders:
            print("\n\n==============================")
            print(f"DATASET: {ds['name']} | EMBEDDER: {emb_type}")
            print("==============================\n")

            row = run_hpo_for_dataset(
                dataset_cfg=ds,
                embedder_type=emb_type,
                glove_path=glove_path if emb_type == "glove" else None,
                fasttext_path=fasttext_path if emb_type == "fasttext" else None,
                n_trials=args.trials,
                seed=args.seed,
                runs_root=args.runs_root,
                cache_root=args.cache_root,
            )
            summary_rows.append({
                "dataset": row["dataset"],
                "embedder": row["embedder_type"],
                "best_value": row["best_value"],
                "arch": row["best_params"].get("arch"),
                "use_attention": row["best_params"].get("use_attention"),
                "output": row["best_params"].get("output"),
                "lr": row["best_params"].get("lr"),
                "weight_decay": row["best_params"].get("weight_decay"),
                "dropout": row["best_params"].get("dropout"),
                "batch_size": row["best_params"].get("batch_size"),
                "epochs": row["best_params"].get("epochs"),
            })

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        summary_path = os.path.join(args.runs_root, "summary_best_params.csv")
        df_sum.to_csv(summary_path, index=False)
        print(f"\n✅ Summary gespeichert in {summary_path}\n")


if __name__ == "__main__":
    main()
