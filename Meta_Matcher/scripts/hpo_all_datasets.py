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
from sklearn.model_selection import train_test_split  # NEW

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
        "meta_scores": "../../data/abt-buy_textual/big_scores_validation_ab.csv",
        "meta_raw": "../../data/abt-buy_textual/validation_full.csv",
        "test_scores": "../../data/abt-buy_textual/big_scores_test_ab.csv",
        "test_raw": "../../data/abt-buy_textual/test_full.csv",
    },
    {
        "name": "amazon-google_structured",
        "meta_scores": "../../data/Amazon-Google_structured/big_scores_validation_ag.csv",
        "meta_raw": "../../data/Amazon-Google_structured/validation_full.csv",
        "test_scores": "../../data/Amazon-Google_structured/big_scores_test_ag.csv",
        "test_raw": "../../data/Amazon-Google_structured/test_full.csv",
    },
    {
        "name": "itunes-amazon",
        "meta_scores": "../../data/Itunes-Amazon/big_scores_validation_ia.csv",
        "meta_raw": "../../data/Itunes-Amazon/validation_full.csv",
        "test_scores": "../../data/Itunes-Amazon/big_scores_test_ia.csv",
        "test_raw": "../../data/Itunes-Amazon/test_full.csv",
    },
    {
        "name": "walmart-amazon_dirty",
        "meta_scores": "../../data/Walmart-Amazon_dirty/big_scores_validation_wm.csv",
        "meta_raw": "../../data/Walmart-Amazon_dirty/validation_full.csv",
        "test_scores": "../../data/Walmart-Amazon_dirty/big_scores_test_wm.csv",
        "test_raw": "../../data/Walmart-Amazon_dirty/test_full.csv",
    },
]
def prob_to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

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
    for key in ["meta_scores", "meta_raw", "test_scores", "test_raw"]:
        raw = out[key]
        out[key] = _resolve_walmart_path(raw) if "Walmart-Amazon" in raw else _resolve_path(raw)
    return out


def validate_dataset_paths(dataset_cfg: Dict[str, str]) -> None:
    missing = [
        k for k in ["meta_scores", "meta_raw", "test_scores", "test_raw"]
        if not Path(dataset_cfg[k]).exists()
    ]
    if missing:
        detail = "\n".join([f"- {k}: {dataset_cfg[k]}" for k in missing])
        raise FileNotFoundError(f"Missing dataset files for {dataset_cfg['name']}:\n{detail}")


def build_embedder(
    embedder_type: str,
    device: str,
    glove_path: Optional[str] = None,
    fasttext_path: Optional[str] = None
):
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


def merge_scores_and_raw(
    scores_path: str,
    raw_path: str,
    adapter_cfg: AutoAdapterConfig,
    use_logits: bool,
    add_logit_aggregates: bool,
) -> pd.DataFrame:
    df_scores = load_scores(scores_path, id_col="id", label_col="label", score_cols=DEFAULT_SCORE_COLS)
    score_cols = list(DEFAULT_SCORE_COLS)

    # (A) probs -> logits

    logits = prob_to_logit(df_scores[score_cols].to_numpy(np.float32))
    df_scores.loc[:, score_cols] = logits

    # (B) aggregates

    L = df_scores[score_cols].to_numpy(np.float32)
    df_scores["logit_mean"] = L.mean(axis=1)
    df_scores["logit_std"]  = L.std(axis=1)
    df_scores["logit_max"]  = L.max(axis=1)
    df_scores["logit_min"]  = L.min(axis=1)
    df_scores["votes_pos"]  = (L > 0).sum(axis=1).astype(np.float32)  # logit>0 <=> prob>0.5

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



from sklearn.model_selection import train_test_split

def prepare_data_for_dataset(
    dataset_cfg: Dict[str, str],
    embedder_type: str,
    embedder_device: str,
    glove_path: Optional[str] = None,
    fasttext_path: Optional[str] = None,
    cache_root: str = "cache_hpo",
    use_logits: bool = True,
    add_logit_aggregates: bool = True,
    mini_val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,   # Xs_tr, y_tr, emb_tr  (meta_train)
    np.ndarray, np.ndarray, np.ndarray,   # Xs_va, y_va, emb_va  (mini_val)
    np.ndarray, np.ndarray, np.ndarray,   # Xs_te, y_te, emb_te  (test)
    int, List[str]                        # emb_dim, feature_cols
]:
    adapter_cfg = AutoAdapterConfig(id_col="id", label_col="label")


    df_meta = merge_scores_and_raw(
        dataset_cfg["meta_scores"],
        dataset_cfg["meta_raw"],
        adapter_cfg,
        use_logits=use_logits,
        add_logit_aggregates=add_logit_aggregates,
    )
    df_te = merge_scores_and_raw(
        dataset_cfg["test_scores"],
        dataset_cfg["test_raw"],
        adapter_cfg,
        use_logits=use_logits,
        add_logit_aggregates=add_logit_aggregates,
    )

    # 2) split meta pool -> meta_train + mini_val
    idx = np.arange(len(df_meta))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=mini_val_frac,
        random_state=seed,
        stratify=df_meta["label"].values
    )
    df_tr = df_meta.iloc[tr_idx].reset_index(drop=True)
    df_va = df_meta.iloc[va_idx].reset_index(drop=True)

    # 3) embedder + embeddings
    embedder, embedder_name = build_embedder(
        embedder_type,
        device=embedder_device,
        glove_path=glove_path,
        fasttext_path=fasttext_path,
    )

    auto_text_cfg = AutoTextConfig(include_keys=True)
    cache_dir = os.path.join(cache_root, dataset_cfg["name"], embedder_type)

    emb_tr = load_or_create_pair_embeddings(df_tr, embedder, embedder_name, "meta_train", cache_dir=cache_dir, auto_text_cfg=auto_text_cfg)
    emb_va = load_or_create_pair_embeddings(df_va, embedder, embedder_name, "mini_val",  cache_dir=cache_dir, auto_text_cfg=auto_text_cfg)
    emb_te = load_or_create_pair_embeddings(df_te, embedder, embedder_name, "test",      cache_dir=cache_dir, auto_text_cfg=auto_text_cfg)

    feature_cols = list(DEFAULT_SCORE_COLS)
    feature_cols += ["logit_mean", "logit_std", "logit_max", "logit_min", "votes_pos"]

    Xs_tr, y_tr = df_tr[feature_cols].to_numpy(np.float32), df_tr["label"].to_numpy(np.int64)
    Xs_va, y_va = df_va[feature_cols].to_numpy(np.float32), df_va["label"].to_numpy(np.int64)
    Xs_te, y_te = df_te[feature_cols].to_numpy(np.float32), df_te["label"].to_numpy(np.int64)

    emb_dim = int(emb_tr.shape[1])
    return Xs_tr, y_tr, emb_tr, Xs_va, y_va, emb_va, Xs_te, y_te, emb_te, emb_dim, feature_cols


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

        epochs=trial.suggest_int("epochs", 10, 100),
        batch_size=trial.suggest_categorical("batch_size", [8,16, 32, 64, 128]),

        best_metric="val_f1",
        device=device,

        scheduler="plateau",
        early_stopping_patience=20,
        grad_clip_norm=1.0,

        activation=trial.suggest_categorical("activation", ["relu"]),
        calibration_mode=trial.suggest_categorical("calibration_mode", ["full", "diagonal"]),
        seed=42,
    )

    # sample these EARLY (affects token_dim)
    cfg.use_score_mask = trial.suggest_categorical("use_score_mask", [True, False])
    cfg.mask_gating = trial.suggest_categorical("mask_gating", [True, False])

    if arch == "mlp":
        cfg.mlp_layers = trial.suggest_int("mlp_layers", 1, 3)
        cfg.mlp_hidden = trial.suggest_categorical("mlp_hidden", [128, 256, 512])
    else:
        cfg.rnn_hidden = trial.suggest_categorical("rnn_hidden", [64, 128, 256])
        cfg.rnn_layers = trial.suggest_int("rnn_layers", 1, 2)
        cfg.bidirectional = trial.suggest_categorical("bidirectional", [False, True])
        cfg.rnn_pooling = trial.suggest_categorical("rnn_pooling", ["last", "mean", "max", "hidden"])

    if use_attention:
        cfg.attn_hidden = trial.suggest_categorical("attn_hidden", [32, 64, 128])
        cfg.attn_mode = trial.suggest_categorical("attn_mode", ["gating", "token", "both"])
        cfg.alpha_temperature = trial.suggest_float("alpha_temperature", 1.0, 4.0)

        cfg.alpha_entropy_weight = trial.suggest_float("alpha_entropy_weight", 0.0, 0.03)
        cfg.alpha_entropy_weight_final = trial.suggest_float("alpha_entropy_weight_final", 0.0, cfg.alpha_entropy_weight)
        cfg.alpha_entropy_anneal = trial.suggest_categorical("alpha_entropy_anneal", ["none", "linear"])

        if cfg.attn_mode in ("token", "both"):
            cfg.model_id_dim = trial.suggest_categorical("model_id_dim", [4, 8, 16])
            cfg.token_attn_heads = trial.suggest_categorical("token_attn_heads", [1, 2, 4])

            token_dim = 1 + emb_dim + cfg.model_id_dim + (1 if cfg.use_score_mask else 0)
            if token_dim % cfg.token_attn_heads != 0:
                raise optuna.TrialPruned(
                    f"Invalid token_attn_heads: token_dim={token_dim} not divisible by heads={cfg.token_attn_heads}"
                )

    cfg.score_dropout = trial.suggest_float("score_dropout", 0.0, 0.25)
    base_m = int(getattr(cfg, "n_base_models", n_models))
    cfg.score_dropout_min_keep = trial.suggest_int("score_dropout_min_keep", 1, max(1, base_m))
    cfg.missing_score_value = trial.suggest_categorical("missing_score_value", [0.0, 0.5])
    cfg.score_noise = trial.suggest_float("score_noise", 0.0, 0.02)

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
    use_logits: bool,
    add_logit_aggregates: bool,
    mini_val_frac: float
) -> Dict[str, Any]:
    dataset_name = dataset_cfg["name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xs_tr, y_tr, emb_tr, Xs_va, y_va, emb_va, Xs_te, y_te, emb_te, emb_dim, feature_cols = prepare_data_for_dataset(
        dataset_cfg=dataset_cfg,
        embedder_type=embedder_type,
        embedder_device=device,
        glove_path=glove_path,
        fasttext_path=fasttext_path,
        cache_root=cache_root,
        use_logits=use_logits,
        add_logit_aggregates=add_logit_aggregates,
        mini_val_frac=mini_val_frac,
        seed=seed,
    )

    print(f"Feature cols ({len(feature_cols)}): {feature_cols}")

    ds_tr = PairScoreDataset(Xs_tr, emb_tr, y_tr)
    ds_va = PairScoreDataset(Xs_va, emb_va, y_va)
    ds_te = PairScoreDataset(Xs_te, emb_te, y_te)

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
        total_score_dim = int(Xs_tr.shape[1])  # base + aggregates
        base_models = int(len(DEFAULT_SCORE_COLS))  # base only
        extra_feats = int(total_score_dim - base_models)

        if extra_feats < 0:
            raise RuntimeError(
                f"Score dim mismatch: Xs_tr has {total_score_dim} cols but DEFAULT_SCORE_COLS has {base_models}"
            )

        # n_models MUST be total score feature dim expected by the model
        cfg = suggest_cfg(trial, n_models=total_score_dim, emb_dim=emb_dim, device=device)

        # explicitly tell model how to split base vs extra
        cfg.n_base_models = base_models
        cfg.n_extra_features = extra_feats

        train_loader = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

        trial_dir = os.path.join(out_dir, f"trial_{trial.number:04d}")
        result = train(cfg, train_loader, val_loader, test_loader, out_dir=trial_dir)

        best_metrics = result["best"]["metrics"]
        best_val = (best_metrics or {}).get("val", {}) if isinstance(best_metrics, dict) else {}

        score = best_val.get("f1", float("nan"))
        if isinstance(score, float) and np.isnan(score):
            score = best_val.get("acc", float("-inf"))

        trial.set_user_attr("best_epoch", result["best"]["epoch"])
        trial.set_user_attr("best_val_metrics", best_val)
        trial.set_user_attr("feature_cols", feature_cols)
        trial.set_user_attr("n_models_total", total_score_dim)
        trial.set_user_attr("n_base_models", base_models)
        trial.set_user_attr("n_extra_features", extra_feats)
        return float(score)

    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    best_summary = {
        "dataset": dataset_name,
        "embedder_type": embedder_type,
        "best_value": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "use_logits": use_logits,
        "add_logit_aggregates": add_logit_aggregates,
        "feature_cols": feature_cols,
    }

    best_params_path = os.path.join(out_dir, "best_params.json")
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    print(f"\n=== DONE: {dataset_name} ({embedder_type}) ===")
    print("Best value:", best.value)
    print("Saved:", best_params_path)
    return best_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HPO over all datasets and embedders.")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-root", type=str, default="runs_hpo")
    parser.add_argument("--cache-root", type=str, default="cache_hpo")

    parser.add_argument("--use-logits", action="store_true", help="Convert probs -> logits before training")
    parser.add_argument("--add-aggregates", action="store_true", help="Add logit aggregates (mean/std/max/min/votes_pos)")

    parser.add_argument("--embedders", nargs="+", default=["minilm"], choices=["minilm"])
    parser.add_argument("--dataset", type=str, default="all")

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
            names = ", ".join(sorted({d["name"] for d in DEFAULT_DATASETS}))
            raise ValueError(f"Unknown dataset '{args.dataset}'. Available: {names}")

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
                use_logits=args.use_logits,
                add_logit_aggregates=args.add_aggregates,
                mini_val_frac=0.2
            )
            summary_rows.append({
                "dataset": row["dataset"],
                "embedder": row["embedder_type"],
                "best_value": row["best_value"],
                "arch": row["best_params"].get("arch"),
                "use_attention": row["best_params"].get("use_attention"),
                "attn_mode": row["best_params"].get("attn_mode"),
                "output": row["best_params"].get("output"),
                "lr": row["best_params"].get("lr"),
                "weight_decay": row["best_params"].get("weight_decay"),
                "dropout": row["best_params"].get("dropout"),
                "batch_size": row["best_params"].get("batch_size"),
                "epochs": row["best_params"].get("epochs"),
                "score_dropout": row["best_params"].get("score_dropout"),
                "alpha_temperature": row["best_params"].get("alpha_temperature"),
                "alpha_entropy_weight": row["best_params"].get("alpha_entropy_weight"),
            })

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        summary_path = os.path.join(args.runs_root, "summary_best_params.csv")
        df_sum.to_csv(summary_path, index=False)
        print(f"\nSummary gespeichert in {summary_path}\n")


if __name__ == "__main__":
    main()
