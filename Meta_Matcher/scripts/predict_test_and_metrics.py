import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from Meta_Matcher.datasets.auto_adapter import AutoAdapterConfig, standardize_raw_auto
from Meta_Matcher.embedders.fasttext import FastTextEmbedder
from Meta_Matcher.embedders.glove import GloveEmbedder
from Meta_Matcher.embedders.minilm import MiniLMEmbedder
from Meta_Matcher.embedders.pair import AutoTextConfig, load_or_create_pair_embeddings
from Meta_Matcher.io.checkpoints import load_model
from Meta_Matcher.io.scoring_loader import DEFAULT_SCORE_COLS, load_scores
from Meta_Matcher.train.metrics import compute_binary_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load MetaMatcher checkpoint, predict on test data, and export metrics CSV.")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--test-scores", required=True, help="CSV with id,label and base model scores")
    p.add_argument("--test-raw", required=True, help="CSV with id,label,left_*/right_* columns")
    p.add_argument("--output-dir", default="runs_inference", help="Output directory for CSV artifacts")

    p.add_argument("--embedder", choices=["minilm", "glove", "fasttext"], default="minilm")
    p.add_argument("--glove-path", default=None)
    p.add_argument("--fasttext-path", default=None)
    p.add_argument("--cache-dir", default="cache_inference")
    p.add_argument("--split-name", default="test")

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--score-cols", nargs="+", default=None, help="Override score columns")
    return p.parse_args()


def build_embedder(embedder_type: str, device: str, glove_path: Optional[str], fasttext_path: Optional[str]):
    if embedder_type == "minilm":
        return MiniLMEmbedder(device=device), "all-MiniLM-L6-v2"
    if embedder_type == "glove":
        if not glove_path:
            raise ValueError("--glove-path is required for --embedder glove")
        return GloveEmbedder(glove_w2v_path=glove_path, binary=False), "glove"
    if embedder_type == "fasttext":
        if not fasttext_path:
            raise ValueError("--fasttext-path is required for --embedder fasttext")
        return FastTextEmbedder(path=fasttext_path, mode="native"), "fasttext"
    raise ValueError(f"Unsupported embedder: {embedder_type}")


def merge_scores_and_raw(test_scores: str, test_raw: str, score_cols: List[str]) -> pd.DataFrame:
    df_scores = load_scores(test_scores, id_col="id", label_col="label", score_cols=score_cols)
    df_raw = pd.read_csv(test_raw)

    adapter_cfg = AutoAdapterConfig(id_col="id", label_col="label")
    df_raw_std = standardize_raw_auto(df_raw, adapter_cfg).drop(columns=["label"], errors="ignore")

    df = df_scores.merge(df_raw_std, on="id", how="inner")
    if df.empty:
        raise ValueError("Merged test dataframe is empty. Check matching ids between test-scores and test-raw.")
    if "label" not in df.columns:
        raise ValueError("Label column missing after merge; required to compute metrics.")
    if "left_text" not in df.columns or "right_text" not in df.columns:
        raise ValueError("left_text/right_text missing after standardization; cannot build pair embeddings.")
    return df


@torch.inference_mode()
def predict(model: torch.nn.Module, scores: np.ndarray, emb: np.ndarray, batch_size: int, device: str):
    n = scores.shape[0]
    logits_all = []
    probs_all = []
    preds_all = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        s = torch.tensor(scores[start:end], dtype=torch.float32, device=device)
        e = torch.tensor(emb[start:end], dtype=torch.float32, device=device)

        out = model(s, e)
        logits = out["logits"]

        if model.cfg.output == "sigmoid":
            probs = torch.sigmoid(logits).squeeze(-1)
        else:
            probs = torch.softmax(logits, dim=-1)[:, 1]

        logits_all.append(logits.detach().cpu().numpy())
        probs_np = probs.detach().cpu().numpy()
        probs_all.append(probs_np)

    logits_cat = np.concatenate(logits_all, axis=0)
    probs_cat = np.concatenate(probs_all, axis=0)
    return logits_cat, probs_cat


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device=device)

    score_cols = args.score_cols or DEFAULT_SCORE_COLS[: model.cfg.n_models]
    if len(score_cols) != model.cfg.n_models:
        raise ValueError(f"Number of score columns ({len(score_cols)}) must match model.cfg.n_models ({model.cfg.n_models})")

    df = merge_scores_and_raw(args.test_scores, args.test_raw, score_cols=score_cols)

    embedder, embedder_name = build_embedder(args.embedder, device=device, glove_path=args.glove_path, fasttext_path=args.fasttext_path)

    emb = load_or_create_pair_embeddings(
        df=df,
        embedder=embedder,
        embedder_name=embedder_name,
        split_name=args.split_name,
        cache_dir=args.cache_dir,
        left_col="left_text",
        right_col="right_text",
        auto_text_cfg=AutoTextConfig(include_keys=True),
    )

    scores = df[score_cols].to_numpy(np.float32)
    y_true = df["label"].to_numpy(np.int32)

    _, y_prob = predict(model, scores, emb, batch_size=args.batch_size, device=device)
    y_pred = (y_prob >= args.threshold).astype(np.int32)

    metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=args.threshold)

    predictions_path = os.path.join(args.output_dir, "test_predictions.csv")
    metrics_path = os.path.join(args.output_dir, "test_metrics.csv")

    pred_df = pd.DataFrame({
        "id": df["id"].tolist(),
        "label": y_true.tolist(),
        "pred_prob": y_prob.tolist(),
        "pred_label": y_pred.tolist(),
    })
    pred_df.to_csv(predictions_path, index=False)

    metrics_df = pd.DataFrame([
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "test_scores": str(Path(args.test_scores).resolve()),
            "test_raw": str(Path(args.test_raw).resolve()),
            "threshold": args.threshold,
            "accuracy": metrics["acc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
            "n_samples": int(len(y_true)),
        }
    ])
    metrics_df.to_csv(metrics_path, index=False)

    with open(os.path.join(args.output_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump({
            "device": device,
            "embedder": args.embedder,
            "score_cols": score_cols,
            "predictions_csv": predictions_path,
            "metrics_csv": metrics_path,
        }, f, indent=2)

    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
