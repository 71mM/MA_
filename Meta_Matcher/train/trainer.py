import os, json
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import MetaMatcherConfig
from ..models import MetaMatcher
from .metrics import compute_binary_metrics
from .plots import plot_history
from ..io.checkpoints import save_best_checkpoint


def make_loss(cfg: MetaMatcherConfig):
    # cfg.output == "sigmoid" => model outputs logits for binary classification
    return nn.BCEWithLogitsLoss() if cfg.output == "sigmoid" else nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model: nn.Module, loader: Optional[DataLoader], cfg: MetaMatcherConfig) -> Optional[Dict[str, Any]]:
    """
    Evaluate model on a loader. Returns None if loader is None.
    """
    if loader is None:
        return None

    model.eval()
    loss_fn = make_loss(cfg)

    all_pos_probs = []
    all_true = []
    total_loss = 0.0
    n = 0

    # collect gating weights if present
    alphas = []

    for scores, emb, labels in loader:
        scores = scores.to(cfg.device)
        emb = emb.to(cfg.device)

        if cfg.output == "sigmoid":
            y = labels.float().to(cfg.device).view(-1, 1)
        else:
            y = labels.long().to(cfg.device)

        out = model(scores, emb)
        logits = out["logits"]

        # alpha capture
        if isinstance(out, dict) and out.get("alpha", None) is not None:
            alphas.append(out["alpha"].detach().cpu().numpy())

        loss = loss_fn(logits, y)

        bs = scores.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

        y_true_np = labels.detach().cpu().numpy().astype(np.int32)
        all_true.append(y_true_np)

        if cfg.output == "sigmoid":
            pos_prob = torch.sigmoid(logits).squeeze(-1).detach().cpu().numpy()
            all_pos_probs.append(pos_prob)
        else:
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            if cfg.n_classes == 2:
                all_pos_probs.append(probs[:, 1])

    avg_loss = total_loss / max(1, n)
    y_true = np.concatenate(all_true, axis=0) if all_true else np.array([], dtype=np.int32)

    metrics: Dict[str, Any] = {"loss": avg_loss}

    # Binary metrics if possible
    if cfg.output == "sigmoid" or (cfg.output == "softmax" and cfg.n_classes == 2):
        y_prob = np.concatenate(all_pos_probs, axis=0) if all_pos_probs else np.array([])
        bm = compute_binary_metrics(y_true, y_prob)
        metrics.update(bm)

    # alpha stats
    if alphas:
        A = np.concatenate(alphas, axis=0)  # (N, M)
        metrics["alpha_mean"] = A.mean(axis=0).tolist()
        metrics["alpha_std"] = A.std(axis=0).tolist()

        eps = 1e-12
        metrics["alpha_entropy"] = float((-A * np.log(A + eps)).sum(axis=1).mean())

    return metrics


def select_score_from_metrics(val_metrics: Dict[str, Any], cfg: MetaMatcherConfig) -> float:
    """
    IMPORTANT: Choose best checkpoint ONLY by validation metrics.
    Never by test metrics (prevents test leakage).
    """
    if cfg.best_metric == "val_loss":
        return -float(val_metrics["loss"])
    if cfg.best_metric == "val_auc":
        return float(val_metrics.get("auc", float("-inf")))
    # default: val_f1
    return float(val_metrics.get("f1", float("-inf")))


def train(
    cfg: MetaMatcherConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,   # optional: only for logging
    out_dir: str = "runs/meta_matcher",
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    model = MetaMatcher(cfg).to(cfg.device)
    loss_fn = make_loss(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # entropy regularization weight (prevents alpha collapse)
    alpha_entropy_weight = float(getattr(cfg, "alpha_entropy_weight", 0.0))

    history: Dict[str, Any] = {
        "train_loss": [],
        "val_loss": [], "val_f1": [], "val_auc": [], "val_acc": [],
        "test_loss": [], "test_f1": [], "test_auc": [], "test_acc": [],
        "val_alpha_entropy": [], "val_alpha_mean": [], "val_alpha_std": [],
        "test_alpha_entropy": [], "test_alpha_mean": [], "test_alpha_std": [],
    }

    best = {"score": float("-inf"), "epoch": -1, "path": None, "metrics": None}
    best_path = os.path.join(out_dir, "best_model.pt")

    # early stopping
    patience = int(getattr(cfg, "early_stopping_patience", 0) or 0)
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, n = 0.0, 0

        for scores, emb, labels in train_loader:
            scores = scores.to(cfg.device)
            emb = emb.to(cfg.device)
            y = labels.float().to(cfg.device).view(-1, 1) if cfg.output == "sigmoid" else labels.long().to(cfg.device)

            opt.zero_grad(set_to_none=True)

            out = model(scores, emb)
            logits = out["logits"]

            # main loss
            loss = loss_fn(logits, y)

            # entropy regularization (maximize entropy -> subtract)
            if alpha_entropy_weight > 0.0 and isinstance(out, dict) and out.get("alpha", None) is not None:
                alpha = out["alpha"]
                eps = 1e-12
                entropy = (-alpha * torch.log(alpha + eps)).sum(dim=1).mean()
                loss = loss - alpha_entropy_weight * entropy

            loss.backward()
            opt.step()

            bs = scores.size(0)
            running += float(loss.item()) * bs
            n += bs

        train_loss = running / max(1, n)

        val_metrics = evaluate(model, val_loader, cfg)
        test_metrics = evaluate(model, test_loader, cfg) if test_loader is not None else None

        # history
        history["train_loss"].append(train_loss)

        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics.get("acc", float("nan")))
        history["val_f1"].append(val_metrics.get("f1", float("nan")))
        history["val_auc"].append(val_metrics.get("auc", float("nan")))
        history["val_alpha_entropy"].append(val_metrics.get("alpha_entropy", float("nan")))
        history["val_alpha_mean"].append(val_metrics.get("alpha_mean", None))
        history["val_alpha_std"].append(val_metrics.get("alpha_std", None))

        if test_metrics is not None:
            history["test_loss"].append(test_metrics["loss"])
            history["test_acc"].append(test_metrics.get("acc", float("nan")))
            history["test_f1"].append(test_metrics.get("f1", float("nan")))
            history["test_auc"].append(test_metrics.get("auc", float("nan")))
            history["test_alpha_entropy"].append(test_metrics.get("alpha_entropy", float("nan")))
            history["test_alpha_mean"].append(test_metrics.get("alpha_mean", None))
            history["test_alpha_std"].append(test_metrics.get("alpha_std", None))

        # BEST selection: VALIDATION ONLY
        score = select_score_from_metrics(val_metrics, cfg)
        improved = score > best["score"]

        if improved:
            best.update({
                "score": float(score),
                "epoch": epoch,
                "path": best_path,
                "metrics": {"val": val_metrics, "test": test_metrics}
            })
            save_best_checkpoint(best_path, cfg, model, epoch, best["metrics"])
            bad_epochs = 0
        else:
            bad_epochs += 1

        # logging
        alpha_info = ""
        if "alpha_mean" in val_metrics:
            alpha_info = (
                f" | val_alpha_mean={[round(x,3) for x in val_metrics['alpha_mean']]} "
                f"val_alpha_std={[round(x,3) for x in val_metrics.get('alpha_std', [])]} "
                f"val_alpha_ent={val_metrics.get('alpha_entropy', float('nan')):.3f}"
            )

        test_info = ""
        if test_metrics is not None:
            test_info = (
                f" | test_loss={test_metrics['loss']:.4f} "
                f"test_f1={test_metrics.get('f1', float('nan')):.4f}"
            )

        extra = ""
        if alpha_entropy_weight > 0:
            extra += f" | ent_w={alpha_entropy_weight:g}"
        if patience > 0:
            extra += f" | bad={bad_epochs}/{patience}"

        print(
            f"[Epoch {epoch:03d}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics.get('acc', float('nan')):.4f} "
            f"val_f1={val_metrics.get('f1', float('nan')):.4f} "
            f"val_auc={val_metrics.get('auc', float('nan')):.4f}"
            + test_info
            + alpha_info
            + extra
        )

        # early stopping (based on validation selection metric)
        if patience > 0 and bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_history(history, out_dir)
    return {"best": best, "history": history, "out_dir": out_dir}
