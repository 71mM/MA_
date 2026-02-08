import os
import json
import random
import subprocess
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import MetaMatcherConfig
from ..models import MetaMatcher
from .metrics import compute_binary_metrics
from .plots import plot_history
from ..io.checkpoints import save_best_checkpoint


def resolve_runtime_device(cfg: MetaMatcherConfig) -> str:
    if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] cfg.device='{cfg.device}' but CUDA is unavailable. Falling back to CPU.")
        return "cpu"
    return cfg.device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_pos_weight(train_loader: DataLoader, device: str) -> Optional[torch.Tensor]:
    pos = 0
    neg = 0
    for _, _, labels in train_loader:
        labels_np = labels.detach().cpu().numpy().astype(np.int32)
        pos += int((labels_np == 1).sum())
        neg += int((labels_np == 0).sum())
    if pos == 0:
        return None
    return torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        pt = torch.exp(-bce)
        loss = ((1 - pt) ** self.gamma) * bce
        return loss.mean()


def make_loss(cfg: MetaMatcherConfig, train_loader: Optional[DataLoader] = None, device: Optional[str] = None):
    runtime_device = device if device is not None else cfg.device

    if cfg.loss_type == "ce" or (cfg.loss_type == "auto" and cfg.output == "softmax"):
        return nn.CrossEntropyLoss()

    pos_weight = None
    if cfg.use_pos_weight and train_loader is not None:
        pos_weight = infer_pos_weight(train_loader, runtime_device)

    if cfg.loss_type == "focal_bce":
        return FocalBCEWithLogitsLoss(gamma=cfg.focal_gamma, pos_weight=pos_weight)

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cfg: MetaMatcherConfig,
    device: str,
    loss_fn: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    model.eval()
    local_loss_fn = loss_fn if loss_fn is not None else make_loss(cfg, device=device)

    all_pos_probs = []
    all_true = []
    total_loss = 0.0
    n = 0
    alphas = []

    for scores, emb, labels in loader:
        scores = scores.to(device)
        emb = emb.to(device)

        if cfg.output == "sigmoid":
            y = labels.float().to(device).view(-1, 1)
        else:
            y = labels.long().to(device)

        out = model(scores, emb)
        logits = out["logits"]

        if isinstance(out, dict) and out.get("alpha", None) is not None:
            alphas.append(out["alpha"].detach().cpu().numpy())

        loss = local_loss_fn(logits, y)

        bs = scores.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

        all_true.append(y_true_np)

        if is_binary_eval:
            pos_prob = _extract_pos_prob(logits, cfg)
            if pos_prob is not None:
                all_pos_probs.append(pos_prob)

        alpha = out.get("alpha", None) if isinstance(out, dict) else None
        if alpha is not None:
            alpha_stats.update(alpha)

    avg_loss = total_loss / max(1, n)
    y_true = np.concatenate(all_true, axis=0) if all_true else np.array([], dtype=np.int32)

    metrics: Dict[str, Any] = {"loss": avg_loss}

    if cfg.output == "sigmoid" or (cfg.output == "softmax" and cfg.n_classes == 2):
        y_prob = np.concatenate(all_pos_probs, axis=0) if all_pos_probs else np.array([])
        bm = compute_binary_metrics(y_true, y_prob)
        metrics.update(bm)

    if alphas:
        a = np.concatenate(alphas, axis=0)
        metrics["alpha_mean"] = a.mean(axis=0).tolist()
        metrics["alpha_std"] = a.std(axis=0).tolist()

        eps = 1e-12
        metrics["alpha_entropy"] = float((-a * np.log(a + eps)).sum(axis=1).mean())

    metrics.update(alpha_stats.finalize())
    return metrics


def select_score_from_metrics(val_metrics: Dict[str, Any], cfg: MetaMatcherConfig) -> float:
    if cfg.best_metric == "val_loss":
        return -float(val_metrics["loss"])
    if cfg.best_metric == "val_auc":
        return float(val_metrics.get("auc", float("-inf")))
    return float(val_metrics.get("f1", float("-inf")))


def entropy_weight_for_epoch(cfg: MetaMatcherConfig, epoch: int) -> float:
    if cfg.epochs <= 1:
        return float(cfg.alpha_entropy_weight_final)
    ratio = (epoch - 1) / (cfg.epochs - 1)
    start = float(cfg.alpha_entropy_weight)
    end = float(cfg.alpha_entropy_weight_final)
    return start + (end - start) * ratio


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def train(
    cfg: MetaMatcherConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    out_dir: str = "runs/meta_matcher",
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.seed)

    runtime_device = resolve_runtime_device(cfg)

    model = MetaMatcher(cfg).to(runtime_device)
    loss_fn = make_loss(cfg, train_loader=train_loader, device=runtime_device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = None
    if cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min" if cfg.best_metric == "val_loss" else "max",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
        )

    history: Dict[str, Any] = {
        "train_loss": [], "val_loss": [], "test_loss": [],
        "val_f1": [], "val_auc": [], "val_acc": [],
        "test_f1": [], "test_auc": [], "test_acc": [],
        "val_alpha_entropy": [], "test_alpha_entropy": [],
        "val_alpha_mean": [], "test_alpha_mean": [],
        "val_alpha_std": [], "test_alpha_std": [],
        "lr": [], "entropy_weight": [],
        "device": runtime_device,
    }

    best = {"score": float("-inf"), "epoch": -1, "path": None, "metrics": None}
    best_path = os.path.join(out_dir, "best_model.pt")
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, n = 0.0, 0
        ent_weight = entropy_weight_for_epoch(cfg, epoch)

        for scores, emb, labels in train_loader:
            scores = scores.to(runtime_device)
            emb = emb.to(runtime_device)
            y = labels.float().to(runtime_device).view(-1, 1) if cfg.output == "sigmoid" else labels.long().to(runtime_device)

            opt.zero_grad(set_to_none=True)

            out = model(scores, emb)
            logits = out["logits"]

            loss = loss_fn(logits, y)

            if ent_weight > 0.0 and isinstance(out, dict) and out.get("alpha", None) is not None:
                alpha = out["alpha"]
                eps = 1e-12
                entropy = (-alpha * torch.log(alpha + eps)).sum(dim=1).mean()
                loss = loss - ent_weight * entropy

            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
            opt.step()

            bs = scores.size(0)
            running += float(loss.item()) * bs
            n += bs

        train_loss = running / max(1, n)

        val_metrics = evaluate(model, val_loader, cfg, device=runtime_device, loss_fn=loss_fn)
        test_metrics = (
            evaluate(model, test_loader, cfg, device=runtime_device, loss_fn=loss_fn)
            if test_loader is not None
            else None
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics.get("acc", float("nan")))
        history["val_f1"].append(val_metrics.get("f1", float("nan")))
        history["val_auc"].append(val_metrics.get("auc", float("nan")))
        history["val_alpha_entropy"].append(val_metrics.get("alpha_entropy", float("nan")))
        history["val_alpha_mean"].append(val_metrics.get("alpha_mean", None))
        history["val_alpha_std"].append(val_metrics.get("alpha_std", None))
        history["lr"].append(opt.param_groups[0]["lr"])
        history["entropy_weight"].append(ent_weight)

        if test_metrics is not None:
            history["test_loss"].append(test_metrics["loss"])
            history["test_acc"].append(test_metrics.get("acc", float("nan")))
            history["test_f1"].append(test_metrics.get("f1", float("nan")))
            history["test_auc"].append(test_metrics.get("auc", float("nan")))
            history["test_alpha_entropy"].append(test_metrics.get("alpha_entropy", float("nan")))
            history["test_alpha_mean"].append(test_metrics.get("alpha_mean", None))
            history["test_alpha_std"].append(test_metrics.get("alpha_std", None))

        score = select_score_from_metrics(val_metrics, cfg)
        if score > best["score"]:
            best.update({"score": score, "epoch": epoch, "path": best_path,
                         "metrics": {"val": val_metrics, "test": test_metrics}})
            save_best_checkpoint(best_path, cfg, model, epoch, best["metrics"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            if cfg.best_metric == "val_loss":
                scheduler.step(val_metrics["loss"])
            elif cfg.best_metric == "val_auc":
                scheduler.step(val_metrics.get("auc", float("-inf")))
            else:
                scheduler.step(val_metrics.get("f1", float("-inf")))

        print(
            f"[Epoch {epoch:03d}/{cfg.epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics.get('f1', float('nan')):.4f} "
            f"val_auc={val_metrics.get('auc', float('nan')):.4f} "
            f"lr={opt.param_groups[0]['lr']:.3e} ent_w={ent_weight:.4f} device={runtime_device}"
        )

        if cfg.early_stopping_patience > 0 and epochs_without_improvement >= cfg.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (patience={cfg.early_stopping_patience})")
            break

    run_meta = {
        "seed": cfg.seed,
        "git_commit": get_git_commit(),
        "config": cfg.__dict__,
        "best_epoch": best["epoch"],
        "runtime_device": runtime_device,
    }

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    plot_history(history, out_dir)
    return {"best": best, "history": history, "out_dir": out_dir, "run_meta": run_meta}
