import torch
from typing import Optional
from ..config import MetaMatcherConfig
from ..models import MetaMatcher

def save_best_checkpoint(path: str, cfg: MetaMatcherConfig, model: MetaMatcher, epoch: int, val_metrics: dict):
    torch.save(
        {
            "config": cfg.__dict__,
            "state_dict": model.state_dict(),
            "best_epoch": epoch,
            "best_val_metrics": val_metrics,
        },
        path,
    )

def load_model(ckpt_path: str, device: Optional[str] = None) -> MetaMatcher:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = MetaMatcherConfig(**ckpt["config"])
    if device is not None:
        cfg.device = device
    model = MetaMatcher(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(cfg.device)
    model.eval()
    return model
