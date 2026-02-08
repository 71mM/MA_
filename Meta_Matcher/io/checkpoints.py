import torch
from typing import Optional
from ..config import MetaMatcherConfig
from ..models import MetaMatcher


def _remap_legacy_state_dict_keys(state_dict: dict) -> dict:
    """Map legacy checkpoint keys to the current module layout."""
    remapped = dict(state_dict)

    # Legacy checkpoints stored calibration weights directly at score_calib.*
    # while the refactor moved them under score_calib.linear.* in full mode.
    if "score_calib.weight" in remapped and "score_calib.linear.weight" not in remapped:
        remapped["score_calib.linear.weight"] = remapped.pop("score_calib.weight")
    if "score_calib.bias" in remapped and "score_calib.linear.bias" not in remapped:
        remapped["score_calib.linear.bias"] = remapped.pop("score_calib.bias")

    return remapped


def _validate_compatibility(missing_keys: list, unexpected_keys: list):
    """
    Allow only known compatibility differences when loading legacy checkpoints.
    """
    allowed_missing_prefixes = ("model_id_emb.",)
    allowed_missing_exact = {
        "score_calib.scale",
        "score_calib.bias",
        "score_calib.linear.weight",
        "score_calib.linear.bias",
    }

    bad_missing = [
        key for key in missing_keys
        if key not in allowed_missing_exact
        and not key.startswith(allowed_missing_prefixes)
    ]
    if bad_missing or unexpected_keys:
        raise RuntimeError(
            "Checkpoint state dict is incompatible with current MetaMatcher. "
            f"Missing keys: {bad_missing}. Unexpected keys: {unexpected_keys}."
        )

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
    state_dict = _remap_legacy_state_dict_keys(ckpt["state_dict"])
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    _validate_compatibility(missing_keys, unexpected_keys)
    model.to(cfg.device)
    model.eval()
    return model
