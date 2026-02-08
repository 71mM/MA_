from dataclasses import dataclass
from typing import Literal
import torch


@dataclass
class MetaMatcherConfig:
    n_models: int
    emb_dim: int

    arch: Literal["mlp", "rnn", "lstm"] = "mlp"

    use_attention: bool = True
    attn_hidden: int = 64
    attn_mode: Literal["gating", "token"] = "gating"
    alpha_temperature: float = 2.0
    alpha_entropy_weight: float = 0.01
    alpha_entropy_weight_final: float = 0.0

    mlp_hidden: int = 128
    mlp_layers: int = 1
    dropout: float = 0.5
    activation: Literal["relu", "gelu", "silu"] = "relu"

    # token-level features / attention
    model_id_dim: int = 8
    token_attn_heads: int = 1

    # score calibration
    calibration_mode: Literal["full", "diagonal"] = "full"

    # rnn/lstm params
    rnn_hidden: int = 128
    rnn_layers: int = 1
    bidirectional: bool = True
    rnn_pooling: Literal["last", "mean", "max", "hidden"] = "hidden"

    # output
    output: Literal["sigmoid", "softmax"] = "sigmoid"
    n_classes: int = 2  # only used if output="softmax"

    # optimization
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    loss_type: Literal["auto", "bce", "focal_bce", "ce"] = "auto"
    focal_gamma: float = 2.0
    use_pos_weight: bool = False

    # scheduler / stopping / gradients
    scheduler: Literal["none", "plateau"] = "plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    early_stopping_patience: int = 5
    grad_clip_norm: float = 1.0

    # misc
    best_metric: Literal["val_f1", "val_auc", "val_loss"] = "val_f1"
    seed: int = 42
    debug_checks: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
