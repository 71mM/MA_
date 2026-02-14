from dataclasses import dataclass
from typing import Literal, Optional
import torch


@dataclass
class MetaMatcherConfig:

    n_models: int


    emb_dim: int


    n_base_models: Optional[int] = None
    n_extra_features: Optional[int] = None

    base_score_input: Literal["prob", "logit"] = "prob"
    score_logit_eps: float = 1e-6


    arch: Literal["mlp", "rnn", "lstm"] = "mlp"


    use_attention: bool = True
    attn_mode: Literal["gating", "token", "both"] = "gating"

    attn_hidden: int = 64
    alpha_temperature: float = 2.0

    alpha_entropy_weight: float = 0.01
    alpha_entropy_weight_final: float = 0.0
    alpha_entropy_anneal: Literal["none", "linear"] = "linear"


    mlp_hidden: int = 256
    mlp_layers: int = 2
    dropout: float = 0.2
    activation: Literal["relu"] = "relu"


    model_id_dim: int = 8
    token_attn_heads: int = 1


    rnn_hidden: int = 128
    rnn_layers: int = 1
    bidirectional: bool = True
    rnn_pooling: Literal["last", "mean", "max", "hidden"] = "hidden"


    calibration_mode: Literal["full", "diagonal"] = "diagonal"


    score_dropout: float = 0.10
    score_dropout_min_keep: int = 1
    missing_score_value: float = 0.5

    use_score_mask: bool = True
    mask_gating: bool = True

    score_noise: float = 0.0


    output: Literal["sigmoid", "softmax"] = "sigmoid"
    n_classes: int = 2


    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 10

    loss_type: Literal["auto", "bce", "focal_bce", "ce"] = "auto"
    focal_gamma: float = 2.0
    use_pos_weight: bool = False


    scheduler: Literal["none", "plateau"] = "plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2

    early_stopping: bool = False
    early_stopping_patience: int = 20

    grad_clip_norm: float = 1.0


    best_metric: Literal["val_f1", "val_auc", "val_loss"] = "val_f1"
    seed: int = 42
    debug_checks: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        if self.n_base_models is None:
            self.n_base_models = int(self.n_models)

        #
        if self.n_extra_features is None:
            self.n_extra_features = int(self.n_models - self.n_base_models)

        # Safety checks
        self.n_base_models = int(self.n_base_models)
        self.n_extra_features = int(self.n_extra_features)

        if self.n_base_models <= 0:
            raise ValueError("n_base_models must be > 0")
        if self.n_extra_features < 0:
            raise ValueError("n_extra_features must be >= 0")
        if self.n_base_models + self.n_extra_features != int(self.n_models):
            raise ValueError(
                f"Inconsistent dims: n_models={self.n_models} "
                f"but n_base_models({self.n_base_models}) + n_extra_features({self.n_extra_features}) != n_models"
            )
