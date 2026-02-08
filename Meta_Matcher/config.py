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

    attn_mode: str = "gating"   # "gating" | "token"
    alpha_temperature = 2.0
    alpha_entropy_weight = 0.01

    mlp_hidden: int = 128
    mlp_layers: int = 1
    dropout: float = 0.5
    activation: Literal["relu"] = "relu"

    # rnn/lstm params
    rnn_hidden: int = 128
    rnn_layers: int = 1
    bidirectional: bool = True

    # output
    output: Literal["sigmoid", "softmax"] = "sigmoid"
    n_classes: int = 2  # only used if output="softmax"

    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    best_metric: Literal["val_f1", "val_auc", "val_loss","test_f1", "test_auc", "test_loss"] = "test_f1"