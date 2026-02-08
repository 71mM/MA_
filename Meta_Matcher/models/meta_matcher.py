import torch
import torch.nn as nn
from typing import Dict, Optional
from ..config import MetaMatcherConfig


class MetaMatcher(nn.Module):
    """
    MetaMatcher with:
      - learnable score calibration (per base model)
      - context gating: alpha = softmax(g(pair_emb)) over base models
      - optional backbone: MLP (recommended) or (L)STM over model tokens

    Returns:
      logits: (B, 1) for sigmoid or (B, C) for softmax
      alpha:  (B, M) gating weights (if enabled)
      scores_cal: (B, M) calibrated scores (for diagnostics)
    """

    def __init__(self, cfg: MetaMatcherConfig):
        super().__init__()
        self.cfg = cfg
        act = nn.ReLU if cfg.activation == "relu" else nn.ReLU

        # -------------------------
        # 1) Score calibration
        # -------------------------
        # Map scores (B,M) -> (B,M), initialized as identity then sigmoid
        self.score_calib = nn.Linear(cfg.n_models, cfg.n_models, bias=True)
        with torch.no_grad():
            self.score_calib.weight.copy_(torch.eye(cfg.n_models))
            self.score_calib.bias.zero_()

        # -------------------------
        # 2) Context gating over models (recommended)
        # -------------------------
        # If cfg.use_attention=True, interpret as "use gating"
        self.use_gating = bool(cfg.use_attention)

        if self.use_gating:
            self.gate = nn.Sequential(
                nn.Linear(cfg.emb_dim, cfg.attn_hidden),
                act(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.attn_hidden, cfg.n_models),
            )
        else:
            self.gate = None

        # -------------------------
        # 3) Backbone
        # -------------------------
        if cfg.arch == "mlp":
            # Input features:
            # - calibrated scores: M
            # - pair_emb: E (= cfg.emb_dim)
            # - if gating: add s_meta scalar and (optionally) alpha
            # We'll include:
            #   x = [s_meta, scores_cal, pair_emb]
            # so in_dim = 1 + M + E
            # Without gating:
            #   x = [scores_cal, pair_emb] => M + E
            if self.use_gating:
                in_dim = 1 + cfg.n_models + cfg.emb_dim
            else:
                in_dim = cfg.n_models + cfg.emb_dim

            layers = []
            d = in_dim
            for _ in range(cfg.mlp_layers):
                layers += [nn.Linear(d, cfg.mlp_hidden), act(), nn.Dropout(cfg.dropout)]
                d = cfg.mlp_hidden
            self.backbone = nn.Sequential(*layers)
            backbone_out = cfg.mlp_hidden

            self.rnn = None

        elif cfg.arch in ("rnn", "lstm"):
            # We treat per-model token as sequence.
            # Token includes calibrated score plus context features (pair_emb)
            # tok_i = [scores_cal_i, pair_emb]
            token_dim = 1 + cfg.emb_dim
            rnn_cls = nn.RNN if cfg.arch == "rnn" else nn.LSTM
            self.rnn = rnn_cls(
                input_size=token_dim,
                hidden_size=cfg.rnn_hidden,
                num_layers=cfg.rnn_layers,
                batch_first=True,
                bidirectional=cfg.bidirectional,
                dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0
            )
            backbone_out = cfg.rnn_hidden * (2 if cfg.bidirectional else 1)
            self.backbone = None

        else:
            raise ValueError(f"Unknown arch: {cfg.arch}")

        # -------------------------
        # 4) Output head
        # -------------------------
        self.head = nn.Linear(backbone_out, 1 if cfg.output == "sigmoid" else cfg.n_classes)

    def forward(self, scores: torch.Tensor, pair_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        scores:   (B, M) raw base model scores in [0,1]
        pair_emb: (B, E) pair embedding (E = cfg.emb_dim)
        """
        # (B,M) -> (B,M) calibrated scores in (0,1)
        scores_cal = torch.sigmoid(self.score_calib(scores))

        alpha = None

        if self.cfg.arch == "mlp":
            if self.use_gating:
                gate_logits = self.gate(pair_emb)

                T = float(getattr(self.cfg, "alpha_temperature", 1.0))
                alpha = torch.softmax(gate_logits / max(T, 1e-6), dim=-1)

                s_meta = (alpha * scores_cal).sum(dim=-1, keepdim=True)
                x = torch.cat([s_meta, scores_cal, pair_emb], dim=-1)
            else:
                x = torch.cat([scores_cal, pair_emb], dim=-1)

            h = self.backbone(x)
            logits = self.head(h)

        else:
            B, M = scores_cal.shape
            s = scores_cal.unsqueeze(-1)
            c = pair_emb.unsqueeze(1).expand(B, M, pair_emb.size(-1))
            tok = torch.cat([s, c], dim=-1)
            out, _ = self.rnn(tok)
            h = out[:, -1, :]
            logits = self.head(h)

            if self.use_gating:
                gate_logits = self.gate(pair_emb)

                T = float(getattr(self.cfg, "alpha_temperature", 1.0))
                alpha = torch.softmax(gate_logits / max(T, 1e-6), dim=-1)

        return {"logits": logits, "alpha": alpha, "scores_cal": scores_cal}
