import torch
import torch.nn as nn
from typing import Dict, Optional
from ..config import MetaMatcherConfig


class ScoreCalibrator(nn.Module):
    def __init__(self, n_models: int, mode: str = "full"):
        super().__init__()
        self.mode = mode
        if mode == "full":
            self.linear = nn.Linear(n_models, n_models, bias=True)
            with torch.no_grad():
                self.linear.weight.copy_(torch.eye(n_models))
                self.linear.bias.zero_()
        elif mode == "diagonal":
            self.scale = nn.Parameter(torch.ones(n_models))
            self.bias = nn.Parameter(torch.zeros(n_models))
        else:
            raise ValueError(f"Unknown calibration_mode: {mode}")

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        if self.mode == "full":
            return torch.sigmoid(self.linear(scores))
        return torch.sigmoid(scores * self.scale + self.bias)


def resolve_activation(name: str):
    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU
    if name == "silu":
        return nn.SiLU
    raise ValueError(f"Unsupported activation: {name}")


class MetaMatcher(nn.Module):
    def __init__(self, cfg: MetaMatcherConfig):
        super().__init__()
        self.cfg = cfg
        act = resolve_activation(cfg.activation)

        self.score_calib = ScoreCalibrator(cfg.n_models, mode=cfg.calibration_mode)

        self.model_id_emb = nn.Embedding(cfg.n_models, cfg.model_id_dim)

        self.use_gating = bool(cfg.use_attention and cfg.attn_mode == "gating")
        self.use_token_attention = bool(cfg.use_attention and cfg.attn_mode == "token")

        if self.use_gating:
            self.gate = nn.Sequential(
                nn.Linear(cfg.emb_dim, cfg.attn_hidden),
                act(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.attn_hidden, cfg.n_models),
            )
        else:
            self.gate = None

        self.token_dim = 1 + cfg.emb_dim + cfg.model_id_dim
        if self.use_token_attention:
            self.token_attn = nn.MultiheadAttention(
                embed_dim=self.token_dim,
                num_heads=cfg.token_attn_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.token_norm = nn.LayerNorm(self.token_dim)
        else:
            self.token_attn = None
            self.token_norm = None

        if cfg.arch == "mlp":
            in_dim = cfg.n_models + cfg.emb_dim
            if self.use_gating:
                in_dim += 1
            if self.use_token_attention:
                in_dim += self.token_dim

            layers = []
            d = in_dim
            for _ in range(cfg.mlp_layers):
                layers += [nn.Linear(d, cfg.mlp_hidden), act(), nn.Dropout(cfg.dropout)]
                d = cfg.mlp_hidden
            self.backbone = nn.Sequential(*layers)
            backbone_out = cfg.mlp_hidden
            self.rnn = None

        elif cfg.arch in ("rnn", "lstm"):
            rnn_cls = nn.RNN if cfg.arch == "rnn" else nn.LSTM
            self.rnn = rnn_cls(
                input_size=self.token_dim,
                hidden_size=cfg.rnn_hidden,
                num_layers=cfg.rnn_layers,
                batch_first=True,
                bidirectional=cfg.bidirectional,
                dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0,
            )
            backbone_out = cfg.rnn_hidden * (2 if cfg.bidirectional else 1)
            self.backbone = None
        else:
            raise ValueError(f"Unknown arch: {cfg.arch}")

        self.head = nn.Linear(backbone_out, 1 if cfg.output == "sigmoid" else cfg.n_classes)

    def _build_tokens(self, scores_cal: torch.Tensor, pair_emb: torch.Tensor) -> torch.Tensor:
        bsz, n_models = scores_cal.shape
        model_idx = torch.arange(n_models, device=scores_cal.device).unsqueeze(0).expand(bsz, -1)
        model_feat = self.model_id_emb(model_idx)
        score_feat = scores_cal.unsqueeze(-1)
        emb_feat = pair_emb.unsqueeze(1).expand(bsz, n_models, pair_emb.size(-1))
        return torch.cat([score_feat, emb_feat, model_feat], dim=-1)

    def _apply_token_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.use_token_attention:
            return tokens
        attn_out, _ = self.token_attn(tokens, tokens, tokens)
        return self.token_norm(attn_out + tokens)

    def _rnn_pool(self, out: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        mode = self.cfg.rnn_pooling
        if mode == "last":
            return out[:, -1, :]
        if mode == "mean":
            return out.mean(dim=1)
        if mode == "max":
            return out.max(dim=1).values
        if mode == "hidden":
            if self.cfg.arch == "lstm":
                hidden = hidden[0]
            last = hidden[-2:] if self.cfg.bidirectional else hidden[-1:]
            return torch.cat(list(last), dim=-1)
        raise ValueError(f"Unsupported rnn_pooling: {mode}")

    def forward(self, scores: torch.Tensor, pair_emb: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        if self.cfg.debug_checks:
            if scores.ndim != 2 or pair_emb.ndim != 2:
                raise ValueError(f"Expected 2D tensors, got {scores.shape=} and {pair_emb.shape=}")
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                raise ValueError("scores contains NaN or Inf")
            if torch.isnan(pair_emb).any() or torch.isinf(pair_emb).any():
                raise ValueError("pair_emb contains NaN or Inf")
            if scores.size(0) != pair_emb.size(0):
                raise ValueError("Batch size mismatch between scores and pair_emb")
            if (scores.min() < 0) or (scores.max() > 1):
                raise ValueError("scores are expected in [0, 1]")

        scores_cal = self.score_calib(scores)
        alpha = None

        if self.cfg.arch == "mlp":
            chunks = [scores_cal, pair_emb]

            if self.use_gating:
                gate_logits = self.gate(pair_emb)
                t = max(float(self.cfg.alpha_temperature), 1e-6)
                alpha = torch.softmax(gate_logits / t, dim=-1)
                s_meta = (alpha * scores_cal).sum(dim=-1, keepdim=True)
                chunks.insert(0, s_meta)

            if self.use_token_attention:
                tok = self._build_tokens(scores_cal, pair_emb)
                tok = self._apply_token_attention(tok)
                tok_pool = tok.mean(dim=1)
                chunks.append(tok_pool)

            x = torch.cat(chunks, dim=-1)
            h = self.backbone(x)
            logits = self.head(h)

        else:
            tok = self._build_tokens(scores_cal, pair_emb)
            tok = self._apply_token_attention(tok)
            out, hidden = self.rnn(tok)
            h = self._rnn_pool(out, hidden)
            logits = self.head(h)

            if self.use_gating:
                gate_logits = self.gate(pair_emb)
                t = max(float(self.cfg.alpha_temperature), 1e-6)
                alpha = torch.softmax(gate_logits / t, dim=-1)

        return {"logits": logits, "alpha": alpha, "scores_cal": scores_cal}
