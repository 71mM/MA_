import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from ..config import MetaMatcherConfig


class ScoreCalibrator(nn.Module):
    """
    Calibrates per-model probabilities by applying an affine transform in logit-space.
    Input: probs in [0,1]
    Output: calibrated probs in (0,1)
    """
    def __init__(self, n_models: int, mode: str = "full", eps: float = 1e-6):
        super().__init__()
        self.mode = mode
        self.eps = eps

        if mode == "full":
            self.linear = nn.Linear(n_models, n_models, bias=True)
            with torch.no_grad():
                self.linear.weight.copy_(torch.eye(n_models))
                self.linear.bias.zero_()
        elif mode == "diagonal":
            self.scale = nn.Parameter(torch.ones(n_models))
            self.bias  = nn.Parameter(torch.zeros(n_models))
        else:
            raise ValueError(f"Unknown calibration_mode: {mode}")

    def _prob_to_logit(self, p: torch.Tensor) -> torch.Tensor:
        p = p.clamp(self.eps, 1.0 - self.eps)
        return torch.log(p / (1.0 - p))

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        logits = self._prob_to_logit(probs)
        if self.mode == "full":
            logits_cal = self.linear(logits)
        else:
            logits_cal = logits * self.scale + self.bias
        return torch.sigmoid(logits_cal)


def resolve_activation(name: str):
    if name == "relu":
        return nn.ReLU
    raise ValueError(f"Unsupported activation: {name}")


class MetaMatcher(nn.Module):
    """
    scores: (B, M+K)
      - first M = base model outputs
      - last  K = extra features (aggregates etc.)

    base_score_input:
      - "prob": base scores in [0,1], calibration happens in logit-space (recommended)
    """
    def __init__(self, cfg: MetaMatcherConfig):
        super().__init__()
        self.cfg = cfg
        act = resolve_activation(cfg.activation)

        # ---- base models count ----
        # If not set, fallback to cfg.n_models (backward-compatible: all are base)
        self.M = int(getattr(cfg, "n_base_models", cfg.n_models))
        self.base_score_input = getattr(cfg, "base_score_input", "prob")  # "prob" or "logit"
        self.score_logit_eps = float(getattr(cfg, "score_logit_eps", 1e-6))

        # ---- extra feature count ----
        # If cfg doesn't define it, infer from total n_models
        if getattr(cfg, "n_extra_features", None) is None:
            cfg.n_extra_features = int(cfg.n_models - self.M)


        if getattr(cfg, "debug_checks", False):
            if self.M > int(cfg.n_models):
                raise ValueError(f"n_base_models (M={self.M}) cannot be > n_models ({cfg.n_models})")
            if int(cfg.n_extra_features) != int(cfg.n_models - self.M):
                raise ValueError(
                    f"Inconsistent dims: n_models={cfg.n_models}, M={self.M} -> "
                    f"expected n_extra_features={cfg.n_models - self.M} but got {cfg.n_extra_features}"
                )

        # ---- calibrator operates ONLY on base models ----
        self.score_calib = ScoreCalibrator(self.M, mode=cfg.calibration_mode, eps=self.score_logit_eps)

        # model id embedding for base-model tokens
        self.model_id_emb = nn.Embedding(self.M, cfg.model_id_dim)

        # attention flags
        self.use_gating = bool(cfg.use_attention and cfg.attn_mode in ("gating", "both"))
        self.use_token_attention = bool(cfg.use_attention and cfg.attn_mode in ("token", "both"))

        # gating network (context -> per-model logits)
        if self.use_gating:
            self.gate = nn.Sequential(
                nn.Linear(cfg.emb_dim, cfg.attn_hidden),
                act(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.attn_hidden, self.M),
            )
        else:
            self.gate = None

        # token dimension: [score(1), (mask(1)?), pair_emb(E), model_id(D)]
        self.token_dim = 1 + cfg.emb_dim + cfg.model_id_dim
        if getattr(cfg, "use_score_mask", False):
            self.token_dim += 1

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

        # backbone
        if cfg.arch == "mlp":
            # input = [optional s_meta] + base_scores_cal(M) + [optional mask(M)] + extra(K) + pair_emb(E) + [optional tok_pool]
            self.in_dim = self._compute_mlp_in_dim()
            layers = []
            d = self.in_dim
            for _ in range(cfg.mlp_layers):
                layers += [nn.Linear(d, cfg.mlp_hidden), act(), nn.Dropout(cfg.dropout)]
                d = cfg.mlp_hidden
            self.backbone = nn.Sequential(*layers)
            backbone_out = cfg.mlp_hidden
            self.rnn = None

        elif cfg.arch in ("rnn", "lstm"):
            # RNN runs ONLY over base-model tokens. Extras will be fused AFTER pooling.
            rnn_cls = nn.RNN if cfg.arch == "rnn" else nn.LSTM
            self.rnn = rnn_cls(
                input_size=self.token_dim,
                hidden_size=cfg.rnn_hidden,
                num_layers=cfg.rnn_layers,
                batch_first=True,
                bidirectional=cfg.bidirectional,
                dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0,
            )
            rnn_out = cfg.rnn_hidden * (2 if cfg.bidirectional else 1)

            # fusion MLP after pooling: [rnn_pool + extra(K) + pair_emb(E) + optional s_meta + optional mask(M)]
            fusion_in = rnn_out + cfg.emb_dim
            fusion_in += int(getattr(cfg, "n_extra_features", 0))
            if self.use_gating:
                fusion_in += 1
            if getattr(cfg, "use_score_mask", False):
                fusion_in += self.M

            self.fusion = nn.Sequential(
                nn.Linear(fusion_in, cfg.mlp_hidden),
                act(),
                nn.Dropout(cfg.dropout),
            )
            backbone_out = cfg.mlp_hidden
            self.backbone = None

        else:
            raise ValueError(f"Unknown arch: {cfg.arch}")

        self.head = nn.Linear(backbone_out, 1 if cfg.output == "sigmoid" else cfg.n_classes)


    def _split_scores(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base = scores[:, :self.M]
        extra = scores[:, self.M:] if scores.size(1) > self.M else scores.new_zeros(scores.size(0), 0)
        return base, extra

    def _compute_mlp_in_dim(self) -> int:
        E = int(self.cfg.emb_dim)
        K = int(getattr(self.cfg, "n_extra_features", 0))
        dim = 0
        if self.use_gating:
            dim += 1
        dim += self.M
        if getattr(self.cfg, "use_score_mask", False):
            dim += self.M
        dim += K
        dim += E
        if self.use_token_attention:
            dim += self.token_dim
        return dim

    def _build_tokens(self, scores_cal: torch.Tensor, pair_emb: torch.Tensor, score_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, n_models = scores_cal.shape
        model_idx = torch.arange(n_models, device=scores_cal.device).unsqueeze(0).expand(bsz, -1)
        model_feat = self.model_id_emb(model_idx)

        score_feat = scores_cal.unsqueeze(-1)
        emb_feat = pair_emb.unsqueeze(1).expand(bsz, n_models, pair_emb.size(-1))

        feats = [score_feat]
        if getattr(self.cfg, "use_score_mask", False):
            if score_mask is None:
                score_mask = torch.ones_like(scores_cal)
            feats.append(score_mask.unsqueeze(-1))

        feats += [emb_feat, model_feat]
        return torch.cat(feats, dim=-1)

    def _apply_token_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.use_token_attention:
            return tokens
        attn_out, _ = self.token_attn(tokens, tokens, tokens)
        return self.token_norm(attn_out + tokens)

    def _apply_model_dropout(self, scores_cal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        score_mask = torch.ones_like(scores_cal)
        p = float(getattr(self.cfg, "score_dropout", 0.0))
        if (not self.training) or p <= 0.0:
            return scores_cal, score_mask

        score_mask = (torch.rand_like(scores_cal) > p).float()

        min_keep = int(getattr(self.cfg, "score_dropout_min_keep", 1))
        if min_keep > 0:
            kept = score_mask.sum(dim=1)
            need_fix = kept < min_keep
            if need_fix.any():
                idx = torch.nonzero(need_fix).squeeze(1)
                for b in idx.tolist():
                    perm = torch.randperm(scores_cal.size(1), device=scores_cal.device)
                    score_mask[b, perm[:min_keep]] = 1.0

        missing_val = float(getattr(self.cfg, "missing_score_value", 0.5))
        scores_cal = scores_cal * score_mask + missing_val * (1.0 - score_mask)
        return scores_cal, score_mask

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

    # ---------- forward ----------
    def forward(self, scores: torch.Tensor, pair_emb: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        if getattr(self.cfg, "debug_checks", False):
            if scores.ndim != 2 or pair_emb.ndim != 2:
                raise ValueError(f"Expected 2D tensors, got {scores.shape=} and {pair_emb.shape=}")
            if scores.size(0) != pair_emb.size(0):
                raise ValueError("Batch size mismatch between scores and pair_emb")
            # helpful: verify score dim matches config
            if hasattr(self.cfg, "n_models") and scores.size(1) != int(self.cfg.n_models):
                raise ValueError(f"Expected scores dim {self.cfg.n_models}, got {scores.size(1)}")

        base, extra = self._split_scores(scores)

        # ---- calibrate base scores ----
        if self.base_score_input == "prob":
            if getattr(self.cfg, "debug_checks", False):
                if (base.min() < 0) or (base.max() > 1):
                    raise ValueError("Base scores expected in [0,1] because base_score_input='prob'")
            base_cal = self.score_calib(base)
        elif self.base_score_input == "logit":
            probs = torch.sigmoid(base)
            base_cal = self.score_calib(probs)
        else:
            raise ValueError(f"Unknown base_score_input: {self.base_score_input}")

        # missingness augmentation operates on calibrated probs
        base_cal, score_mask = self._apply_model_dropout(base_cal)

        # optional noise on calibrated probs
        noise = float(getattr(self.cfg, "score_noise", 0.0))
        if self.training and noise > 0.0:
            base_cal = (base_cal + noise * torch.randn_like(base_cal)).clamp(0.0, 1.0)

        alpha = None
        s_meta = None

        # ---- gating scalar ----
        if self.use_gating:
            gate_logits = self.gate(pair_emb)
            t = max(float(getattr(self.cfg, "alpha_temperature", 1.0)), 1e-6)
            alpha = torch.softmax(gate_logits / t, dim=-1)

            if getattr(self.cfg, "mask_gating", False):
                alpha = alpha * score_mask
                alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-12)

            s_meta = (alpha * base_cal).sum(dim=-1, keepdim=True)

        # ---- MLP path ----
        if self.cfg.arch == "mlp":
            chunks = []
            if s_meta is not None:
                chunks.append(s_meta)

            chunks.append(base_cal)

            if getattr(self.cfg, "use_score_mask", False):
                chunks.append(score_mask)

            if extra.size(1) > 0:
                chunks.append(extra)

            chunks.append(pair_emb)

            if self.use_token_attention:
                tok = self._build_tokens(base_cal, pair_emb, score_mask=score_mask)
                tok = self._apply_token_attention(tok)
                tok_pool = tok.mean(dim=1)
                chunks.append(tok_pool)

            x = torch.cat(chunks, dim=-1)
            h = self.backbone(x)
            logits = self.head(h)
            return {"logits": logits, "alpha": alpha, "scores_cal": base_cal, "score_mask": score_mask}

        # ---- RNN/LSTM path ----
        tok = self._build_tokens(base_cal, pair_emb, score_mask=score_mask)
        tok = self._apply_token_attention(tok)

        out, hidden = self.rnn(tok)
        h_rnn = self._rnn_pool(out, hidden)

        fusion_chunks = [h_rnn, pair_emb]
        if extra.size(1) > 0:
            fusion_chunks.append(extra)
        if s_meta is not None:
            fusion_chunks.append(s_meta)
        if getattr(self.cfg, "use_score_mask", False):
            fusion_chunks.append(score_mask)

        h = self.fusion(torch.cat(fusion_chunks, dim=-1))
        logits = self.head(h)
        return {"logits": logits, "alpha": alpha, "scores_cal": base_cal, "score_mask": score_mask}
