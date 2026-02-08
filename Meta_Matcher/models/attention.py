import torch
import torch.nn as nn
from typing import Tuple

class ScoreAttention(nn.Module):
    def __init__(
        self,
        n_models: int,
        context_dim: int,
        hidden: int,
        temperature: float = 1.0,
        score_dropout: float = 0.0,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.score_dropout = float(score_dropout)

        token_dim = 1 + context_dim
        self.scorer = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, scores: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        B, M = scores.shape

        # Optional: randomly drop some scores during training (exploration)
        if self.training and self.score_dropout > 0:
            keep = (torch.rand_like(scores) > self.score_dropout).float()
            scores = scores * keep

        s = scores.unsqueeze(-1)  # (B, M, 1)
        c = context.unsqueeze(1).expand(B, M, context.shape[-1])  # (B, M, C)
        tok = torch.cat([s, c], dim=-1)  # (B, M, 1+C)

        logits = self.scorer(tok).squeeze(-1)  # (B, M)

        T = max(self.temperature, 1e-6)
        attn = torch.softmax(logits / T, dim=-1)  # (B, M)

        pooled = (attn.unsqueeze(-1) * tok).sum(dim=1)  # (B, 1+C)
        return pooled, attn
