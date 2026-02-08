import numpy as np
import torch
from torch.utils.data import Dataset

class PairScoreDataset(Dataset):
    def __init__(self, scores: np.ndarray, pair_emb: np.ndarray, labels: np.ndarray):
        assert scores.ndim == 2
        assert pair_emb.ndim == 2
        assert scores.shape[0] == pair_emb.shape[0] == labels.shape[0]
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.pair_emb = torch.tensor(pair_emb, dtype=torch.float32)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return self.scores[idx], self.pair_emb[idx], self.labels[idx]
