from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Embedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return np.float32 [N, dim]."""
        ...