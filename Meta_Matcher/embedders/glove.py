from typing import List
import numpy as np
import re
from .base import Embedder

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())

class GloveEmbedder(Embedder):
    """
    Lädt GloVe als KeyedVectors (word2vec format).
    Tipp: GloVe .txt vorher konvertieren oder fertige word2vec-Version nutzen.
    """
    def __init__(self, glove_w2v_path: str, binary: bool = False):
        from gensim.models import KeyedVectors
        self.kv = KeyedVectors.load_word2vec_format(glove_w2v_path, binary=binary)
        self._dim = int(self.kv.vector_size)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            vecs = [self.kv[w] for w in toks if w in self.kv]
            if vecs:
                v = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
                n = np.linalg.norm(v) + 1e-12
                out[i] = v / n
        return out
