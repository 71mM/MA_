from typing import List, Literal
import numpy as np
import re
from .base import Embedder

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())

class FastTextEmbedder(Embedder):
    """
    Option A (empfohlen): native fastText .bin -> OOV via Subword
    Option B: gensim .vec/.bin als KeyedVectors (weniger OOV)
    """
    def __init__(self, path: str, mode: Literal["native", "gensim"] = "native", binary: bool = True):
        self.mode = mode
        if mode == "native":
            import fasttext
            self.ft = fasttext.load_model(path)
            self._dim = int(self.ft.get_dimension())
        else:
            from gensim.models import KeyedVectors
            self.kv = KeyedVectors.load_word2vec_format(path, binary=binary)
            self._dim = int(self.kv.vector_size)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            if not toks:
                continue
            if self.mode == "native":
                vecs = [self.ft.get_word_vector(w) for w in toks]
            else:
                vecs = [self.kv[w] for w in toks if w in self.kv]
                if not vecs:
                    continue
            v = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
            n = np.linalg.norm(v) + 1e-12
            out[i] = v / n
        return out
