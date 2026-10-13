import hashlib
import math
import re

from app.errors import EmbeddingUnavailableError

_WORD = re.compile(r"\w+")


class FakeEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors.

    Uses md5 rather than hash(), because Python randomizes hash() for strings per process.
    """

    def __init__(self, dimension: int = 384, max_input_tokens: int = 510, fail: bool = False):
        self.dimension = dimension
        self.max_input_tokens = max_input_tokens
        self.fail = fail

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _vector(self, text: str) -> list[float]:
        if self.fail:
            raise EmbeddingUnavailableError("Fake embedding failure.")
        vector = [0.0] * self.dimension
        for word in _WORD.findall(text.lower()):
            slot = int.from_bytes(hashlib.md5(word.encode()).digest()[:4], "big") % self.dimension
            vector[slot] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]
