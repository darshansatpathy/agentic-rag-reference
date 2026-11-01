"""BM25 (Okapi) lexical retrieval over chunks."""

from __future__ import annotations

import math
from collections import Counter

from agentic_rag.models import Chunk
from agentic_rag.text import tokens


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._docs = [Counter(tokens(c.context_text)) for c in chunks]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @property
    def idf(self) -> dict[str, float]:
        return dict(self._idf)

    def search(self, query: str, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        """Return (chunk index, score) pairs, best first. `allowed` limits the candidates."""
        q = tokens(query)
        scores: list[tuple[int, float]] = []
        for i, doc in enumerate(self._docs):
            if allowed is not None and i not in allowed:
                continue
            score = 0.0
            length = self._lengths[i]
            for term in q:
                tf = doc.get(term, 0)
                if not tf:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self._avg or 1))
                score += idf * tf * (self.k1 + 1) / denom
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda s: (-s[1], s[0]))
        return scores[:k]
