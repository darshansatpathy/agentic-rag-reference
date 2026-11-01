"""Embedders behind one interface.

- `LsaEmbedder` (default): TF-IDF reduced with truncated SVD, fitted on the corpus. Runs
  offline in milliseconds with only numpy, so the repo works with no keys and no model
  downloads. It captures co-occurrence, not world knowledge, so it is the baseline.
- `OpenAICompatibleEmbedder`: any `/embeddings` endpoint (OpenAI, Azure OpenAI, vLLM,
  Ollama, LiteLLM, TEI). Swap it in with `ARAG_EMBEDDER=openai` and re-run the eval.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from agentic_rag.text import tokens

Matrix = npt.NDArray[np.float32]


class Embedder(Protocol):
    name: str

    def fit(self, texts: Sequence[str]) -> None: ...

    def embed(self, texts: Sequence[str]) -> Matrix: ...


def _l2(m: Matrix) -> Matrix:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: Matrix = np.asarray(m / norms, dtype=np.float32)
    return out


@dataclass
class LsaEmbedder:
    dims: int = 96
    name: str = "lsa"
    _vocab: dict[str, int] = field(default_factory=dict)
    _idf: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    _components: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))

    def fit(self, texts: Sequence[str]) -> None:
        docs = [Counter(tokens(t)) for t in texts]
        vocab = sorted({term for d in docs for term in d})
        self._vocab = {t: i for i, t in enumerate(vocab)}
        n = len(docs)
        df = np.zeros(len(vocab))
        for d in docs:
            for term in d:
                df[self._vocab[term]] += 1
        self._idf = np.log((1 + n) / (1 + df)) + 1.0
        tfidf = self._tfidf(docs)
        k = max(1, min(self.dims, min(tfidf.shape) - 1))
        _, _, vt = np.linalg.svd(tfidf, full_matrices=False)
        self._components = vt[:k]

    def embed(self, texts: Sequence[str]) -> Matrix:
        if not self._vocab:
            raise RuntimeError("LsaEmbedder.fit must be called before embed")
        tfidf = self._tfidf([Counter(tokens(t)) for t in texts])
        return _l2((tfidf @ self._components.T).astype(np.float32))

    def _tfidf(self, docs: list[Counter[str]]) -> npt.NDArray[np.float64]:
        m = np.zeros((len(docs), len(self._vocab)))
        for row, d in enumerate(docs):
            for term, count in d.items():
                col = self._vocab.get(term)
                if col is not None:
                    m[row, col] = (1 + math.log(count)) * self._idf[col]
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        result: npt.NDArray[np.float64] = m / norms
        return result


@dataclass
class OpenAICompatibleEmbedder:
    base_url: str
    model: str
    api_key: str = ""
    azure_api_version: str | None = None
    batch_size: int = 64
    timeout_seconds: float = 60.0
    transport: Any = None
    name: str = "openai-compatible"

    @classmethod
    def from_env(cls) -> OpenAICompatibleEmbedder:
        return cls(
            base_url=os.environ.get(
                "ARAG_EMBED_BASE_URL", os.environ.get("ARAG_LLM_BASE_URL", "https://api.openai.com/v1")
            ),
            model=os.environ.get("ARAG_EMBED_MODEL", "text-embedding-3-small"),
            api_key=os.environ.get("ARAG_EMBED_API_KEY", os.environ.get("ARAG_LLM_API_KEY", "")),
            azure_api_version=os.environ.get("ARAG_AZURE_API_VERSION") or None,
        )

    def fit(self, texts: Sequence[str]) -> None:
        """Hosted embedders need no fitting."""

    def embed(self, texts: Sequence[str]) -> Matrix:
        import httpx

        headers = {"Content-Type": "application/json"}
        params: dict[str, str] = {}
        if self.azure_api_version:
            headers["api-key"] = self.api_key
            params["api-version"] = self.azure_api_version
        elif self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        vectors: list[list[float]] = []
        with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                resp = client.post(
                    self.base_url.rstrip("/") + "/embeddings",
                    headers=headers,
                    params=params,
                    json={"model": self.model, "input": batch},
                )
                resp.raise_for_status()
                data = sorted(resp.json()["data"], key=lambda d: d["index"])
                vectors.extend(d["embedding"] for d in data)
        return _l2(np.asarray(vectors, dtype=np.float32))


def embedder_from_env() -> Embedder:
    kind = os.environ.get("ARAG_EMBEDDER", "lsa").lower()
    if kind == "openai":
        return OpenAICompatibleEmbedder.from_env()
    if kind == "lsa":
        return LsaEmbedder()
    raise ValueError(f"unknown ARAG_EMBEDDER '{kind}' (use 'lsa' or 'openai')")
