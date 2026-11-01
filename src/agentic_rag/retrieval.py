"""Hybrid retrieval: BM25 and dense search, each permission-filtered, fused with RRF."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from agentic_rag.bm25 import BM25Index
from agentic_rag.embeddings import Embedder
from agentic_rag.fusion import reciprocal_rank_fusion
from agentic_rag.models import Chunk, Hit
from agentic_rag.observability import span
from agentic_rag.vectorstore import InMemoryVectorStore, VectorStore


class Mode(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class Principal:
    """Who is asking. Every principal is in the 'all' group."""

    user_id: str = "anonymous"
    groups: frozenset[str] = field(default_factory=lambda: frozenset({"all"}))

    @classmethod
    def of(cls, user_id: str, groups: list[str] | set[str] | frozenset[str]) -> Principal:
        return cls(user_id=user_id, groups=frozenset({"all", *groups}))

    def can_read(self, chunk: Chunk) -> bool:
        return bool(chunk.groups & self.groups)


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        store: VectorStore | None = None,
        candidates: int = 30,
        rrf_k: int = 60,
    ) -> None:
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.embedder = embedder
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.bm25 = BM25Index(chunks)
        texts = [c.context_text for c in chunks]
        embedder.fit(texts)
        self.store: VectorStore = store or InMemoryVectorStore()
        self.store.add(chunks, embedder.embed(texts))

    def retrieve(self, query: str, principal: Principal, k: int = 10, mode: Mode = Mode.HYBRID) -> list[Hit]:
        allowed = {i for i, c in enumerate(self.chunks) if principal.can_read(c)}
        with span("arag.retrieve", {"arag.mode": mode.value, "arag.allowed_chunks": len(allowed)}):
            lexical: list[str] = []
            dense: list[str] = []
            if mode in (Mode.BM25, Mode.HYBRID):
                lexical = [
                    self.chunks[i].chunk_id for i, _ in self.bm25.search(query, self.candidates, allowed)
                ]
            if mode in (Mode.DENSE, Mode.HYBRID):
                qv = self.embedder.embed([query])[0]
                dense = [c.chunk_id for c, _ in self.store.search(qv, self.candidates, principal.groups)]
            rankings = {name: ids for name, ids in (("bm25", lexical), ("dense", dense)) if ids}
            fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)
            hits = [
                Hit(chunk=self.by_id[cid], score=score, ranks=ranks)
                for cid, score, ranks in fused
                if principal.can_read(self.by_id[cid])  # defence in depth
            ]
        return hits[:k]
