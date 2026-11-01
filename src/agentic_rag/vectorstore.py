"""Vector stores behind one interface, with access groups enforced inside the query.

Permission filtering happens before ranking, not after. Post-filtering a top-k list can
return nothing to a user whose documents rank below everyone else's; worse, a bug in the
post-filter leaks. Here a chunk the principal cannot read is never a candidate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from agentic_rag.embeddings import Matrix
from agentic_rag.models import Chunk


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], vectors: Matrix) -> None: ...

    def search(self, vector: Matrix, k: int, groups: frozenset[str]) -> list[tuple[Chunk, float]]: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: Matrix = np.zeros((0, 0), dtype=np.float32)

    def add(self, chunks: Sequence[Chunk], vectors: Matrix) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        self._chunks.extend(chunks)
        self._vectors = vectors if self._vectors.size == 0 else np.vstack([self._vectors, vectors])

    def search(self, vector: Matrix, k: int, groups: frozenset[str]) -> list[tuple[Chunk, float]]:
        if not self._chunks:
            return []
        mask = np.array([bool(c.groups & groups) for c in self._chunks])
        if not mask.any():
            return []
        scores = self._vectors @ vector.reshape(-1)
        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._chunks[i], float(scores[i])) for i in order if np.isfinite(scores[i])]


class QdrantVectorStore:
    """Qdrant with a payload filter on `groups`. Works against a server or in-process.

    `QdrantVectorStore(location=":memory:")` runs Qdrant in-process, which is how the
    tests exercise it without a server.
    """

    def __init__(
        self, collection: str = "chunks", location: str | None = None, url: str | None = None
    ) -> None:
        from qdrant_client import QdrantClient

        self._client: Any = QdrantClient(location=location, url=url)
        self._collection = collection
        self._chunks: dict[int, Chunk] = {}

    def add(self, chunks: Sequence[Chunk], vectors: Matrix) -> None:
        from qdrant_client import models as qm

        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                self._collection,
                vectors_config=qm.VectorParams(size=int(vectors.shape[1]), distance=qm.Distance.COSINE),
            )
        start = len(self._chunks)
        points = []
        for offset, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True)):
            pid = start + offset
            self._chunks[pid] = chunk
            points.append(
                qm.PointStruct(
                    id=pid,
                    vector=vec.tolist(),
                    payload={"chunk_id": chunk.chunk_id, "groups": sorted(chunk.groups)},
                )
            )
        self._client.upsert(self._collection, points=points)

    def search(self, vector: Matrix, k: int, groups: frozenset[str]) -> list[tuple[Chunk, float]]:
        from qdrant_client import models as qm

        flt = qm.Filter(must=[qm.FieldCondition(key="groups", match=qm.MatchAny(any=sorted(groups)))])
        result = self._client.query_points(
            self._collection, query=vector.reshape(-1).tolist(), query_filter=flt, limit=k
        )
        return [(self._chunks[int(p.id)], float(p.score)) for p in result.points]
