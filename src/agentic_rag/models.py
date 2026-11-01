"""Core data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    groups: frozenset[str]
    source: str  # "internal" or "external"
    sections: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    groups: frozenset[str]
    source: str = "internal"
    position: int = 0

    @property
    def context_text(self) -> str:
        """What retrieval sees: the chunk with its document and section as a header."""
        return f"{self.title}. {self.section}. {self.text}"


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    number: int
    chunk_id: str
    doc_id: str
    title: str
    section: str


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    abstained: bool
    reason: str = ""
    hits: list[Hit] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.text,
            "abstained": self.abstained,
            "reason": self.reason,
            "citations": [
                {
                    "n": c.number,
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "section": c.section,
                    "chunk_id": c.chunk_id,
                }
                for c in self.citations
            ],
            "trace": self.trace,
        }
