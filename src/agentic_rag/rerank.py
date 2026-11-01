"""Re-ranking behind one interface.

The first stage (BM25 + dense, fused) optimises recall over the whole corpus. A
re-ranker looks only at the short list and optimises precision at the top, which is what
the answer step consumes.

- `FeatureReranker` (default, offline): explainable features such as query-term
  coverage, number matches and section-title overlap, blended with the fused score.
  Its `coverage` feature also drives abstention.
- `LLMReranker`: asks a model to grade each candidate 0-10 for relevance. Plug in a
  cross-encoder the same way by implementing `rerank`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from agentic_rag.models import Hit
from agentic_rag.providers import ModelProvider
from agentic_rag.text import numbers, tokens


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]: ...


def coverage(query: str, text: str, idf: Mapping[str, float] | None = None) -> float:
    """Share of the query's content terms that appear in the text.

    With `idf`, each term is weighted by how rare it is in the corpus, so matching
    "ticker" counts for far more than matching "company". Terms the corpus has never seen
    get the highest weight: a question about something absent should not look covered.
    """
    q = set(tokens(query))
    if not q:
        return 0.0
    t = set(tokens(text))
    if idf is None:
        return len(q & t) / len(q)
    unseen = max(idf.values(), default=1.0)
    weights = {term: idf.get(term, unseen) for term in q}
    total = sum(weights.values()) or 1.0
    return sum(w for term, w in weights.items() if term in t) / total


@dataclass
class FeatureReranker:
    idf: Mapping[str, float] | None = field(default=None, repr=False)
    w_fused: float = 0.35
    w_coverage: float = 0.45
    w_section: float = 0.12
    w_numbers: float = 0.08

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        if not hits:
            return []
        top = max(h.score for h in hits) or 1.0
        q_numbers = numbers(query)
        q_terms = set(tokens(query))
        out: list[Hit] = []
        for h in hits:
            cov = coverage(query, h.chunk.context_text, self.idf)
            section_terms = set(tokens(f"{h.chunk.title} {h.chunk.section}"))
            sec = len(q_terms & section_terms) / len(q_terms) if q_terms else 0.0
            num = len(q_numbers & numbers(h.chunk.text)) / len(q_numbers) if q_numbers else 0.0
            fused = h.score / top
            score = self.w_fused * fused + self.w_coverage * cov + self.w_section * sec + self.w_numbers * num
            features = {
                **h.features,
                "fused_norm": round(fused, 4),
                "coverage": round(cov, 4),
                "section": round(sec, 4),
                "numbers": round(num, 4),
            }
            out.append(Hit(chunk=h.chunk, score=score, ranks=dict(h.ranks), features=features))
        out.sort(key=lambda x: (-x.score, x.chunk.chunk_id))
        return out


_SCORE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class LLMReranker:
    provider: ModelProvider
    max_candidates: int = 10

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        candidates = list(hits)[: self.max_candidates]
        listing = "\n".join(
            f"[{i}] {h.chunk.title} / {h.chunk.section}: {h.chunk.text}" for i, h in enumerate(candidates)
        )
        prompt = (
            "Grade how well each passage answers the question, 0 (irrelevant) to 10 (fully "
            'answers). Reply with JSON only: {"scores": [n0, n1, ...]} in passage order.\n\n'
            f"Question: {query}\n\nPassages:\n{listing}"
        )
        reply = self.provider.complete([{"role": "user", "content": prompt}], []).content or ""
        grades = _parse_scores(reply, len(candidates))
        out = [
            Hit(
                chunk=h.chunk,
                score=g / 10.0,
                ranks=dict(h.ranks),
                features={**h.features, "llm_grade": g, "coverage": coverage(query, h.chunk.context_text)},
            )
            for h, g in zip(candidates, grades, strict=True)
        ]
        out.sort(key=lambda x: (-x.score, x.chunk.chunk_id))
        return out + list(hits)[self.max_candidates :]


def _parse_scores(reply: str, n: int) -> list[float]:
    try:
        data = json.loads(reply[reply.index("{") : reply.rindex("}") + 1])
        values = [float(v) for v in data["scores"]]
    except (ValueError, KeyError, TypeError):
        values = [float(v) for v in _SCORE.findall(reply)]
    values = [min(10.0, max(0.0, v)) for v in values[:n]]
    return values + [0.0] * (n - len(values))
