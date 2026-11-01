"""The agentic layer: plan, retrieve per sub-question, check sufficiency, retry, then answer.

Single-shot RAG retrieves once for the whole question. That fails on compound questions
("what is X and who approves Y?") because one query vector has to serve two needs, and
it fails silently when the first retrieval misses. This loop makes those steps explicit:

1. **Plan**: split the question into sub-questions (rule-based offline; a model can do
   it when one is configured).
2. **Retrieve and re-rank** for each sub-question, inside the asker's permissions.
3. **Check sufficiency**: does the best passage cover the sub-question? If not, rewrite
   the query with terms from the closest passage (pseudo-relevance feedback) and try
   again, up to `max_hops`.
4. **Decide**: answer from the evidence, answer the parts that are covered and say what
   is missing, or abstain.

Every step lands in the answer's `trace`, so a reviewer can see why an answer happened.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from agentic_rag.models import Hit
from agentic_rag.providers import ModelProvider
from agentic_rag.text import tokens

_SPLIT = re.compile(
    r"\s*(?:,\s*)?\band\s+(?=(?:what|who|whom|which|when|where|why|how|is|are|do|does|can|must|should)\b)",
    re.IGNORECASE,
)


class Planner(Protocol):
    def decompose(self, question: str) -> list[str]: ...


@dataclass
class RulePlanner:
    """Split on 'and' followed by a question word; re-attach fragments that are too short."""

    min_terms: int = 2

    def decompose(self, question: str) -> list[str]:
        parts = [p.strip(" ?,") for p in _SPLIT.split(question) if p.strip(" ?,")]
        if len(parts) <= 1:
            return [question.strip()]
        merged: list[str] = []
        for part in parts:
            if merged and len(tokens(part)) < self.min_terms:
                # "and how often?" is meaningless alone: carry the subject forward.
                merged.append(f"{part} {merged[-1]}")
            else:
                merged.append(part)
        return [m + "?" for m in merged]


@dataclass
class LLMPlanner:
    provider: ModelProvider
    max_parts: int = 4

    def decompose(self, question: str) -> list[str]:
        prompt = (
            "Split the question into the smallest set of self-contained sub-questions needed "
            "to answer it. If it is already a single question, return it unchanged. Reply "
            'with JSON only: {"questions": ["..."]}\n\nQuestion: ' + question
        )
        reply = self.provider.complete([{"role": "user", "content": prompt}], []).content or ""
        try:
            data = json.loads(reply[reply.index("{") : reply.rindex("}") + 1])
            parts = [str(q).strip() for q in data["questions"] if str(q).strip()]
        except (ValueError, KeyError, TypeError):
            parts = []
        return parts[: self.max_parts] or [question]


def expand_query(query: str, hits: Sequence[Hit], extra_terms: int = 4) -> str:
    """Pseudo-relevance feedback: add the closest passage's distinctive heading terms."""
    if not hits:
        return query
    have = set(tokens(query))
    head = hits[0].chunk
    new = [t for t in tokens(f"{head.title} {head.section}") if t not in have]
    return f"{query} {' '.join(dict.fromkeys(new[:extra_terms]))}".strip()
