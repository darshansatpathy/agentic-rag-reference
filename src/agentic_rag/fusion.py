"""Rank fusion.

Reciprocal Rank Fusion combines ranked lists using only positions:

    score(d) = sum over lists of 1 / (k + rank(d))

It needs no score normalisation, so a BM25 score of 14.2 and a cosine of 0.61 can be
combined without tuning, and it is robust to one retriever having a bad day.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]], k: int = 60
) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse ranked id lists. Returns (id, fused score, {retriever: rank}) best first."""
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ids in rankings.items():
        for rank, item in enumerate(ids, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(item, {})[name] = rank
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(item, score, ranks[item]) for item, score in ordered]


def weighted_score_fusion(
    scored: Mapping[str, Sequence[tuple[str, float]]], weights: Mapping[str, float]
) -> list[tuple[str, float]]:
    """Min-max normalise each list's scores, then take a weighted sum. Kept for comparison."""
    fused: dict[str, float] = {}
    for name, items in scored.items():
        if not items:
            continue
        values = [s for _, s in items]
        lo, hi = min(values), max(values)
        span = (hi - lo) or 1.0
        w = weights.get(name, 1.0)
        for item, s in items:
            fused[item] = fused.get(item, 0.0) + w * (s - lo) / span
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
