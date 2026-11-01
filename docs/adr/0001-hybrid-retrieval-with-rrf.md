# ADR-0001: Hybrid retrieval fused with Reciprocal Rank Fusion

**Status:** accepted

## Context

Enterprise questions mix two kinds of need. Some hinge on exact tokens: "SEV2", "band
E4", "10,000 EUR", a product code. Others are phrased nothing like the document:
"vacation days" for "annual leave". Lexical search is strong on the first and weak on the
second; dense search is the reverse.

## Decision

Run BM25 and dense search independently, each inside the caller's permissions, and fuse
the two rankings with Reciprocal Rank Fusion (k = 60). A re-ranker then orders the fused
short list.

## Alternatives considered

- **Dense only.** Simpler; misses exact identifiers and numbers, which policy questions
  are full of.
- **Weighted score fusion.** Requires calibrating BM25 scores against cosine similarities,
  which shift with corpus and model. Kept in `fusion.py` for comparison only.
- **Learned fusion.** Needs labelled relevance data at a scale most teams don't have at
  the start.

## Consequences

- RRF needs no tuning and is robust to either retriever having an off day.
- On this small, lexically written corpus BM25 alone is as good as hybrid (the eval
  table shows it). Hybrid earns its place on real corpora with real embeddings, where
  paraphrase is common; the eval will show if that stops being true.
