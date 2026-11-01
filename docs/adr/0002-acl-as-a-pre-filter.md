# ADR-0002: Enforce document permissions as a pre-filter

**Status:** accepted

## Context

A common implementation retrieves the top k for everyone and then removes what the user
can't see. That leaks in two ways: the result count and ranking change depending on
hidden documents, and when the top k is dominated by restricted chunks the user gets
nothing useful. It also puts the security check in the one place a refactor is most
likely to drop it.

## Decision

Compute the set of readable chunks before scoring. BM25 scores only those; the vector
store applies the group filter inside the query (Qdrant payload filter). After fusion,
each hit is checked again as defence in depth. The `VectorStore` interface has no search
method without groups.

## Consequences

- Restricted documents can't influence what a user sees, including through ranking.
- The refusal for a forbidden document is the same as for a missing one.
- The eval counts leaks and CI fails on any; tests check every retrieval mode and the
  Qdrant backend.
- ACLs must be kept in sync with source systems at ingest. That is an operational duty
  this code can't remove.
