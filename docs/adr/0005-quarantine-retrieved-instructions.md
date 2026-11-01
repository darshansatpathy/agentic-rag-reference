# ADR-0005: Quarantine retrieved instructions instead of silently dropping them

**Status:** accepted

## Context

Any document in the index can address the model: a vendor newsletter, a wiki page, an
email. A RAG system that pastes retrieved text into a prompt hands that text the same
authority as the user's question.

## Decision

Two layers. First, retrieved chunks with instruction-like text aimed at a model are
quarantined: excluded from the evidence and recorded in the trace with the chunk ID and
reason, so someone can review the document. Second, anything that reaches a model is
fenced as numbered `<source>` blocks, under a system prompt that treats source content as
data, and the answer must cite sources.

## Alternatives considered

- **Drop silently.** Hides a possibly poisoned document from the people who should fix it.
- **Rely on the system prompt alone.** Models follow it most of the time, which is not
  enough for content anyone outside the company can write.
- **A classifier model.** A reasonable later addition behind the same `screen` function.

## Consequences

- The eval includes questions whose top results contain a poisoned document; CI requires
  that no injected instruction appears in an answer.
- Pattern matching has false negatives. The structural defences bound the damage when
  it misses.
