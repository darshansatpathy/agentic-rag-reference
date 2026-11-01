# ADR-0006: A bounded, traced agentic loop rather than an open-ended agent

**Status:** accepted

## Context

Single-shot RAG fails on compound questions, where one query has to serve two needs, and
it fails silently when the first retrieval misses. A fully autonomous agent with tools
fixes that but is hard to bound, test and explain.

## Decision

A fixed loop: plan sub-questions, then for each one retrieve, re-rank, screen, check
sufficiency, and if needed rewrite the query from the closest passage and retry, up to
`max_hops` (default 2). Then answer the covered parts and name the missing ones. The
loop can only read; it calls nothing that changes state. Every step is written to the
answer's `trace`.

## Consequences

- In the eval ablation, planning raises multi-part accuracy from 0% (single-shot) to
  100% and cuts wrong answers from 7 to 1. The retry hop adds nothing on the current
  question set; the ablation stays in the report so that is visible either way.
- Cost and latency are bounded: at most parts × hops retrievals and one answer call.
- The trace makes each answer explainable to an auditor or a support engineer.
- Questions needing reasoning across many documents (for example, comparisons over time)
  are out of scope for this loop.
