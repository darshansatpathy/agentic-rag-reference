# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-01

### Added

- Hybrid retrieval: BM25 and dense search, each permission-filtered, fused with
  Reciprocal Rank Fusion; feature re-ranker and model re-ranker.
- Permission-aware indexing: group ACLs enforced as a pre-filter in BM25, the in-memory
  store and Qdrant, with a post-fusion re-check. Documents without ACLs are rejected.
- Agentic loop: sub-question planning, sufficiency check against the original
  sub-question, pseudo-relevance-feedback query rewriting, bounded hops, full trace.
- Calibrated abstention with IDF-weighted coverage and answer-type checks; partial
  answers that name the missing parts.
- Prompt-injection quarantine for retrieved content; fenced sources for model prompts.
- Extractive answerer (default) and model answerer with citation verification.
- OpenAI-compatible embeddings and chat (OpenAI, Azure OpenAI, vLLM, Ollama).
- Evaluation suite (58 questions): retrieval metrics per retriever, answer accuracy,
  abstention, citation precision, groundedness, permission leaks, injection resistance,
  wrong answers, threshold sweep and agent-loop ablation; CI gate with thresholds.
- FastAPI service with request IDs, service-token check, Prometheus metrics; `arag` CLI.
- Synthetic 30-document corpus with a reproducible generator.
- CI on Python 3.11 to 3.13, CodeQL, pip-audit, Dependabot, container smoke test.
