# Agentic RAG Reference

[![CI](https://github.com/darshansatpathy/agentic-rag-reference/actions/workflows/ci.yml/badge.svg)](https://github.com/darshansatpathy/agentic-rag-reference/actions/workflows/ci.yml)
[![CodeQL](https://github.com/darshansatpathy/agentic-rag-reference/actions/workflows/codeql.yml/badge.svg)](https://github.com/darshansatpathy/agentic-rag-reference/actions/workflows/codeql.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![Typed](https://img.shields.io/badge/mypy-strict-informational)
![Permission leaks](https://img.shields.io/badge/permission%20leaks-0-success)
![License](https://img.shields.io/badge/license-MIT-green)

**Retrieval-augmented answers for an enterprise: permission-aware, cited, and willing to
say "I don't know".**

Most RAG demos answer questions. This one is built around the three ways enterprise RAG
actually fails in production, and measures each of them in CI:

1. **It shows people documents they are not allowed to see.** Here, permissions are
   applied *before* retrieval scores anything, in both the lexical and the vector index,
   and the eval fails the build on a single leak.
2. **It answers confidently when the documents do not contain the answer.** Here, every
   sub-question has to pass a sufficiency check, and the system abstains, or answers the
   part it can and names the part it can't.
3. **A document in the index tells the model what to do.** Here, retrieved text is
   screened for instructions, quarantined passages are kept out of the answer and shown in
   the trace, and model prompts fence sources as quoted data.

```
question ─▶ plan ─▶ for each sub-question:                                   ─▶ cited answer
                     ACL filter ─▶ BM25 + dense ─▶ RRF ─▶ re-rank ─▶ quarantine      or partial answer
                     └─ sufficient? no → rewrite query, retry (max 2 hops)            or abstention
```

> Reference implementation on a synthetic corpus (30 policy documents of a fictional
> logistics company, Aurora Freight). Runs fully offline with no API keys; every model
> component can be swapped for a hosted one through configuration.

---

## Try it in one minute

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
arag demo
```

```text
> an employee asks: What is the per diem in Belgium and who approves an expense of 2,000 EUR?
The daily meal allowance (per diem) is 45 EUR within the Netherlands and Belgium, 60 EUR elsewhere
in the EU, and 80 EUR outside the EU. [1] Expenses between 500 and 5,000 EUR also need the cost-
centre owner's approval. [2]
  [1] Travel and expense policy - Per diem (fin-expense-policy)
  [2] Travel and expense policy - Approval thresholds (fin-expense-policy)
  (quarantined as possible prompt injection: ext-carrier-newsletter#1)

> an employee asks: What is the salary range for band E4?
I can't answer that from the documents available to you.

> someone in HR asks: What is the salary range for band E4?
Band E4 ranges from 70,000 to 88,000 EUR. [1]
  [1] Salary bands 2026 (confidential) - Engineering bands (hr-salary-bands-2026)
```

Note what the employee gets for the salary question: the same refusal as for a question
the corpus cannot answer. A restricted document's existence is not revealed.

```bash
arag ask "How fast must a SEV2 incident be acknowledged and when is its post-incident review due?" --trace
arag search "forklift speed" --groups engineering
arag eval --check          # the same gate CI runs
arag serve                 # HTTP API on :8080
```

## Results

From `arag eval` on the bundled corpus and 58 questions
([evals/questions.yaml](evals/questions.yaml)): factual, paraphrased, multi-part,
unanswerable, permission-restricted and injection cases. Offline configuration (BM25 +
LSA embeddings, feature re-ranker, extractive answerer).

| Metric | Result | CI gate |
|---|---|---|
| Permission leaks (restricted fact in answer, or citation the user can't read) | **0** | must be 0 |
| Injected instructions followed | **0** | must be 0 |
| Groundedness (cited sentences supported by the cited passage) | **100%** | 100% |
| Citation precision | **100%** | ≥ 90% |
| Abstention on unanswerable or forbidden questions | **92%** (11/12) | ≥ 90% |
| Answer accuracy, answerable questions | **85%** | ≥ 75% |
| · factual / multi-part / permitted restricted / poisoned corpus | 94% / 100% / 100% / 100% | |
| · paraphrased | 50% | |
| Wrong answers (answered, but incorrect) | **1** | ≤ 1 |
| Retrieval Recall@5 (document level) | 0.974 | ≥ 0.9 |

**Where the errors are.** Retrieval finds the right document for 97% of questions, and
the four retriever configurations give the same answer accuracy, so retrieval is not the
bottleneck. The misses are paraphrases ("how long can a driver be behind the wheel")
that the offline answer stage cannot verify lexically against "daily driving time is 9
hours". It abstains on those instead of guessing: all six misses on answerable questions
are refusals, not wrong answers. Switching to hosted embeddings and the model
answerer (`ARAG_EMBEDDER=openai ARAG_ANSWERER=llm`) is the intended fix; the eval runs
unchanged against it.

The one wrong answer is kept in the suite on purpose: *"When was Aurora Freight
founded?"* matches the company name in a sustainability passage that contains years.
Lexical sufficiency can't tell "a sentence about the company with a date" from "the
founding date". It is listed in the report under the questions that failed.

The abstention threshold is chosen from a sweep in the same report, not by feel:

| `min_coverage` | Accuracy | Abstention recall | False abstentions |
|---|---|---|---|
| 0.40 | 85% | 83% | 15% |
| **0.50** (default) | **85%** | **92%** | **15%** |
| 0.60 | 77% | 92% | 23% |

And what the agent loop buys, from the ablation in the same report:

| Variant | Accuracy | Multi-part | Wrong answers |
|---|---|---|---|
| Single-shot: whole question, one retrieval | 69% | 0% | 7 |
| Plan sub-questions, no retry | 85% | 100% | 1 |
| Plan + retry up to 2 hops (default) | 85% | 100% | 1 |

Planning is what fixes compound questions. The retry hop changes nothing on this question
set; it is kept because it costs nothing when the first retrieval is sufficient, and the
ablation will show when it starts to matter on a real corpus.

## How it works

| Stage | Offline default | Production option | Why |
|---|---|---|---|
| **Chunking** | Heading-aware, ~90 words, one sentence overlap; chunk carries title and section | same | Headings carry meaning ("Rate limits") that the sentence alone does not. |
| **Permissions** | Group ACL per document, enforced as a pre-filter in both indexes, re-checked after fusion | Qdrant payload filter (`MatchAny` on groups) | Post-filtering top-k leaks through ranking and starves results. [ADR-0002](docs/adr/0002-acl-as-a-pre-filter.md) |
| **Lexical** | BM25 (own implementation, Snowball stemming) | same, or OpenSearch | Exact codes, numbers and names ("SEV2", "E4") are where dense search is weakest. |
| **Dense** | LSA (TF-IDF + SVD) | any OpenAI-compatible `/embeddings` (OpenAI, Azure OpenAI, vLLM, Ollama) | Paraphrase recall. |
| **Fusion** | Reciprocal Rank Fusion, k=60 | same | No score calibration between systems needed. [ADR-0001](docs/adr/0001-hybrid-retrieval-with-rrf.md) |
| **Re-rank** | Explainable features: IDF-weighted coverage, section match, number match | model grader (`ARAG_RERANKER=llm`) or a cross-encoder behind the same interface | Precision at the top of the list, which is all the answer stage reads. |
| **Agent loop** | Rule planner splits compound questions; per part, retrieve → check sufficiency → rewrite query → retry | model planner (`ARAG_PLANNER=llm`) | One query vector can't serve "what is X and who approves Y". [ADR-0006](docs/adr/0006-bounded-agentic-loop.md) |
| **Sufficiency** | IDF-weighted coverage of the *original* sub-question; when-questions need a time in the answer sentence | same, plus the model's own "not in sources" | [ADR-0003](docs/adr/0003-calibrated-abstention.md) |
| **Injection** | Pattern screen on retrieved chunks; quarantine with a trace entry | plus fenced `<source>` blocks and a data-not-instructions system prompt | [ADR-0005](docs/adr/0005-quarantine-retrieved-instructions.md) |
| **Answer** | Extractive: selects and cites sentences; cannot invent text | model answerer; citations verified, uncited answers become abstentions | [ADR-0004](docs/adr/0004-extractive-answerer-as-baseline.md) |

Every answer carries a `trace`: the plan, each retrieval hop with the top chunks and
their coverage, anything quarantined and why, and the final decision. That is what an
analyst needs when someone asks "why did it say that?".

More detail in [docs/architecture.md](docs/architecture.md) and the
[threat model](docs/threat-model.md), which maps controls to the OWASP Top 10 for LLM
applications.

## HTTP API

```bash
arag serve --port 8080
curl -s localhost:8080/v1/ask -H 'content-type: application/json' \
  -H 'X-User-Id: u123' -H 'X-User-Groups: hr' \
  -d '{"question": "What is the salary range for band E4?", "include_trace": true}'
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/ask` | Answer with citations, `abstained`, `reason`, optional `trace` |
| `POST /v1/search` | Ranked passages with per-retriever ranks (debugging relevance) |
| `GET /healthz` | Liveness and index size |
| `GET /metrics` | Prometheus: requests, latency histogram, answered / abstained / quarantined counts |

Identity comes from the authenticating gateway in front of the service (OIDC at the
edge), which forwards `X-User-Id` and `X-User-Groups`. Set `ARAG_SERVICE_TOKEN` and only
callers presenting it can assert identities. Every response has an `X-Request-ID`, which
is also on every log line. Question text is not logged by default.

## Configuration

All settings are environment variables with offline defaults. See
[.env.example](.env.example).

| Variable | Values |
|---|---|
| `ARAG_EMBEDDER` | `lsa` (default), `openai` (any OpenAI-compatible endpoint, Azure supported) |
| `ARAG_VECTOR_STORE` | `memory` (default), `qdrant` (`ARAG_QDRANT_URL`, or in-process when unset) |
| `ARAG_ANSWERER` / `ARAG_PLANNER` / `ARAG_RERANKER` | offline default or `llm` |
| `ARAG_MIN_COVERAGE`, `ARAG_MAX_HOPS` | abstention threshold (0.5) and retries per sub-question (2) |
| `ARAG_SERVICE_TOKEN` | shared secret for the upstream gateway |

`docker compose up` runs the API against a real Qdrant server.

## Engineering

- 78 tests, 99% line coverage (gate: 85%), including Qdrant in-process and the API.
- `ruff`, `mypy --strict`, Python 3.11 to 3.13 in CI.
- The evaluation gate runs on every push and writes its report to the run summary.
- The corpus is generated by [scripts/generate_corpus.py](scripts/generate_corpus.py), and
  CI checks the committed files match it.
- The container image is built and queried in CI: a cited answer, a refusal, an
  authorised answer and a 401.
- CodeQL (security-extended), pip-audit, Dependabot.

## Layout

```
src/agentic_rag/
  corpus.py        front matter, heading-aware chunking
  bm25.py          lexical index
  embeddings.py    LSA and OpenAI-compatible embedders
  vectorstore.py   in-memory and Qdrant stores, both ACL-filtered
  retrieval.py     principals and hybrid retrieval
  fusion.py        reciprocal rank fusion
  rerank.py        coverage, feature and model re-rankers
  safety.py        injection screen and quarantine
  agentic.py       planners and query rewriting
  generation.py    extractive and model answerers, citation checks
  pipeline.py      the agent loop
  evaluation.py    metrics, gate, report
  api.py, cli.py, config.py, observability.py
evals/             questions and CI thresholds
data/corpus/       generated documents
docs/              architecture, threat model, ADRs
```

## Extending it to your documents

1. Export documents as Markdown with front matter: `id`, `title`, `groups` (required: a
   document with no groups is rejected rather than indexed open), `source`.
2. Point `ARAG_CORPUS_DIR` at them and write 30 to 50 questions from real user traffic in
   the same format as `evals/questions.yaml`, including ones that should be refused.
3. Run `arag eval` before and after every retrieval change. If the leak count is ever
   not zero, stop.

## License

MIT. See [LICENSE](LICENSE).
