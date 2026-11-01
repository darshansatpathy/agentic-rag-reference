"""The end-to-end pipeline: permissions → hybrid retrieval → re-rank → quarantine → answer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from agentic_rag.agentic import Planner, RulePlanner, expand_query
from agentic_rag.corpus import chunk_documents, load_corpus
from agentic_rag.embeddings import Embedder, LsaEmbedder
from agentic_rag.generation import ABSTAIN_TEXT, Answerer, ExtractiveAnswerer
from agentic_rag.models import Answer, Chunk, Hit
from agentic_rag.observability import log_query, span
from agentic_rag.rerank import FeatureReranker, Reranker, coverage
from agentic_rag.retrieval import HybridRetriever, Mode, Principal
from agentic_rag.safety import screen
from agentic_rag.vectorstore import VectorStore

DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"
_PACKAGED_CORPUS = Path(__file__).resolve().parent / "data" / "corpus"


def default_corpus() -> Path:
    return _PACKAGED_CORPUS if _PACKAGED_CORPUS.exists() else DEFAULT_CORPUS


@dataclass
class PipelineConfig:
    retrieve_k: int = 8
    evidence_per_part: int = 3
    min_coverage: float = 0.5  # best passage must cover this share of the question's terms
    max_hops: int = 2
    mode: Mode = Mode.HYBRID
    rerank: bool = True
    quarantine: bool = True


@dataclass
class RagPipeline:
    retriever: HybridRetriever
    reranker: Reranker = field(default_factory=FeatureReranker)
    answerer: Answerer = field(default_factory=ExtractiveAnswerer)
    planner: Planner = field(default_factory=RulePlanner)
    config: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def from_corpus(
        cls,
        corpus_dir: str | Path | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        **kwargs: object,
    ) -> RagPipeline:
        chunks = chunk_documents(load_corpus(corpus_dir or default_corpus()))
        retriever = HybridRetriever(chunks, embedder or LsaEmbedder(), store)
        kwargs.setdefault("reranker", FeatureReranker(idf=retriever.bm25.idf))
        kwargs.setdefault("answerer", ExtractiveAnswerer(idf=retriever.bm25.idf))
        return cls(retriever=retriever, **kwargs)  # type: ignore[arg-type]

    @property
    def chunks(self) -> list[Chunk]:
        return self.retriever.chunks

    def search(self, query: str, principal: Principal, k: int | None = None) -> list[Hit]:
        hits = self.retriever.retrieve(query, principal, k or self.config.retrieve_k, self.config.mode)
        if self.config.rerank:
            with span("arag.rerank", {"arag.candidates": len(hits)}):
                hits = self.reranker.rerank(query, hits)
        return hits

    def ask(self, question: str, principal: Principal) -> Answer:
        started = time.perf_counter()
        trace: list[dict[str, object]] = []
        with span("arag.ask", {"arag.user": principal.user_id}):
            parts = self.planner.decompose(question)
            trace.append({"step": "plan", "parts": parts})

            evidence: list[Hit] = []
            covered_parts: list[str] = []
            missing_parts: list[str] = []
            quarantined_ids: list[str] = []
            for part in parts:
                query = part
                best: list[Hit] = []
                best_cov = -1.0
                for hop in range(1, self.config.max_hops + 1):
                    hits = self.search(query, principal)
                    if self.config.quarantine:
                        screened = screen(hits)
                        quarantined_ids += [h.chunk.chunk_id for h in screened.quarantined]
                        hits = screened.kept
                    # Judge sufficiency against the original sub-question, never the expanded
                    # query: expansion terms come from the passage and would inflate it.
                    cov = coverage(part, hits[0].chunk.context_text, self.retriever.bm25.idf) if hits else 0.0
                    trace.append(
                        {
                            "step": "retrieve",
                            "part": part,
                            "query": query,
                            "hop": hop,
                            "top": [h.chunk.chunk_id for h in hits[:3]],
                            "coverage": round(cov, 3),
                        }
                    )
                    if hits and cov > best_cov:
                        best, best_cov = hits, cov
                    if cov >= self.config.min_coverage or not hits:
                        break
                    query = expand_query(part, hits)
                if best and best_cov >= self.config.min_coverage:
                    covered_parts.append(part)
                    for h in best[: self.config.evidence_per_part]:
                        if h.chunk.chunk_id not in {e.chunk.chunk_id for e in evidence}:
                            evidence.append(h)
                else:
                    missing_parts.append(part)

            if quarantined_ids:
                trace.append(
                    {
                        "step": "quarantine",
                        "chunks": sorted(set(quarantined_ids)),
                        "reason": "instruction-like text in retrieved content",
                    }
                )

            if not covered_parts:
                answer = Answer(
                    ABSTAIN_TEXT,
                    [],
                    abstained=True,
                    reason="no passage you can access covers the question",
                    hits=evidence,
                    trace=trace,
                )
            else:
                with span("arag.generate", {"arag.evidence": len(evidence)}):
                    draft = self.answerer.answer(question, covered_parts, evidence)
                text = draft.text
                missing_parts += [p for p in draft.unanswered if p not in missing_parts]
                if missing_parts and not draft.abstained:
                    text += " I couldn't find: " + "; ".join(p.rstrip("?") for p in missing_parts) + "."
                answer = Answer(text, draft.citations, draft.abstained, draft.reason, evidence, trace)
            trace.append(
                {
                    "step": "decide",
                    "abstained": answer.abstained,
                    "covered": covered_parts,
                    "missing": missing_parts,
                }
            )

        log_query(
            {
                "user": principal.user_id,
                "groups": sorted(principal.groups),
                "parts": len(parts),
                "abstained": answer.abstained,
                "citations": [c.doc_id for c in answer.citations],
                "quarantined": len(set(quarantined_ids)),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
        return answer
