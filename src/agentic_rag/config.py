"""Settings from the environment, and building a pipeline from them.

Everything has an offline default, so `arag demo` and the test suite need no services
and no API keys. Each setting swaps one component for a production one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from agentic_rag.agentic import LLMPlanner, RulePlanner
from agentic_rag.embeddings import Embedder, LsaEmbedder, OpenAICompatibleEmbedder
from agentic_rag.generation import ExtractiveAnswerer, LLMAnswerer
from agentic_rag.pipeline import PipelineConfig, RagPipeline, default_corpus
from agentic_rag.providers import OpenAICompatibleProvider
from agentic_rag.rerank import FeatureReranker, LLMReranker
from agentic_rag.vectorstore import InMemoryVectorStore, QdrantVectorStore, VectorStore


class ConfigError(ValueError):
    pass


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in allowed:
        raise ConfigError(f"{name} must be one of {', '.join(allowed)} (got {value!r})")
    return value


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number (got {raw!r})") from exc


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path
    embedder: str = "lsa"  # lsa | openai
    vector_store: str = "memory"  # memory | qdrant
    qdrant_url: str | None = None
    answerer: str = "extractive"  # extractive | llm
    planner: str = "rule"  # rule | llm
    reranker: str = "feature"  # feature | llm
    min_coverage: float = 0.5
    max_hops: int = 2
    service_token: str | None = None  # shared secret the upstream gateway must present
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            corpus_dir=Path(os.environ.get("ARAG_CORPUS_DIR") or default_corpus()),
            embedder=_choice("ARAG_EMBEDDER", "lsa", ("lsa", "openai")),
            vector_store=_choice("ARAG_VECTOR_STORE", "memory", ("memory", "qdrant")),
            qdrant_url=os.environ.get("ARAG_QDRANT_URL") or None,
            answerer=_choice("ARAG_ANSWERER", "extractive", ("extractive", "llm")),
            planner=_choice("ARAG_PLANNER", "rule", ("rule", "llm")),
            reranker=_choice("ARAG_RERANKER", "feature", ("feature", "llm")),
            min_coverage=_float("ARAG_MIN_COVERAGE", 0.5),
            max_hops=int(_float("ARAG_MAX_HOPS", 2)),
            service_token=os.environ.get("ARAG_SERVICE_TOKEN") or None,
            log_level=os.environ.get("ARAG_LOG_LEVEL", "INFO"),
        )


def build_pipeline(settings: Settings | None = None) -> RagPipeline:
    s = settings or Settings.from_env()
    embedder: Embedder = OpenAICompatibleEmbedder.from_env() if s.embedder == "openai" else LsaEmbedder()
    store: VectorStore
    if s.vector_store == "qdrant":
        store = (
            QdrantVectorStore(url=s.qdrant_url) if s.qdrant_url else QdrantVectorStore(location=":memory:")
        )
    else:
        store = InMemoryVectorStore()
    pipeline = RagPipeline.from_corpus(
        s.corpus_dir,
        embedder=embedder,
        store=store,
        config=PipelineConfig(min_coverage=s.min_coverage, max_hops=s.max_hops),
    )
    needs_model = "llm" in (s.answerer, s.planner, s.reranker)
    provider = OpenAICompatibleProvider.from_env() if needs_model else None
    if provider is not None:
        if s.answerer == "llm":
            pipeline.answerer = LLMAnswerer(provider)
        if s.planner == "llm":
            pipeline.planner = LLMPlanner(provider)
        if s.reranker == "llm":
            pipeline.reranker = LLMReranker(provider)
    else:
        pipeline.answerer = ExtractiveAnswerer(idf=pipeline.retriever.bm25.idf)
        pipeline.planner = RulePlanner()
        pipeline.reranker = FeatureReranker(idf=pipeline.retriever.bm25.idf)
    return pipeline
