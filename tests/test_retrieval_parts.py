"""BM25, embeddings, vector stores, fusion and re-ranking in isolation."""

import json

import httpx
import numpy as np
import pytest

from agentic_rag.bm25 import BM25Index
from agentic_rag.embeddings import LsaEmbedder, OpenAICompatibleEmbedder, embedder_from_env
from agentic_rag.fusion import reciprocal_rank_fusion, weighted_score_fusion
from agentic_rag.providers import ScriptedProvider
from agentic_rag.rerank import FeatureReranker, LLMReranker, _parse_scores, coverage
from agentic_rag.vectorstore import InMemoryVectorStore

TEXTS = [
    "Annual leave is 24 working days per calendar year.",
    "Expense claims must be submitted within 60 days with receipts.",
    "Forklift speed is limited to 10 km/h in outdoor yards.",
    "Salary band E4 ranges from 70,000 to 88,000 EUR.",
]


@pytest.fixture
def chunks(make_chunk):  # type: ignore[no-untyped-def]
    groups = [("all",), ("all",), ("all",), ("hr",)]
    return [
        make_chunk(t, doc_id=f"d{i}", groups=g) for i, (t, g) in enumerate(zip(TEXTS, groups, strict=True))
    ]


def test_bm25_ranks_and_filters(chunks) -> None:  # type: ignore[no-untyped-def]
    index = BM25Index(chunks)
    top = index.search("how many annual leave days", k=3)
    assert top[0][0] == 0
    assert index.search("salary band", k=3, allowed={0, 1, 2}) == []
    assert index.search("salary band", k=3)[0][0] == 3
    assert index.idf["leav"] > 0


def test_lsa_embedder_is_normalised_and_semantic(chunks) -> None:  # type: ignore[no-untyped-def]
    emb = LsaEmbedder(dims=3)
    emb.fit([c.context_text for c in chunks])
    m = emb.embed(TEXTS)
    assert m.shape[0] == 4
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)
    q = emb.embed(["leave days per year"])[0]
    assert int(np.argmax(m @ q)) == 0
    with pytest.raises(RuntimeError):
        LsaEmbedder().embed(["not fitted"])


def test_openai_compatible_embedder_batches_and_authenticates() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        inputs = json.loads(request.content)["input"]
        data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in reversed(list(enumerate(inputs)))]
        return httpx.Response(200, json={"data": data})

    emb = OpenAICompatibleEmbedder(
        base_url="https://llm.test/v1",
        model="m",
        api_key="k",
        batch_size=2,
        transport=httpx.MockTransport(handler),
    )
    out = emb.embed(["a", "bb", "ccc"])
    assert out.shape == (3, 2)
    assert len(seen) == 2
    assert seen[0].headers["Authorization"] == "Bearer k"
    assert out[2, 0] > out[0, 0]  # order restored from "index"

    azure = OpenAICompatibleEmbedder(
        base_url="https://x.openai.azure.com/openai/deployments/e",
        model="m",
        api_key="k",
        azure_api_version="2024-10-21",
        transport=httpx.MockTransport(handler),
    )
    azure.embed(["a"])
    assert seen[-1].headers["api-key"] == "k"
    assert seen[-1].url.params["api-version"] == "2024-10-21"


def test_embedder_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAG_EMBEDDER", "openai")
    assert isinstance(embedder_from_env(), OpenAICompatibleEmbedder)
    monkeypatch.setenv("ARAG_EMBEDDER", "nope")
    with pytest.raises(ValueError):
        embedder_from_env()


def test_in_memory_store_filters_before_scoring(chunks) -> None:  # type: ignore[no-untyped-def]
    store = InMemoryVectorStore()
    vectors = np.eye(4, dtype=np.float32)
    store.add(chunks, vectors)
    hits = store.search(vectors[3], k=4, groups=frozenset({"all"}))
    assert "d3" not in [c.doc_id for c, _ in hits]
    assert store.search(vectors[3], k=1, groups=frozenset({"hr"}))[0][0].doc_id == "d3"
    with pytest.raises(ValueError):
        store.add(chunks, vectors[:2])


def test_rrf_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion({"bm25": ["a", "b", "c"], "dense": ["b", "a", "d"]}, k=60)
    ids = [i for i, _, _ in fused]
    assert set(ids[:2]) == {"a", "b"}
    assert ids[-1] in {"c", "d"}
    a = next(r for r in fused if r[0] == "a")
    assert a[1] == pytest.approx(1 / 61 + 1 / 62)
    assert a[2] == {"bm25": 1, "dense": 2}
    assert weighted_score_fusion({"x": [("a", 2.0), ("b", 1.0)]}, {"x": 1.0})[0][0] == "a"


def test_coverage_weights_rare_terms() -> None:
    idf = {"compani": 0.1, "ticker": 5.0}
    assert coverage("company ticker", "the company", idf) < 0.1
    assert coverage("company ticker", "ticker", idf) > 0.9
    assert coverage("unseen words", "nothing", idf) == 0.0
    assert coverage("", "anything") == 0.0


def test_feature_reranker_prefers_covering_passage(make_hit) -> None:  # type: ignore[no-untyped-def]
    hits = [
        make_hit("Hotels are booked centrally.", score=1.0),
        make_hit("Meals are 45 EUR per day.", score=0.8),
    ]
    out = FeatureReranker().rerank("meal allowance per day 45", hits)
    assert out[0].chunk.text.startswith("Meals")
    assert "coverage" in out[0].features
    assert FeatureReranker().rerank("q", []) == []


def test_llm_reranker_orders_by_grade(make_hit) -> None:  # type: ignore[no-untyped-def]
    hits = [make_hit("first", position=0), make_hit("second", position=1), make_hit("third", position=2)]
    provider = ScriptedProvider(['{"scores": [1, 9]}'])
    out = LLMReranker(provider, max_candidates=2).rerank("q", hits)
    assert [h.chunk.text for h in out] == ["second", "first", "third"]
    assert _parse_scores("grades: 3, 12, -1", 4) == [3.0, 10.0, 0.0, 0.0]
