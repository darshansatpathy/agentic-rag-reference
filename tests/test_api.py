import pytest
from fastapi.testclient import TestClient

from agentic_rag import RagPipeline
from agentic_rag.api import create_app
from agentic_rag.config import Settings
from agentic_rag.pipeline import default_corpus


def _client(pipeline: RagPipeline, token: str | None = None) -> TestClient:
    settings = Settings(corpus_dir=default_corpus(), service_token=token)
    return TestClient(create_app(pipeline=pipeline, settings=settings))


@pytest.fixture
def client(pipeline: RagPipeline) -> TestClient:
    return _client(pipeline)


def test_health(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["chunks"] == 94


def test_ask_returns_cited_answer_and_request_id(client: TestClient) -> None:
    r = client.post(
        "/v1/ask",
        json={"question": "How long is warehouse CCTV footage kept?"},
        headers={"X-Request-ID": "req-123"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "30 days" in body["answer"]
    assert body["citations"][0]["doc_id"] == "it-data-retention"
    assert body["request_id"] == "req-123"
    assert r.headers["X-Request-ID"] == "req-123"
    assert "trace" not in body


def test_groups_header_controls_access(client: TestClient) -> None:
    q = {"question": "What is the salary range for band E4?"}
    assert client.post("/v1/ask", json=q).json()["abstained"] is True
    allowed = client.post("/v1/ask", json=q, headers={"X-User-Groups": "HR, finance"}).json()
    assert "70,000" in allowed["answer"]


def test_invalid_group_and_body_are_rejected(client: TestClient) -> None:
    r = client.post("/v1/ask", json={"question": "x"}, headers={"X-User-Groups": "hr; drop"})
    assert r.status_code == 400
    assert client.post("/v1/ask", json={"question": ""}).status_code == 422
    assert client.post("/v1/ask", json={"question": "x" * 2001}).status_code == 422


def test_bad_request_id_is_replaced(client: TestClient) -> None:
    r = client.get("/healthz", headers={"X-Request-ID": "bad id with spaces"})
    assert r.headers["X-Request-ID"] != "bad id with spaces"
    assert len(r.headers["X-Request-ID"]) == 32


def test_trace_on_request(client: TestClient) -> None:
    body = client.post(
        "/v1/ask", json={"question": "Do I need receipts for expense claims?", "include_trace": True}
    ).json()
    assert any(step["step"] == "quarantine" for step in body["trace"])


def test_search_respects_permissions(client: TestClient) -> None:
    hits = client.post("/v1/search", json={"query": "salary band E4", "k": 10}).json()["hits"]
    assert all(h["doc_id"] != "hr-salary-bands-2026" for h in hits)
    assert client.post("/v1/search", json={"query": "q", "k": 0}).status_code == 422


def test_service_token_is_enforced(pipeline: RagPipeline) -> None:
    client = _client(pipeline, token="s3cret")
    q = {"question": "How long is warehouse CCTV footage kept?"}
    assert client.post("/v1/ask", json=q).status_code == 401
    assert client.post("/v1/ask", json=q, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/v1/ask", json=q, headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/healthz").status_code == 200  # probes stay open


def test_metrics(client: TestClient) -> None:
    client.post("/v1/ask", json={"question": "What is the company's stock ticker?"})
    client.post("/v1/ask", json={"question": "How long is warehouse CCTV footage kept?"})
    text = client.get("/metrics").text
    assert 'arag_http_requests_total{route="/v1/ask",status="200"}' in text
    assert 'arag_answers_total{outcome="abstained"}' in text
    assert 'arag_answers_total{outcome="answered"}' in text
    assert 'arag_request_seconds_bucket{route="/v1/ask",le="+Inf"}' in text
