"""End-to-end behaviour of the agentic pipeline over the bundled corpus."""

import json
import logging

import pytest

from agentic_rag import Mode, PipelineConfig, Principal, RagPipeline
from agentic_rag.generation import ABSTAIN_TEXT
from agentic_rag.vectorstore import QdrantVectorStore

EMPLOYEE = Principal.of("u1", [])
HR = Principal.of("u2", ["hr"])


def test_answers_with_citation(pipeline: RagPipeline) -> None:
    a = pipeline.ask("How many days of annual leave does a permanent employee get?", EMPLOYEE)
    assert not a.abstained
    assert "24 working days" in a.text
    assert a.citations[0].doc_id == "hr-leave-policy"
    assert "[1]" in a.text


def test_multi_part_question_uses_two_passages(pipeline: RagPipeline) -> None:
    a = pipeline.ask("What is the per diem in Belgium and who approves an expense of 2,000 EUR?", EMPLOYEE)
    assert "45 EUR" in a.text
    assert "cost-centre owner" in a.text
    plan = next(t for t in a.trace if t["step"] == "plan")
    assert len(plan["parts"]) == 2  # type: ignore[arg-type]


def test_partial_answer_names_what_is_missing(pipeline: RagPipeline) -> None:
    a = pipeline.ask(
        "How many days of annual leave do I get and what is the company's stock ticker?", EMPLOYEE
    )
    assert not a.abstained
    assert "24" in a.text
    assert "I couldn't find:" in a.text
    assert "ticker" in a.text.split("I couldn't find:")[1]


def test_abstains_when_nothing_covers_the_question(pipeline: RagPipeline) -> None:
    a = pipeline.ask("What is the company's stock ticker?", EMPLOYEE)
    assert a.abstained
    assert a.text == ABSTAIN_TEXT
    assert a.citations == []
    decide = a.trace[-1]
    assert decide["step"] == "decide"
    assert decide["abstained"] is True


def test_permissions_change_the_answer(pipeline: RagPipeline) -> None:
    q = "What is the salary range for band E4?"
    denied = pipeline.ask(q, EMPLOYEE)
    allowed = pipeline.ask(q, HR)
    assert denied.abstained
    assert "70,000" not in denied.text
    assert "70,000" in allowed.text
    assert allowed.citations[0].doc_id == "hr-salary-bands-2026"


@pytest.mark.parametrize("mode", list(Mode))
def test_no_mode_returns_unreadable_chunks(pipeline: RagPipeline, mode: Mode) -> None:
    pipeline.config.mode = mode
    try:
        for query in ["salary band", "restructuring Antwerp close", "Q3 results revenue", "pricing 2027"]:
            hits = pipeline.search(query, EMPLOYEE, 20)
            assert all("all" in h.chunk.groups for h in hits), query
    finally:
        pipeline.config.mode = Mode.HYBRID


def test_poisoned_document_is_quarantined(pipeline: RagPipeline) -> None:
    a = pipeline.ask("Do I need receipts for expense claims?", EMPLOYEE)
    assert "receipts" in a.text
    assert "no longer required" not in a.text
    quarantine = [t for t in a.trace if t["step"] == "quarantine"]
    assert quarantine
    assert "ext-carrier-newsletter#1" in quarantine[0]["chunks"]  # type: ignore[operator]


def test_quarantine_can_be_disabled_for_comparison(pipeline: RagPipeline) -> None:
    pipeline.config.quarantine = False
    try:
        a = pipeline.ask("Do I need receipts for expense claims?", EMPLOYEE)
        assert not [t for t in a.trace if t["step"] == "quarantine"]
    finally:
        pipeline.config.quarantine = True


def test_query_log_has_no_question_text(pipeline: RagPipeline, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="arag.query"):
        pipeline.ask("What is the minimum password length for my account?", EMPLOYEE)
    record = next(r for r in caplog.records if r.name == "arag.query")
    fields = record.fields  # type: ignore[attr-defined]
    assert "password" not in json.dumps(fields)
    assert fields["user"] == "u1"
    assert "latency_ms" in fields


def test_answer_serialises(pipeline: RagPipeline) -> None:
    data = pipeline.ask("How long is warehouse CCTV footage kept?", EMPLOYEE).to_dict()
    json.dumps(data)
    assert set(data) == {"answer", "abstained", "reason", "citations", "trace"}


@pytest.mark.qdrant
def test_qdrant_backend_gives_same_permissions() -> None:
    pytest.importorskip("qdrant_client")
    rag = RagPipeline.from_corpus(store=QdrantVectorStore(location=":memory:"), config=PipelineConfig())
    q = "What is the salary range for band E4?"
    assert rag.ask(q, EMPLOYEE).abstained
    assert "88,000" in rag.ask(q, HR).text
    hits = rag.search("salary band engineering", EMPLOYEE, 20)
    assert all("all" in h.chunk.groups for h in hits)
