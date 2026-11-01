import pytest

from agentic_rag import RagPipeline
from agentic_rag.evaluation import (
    EvalReport,
    QuestionResult,
    evaluate,
    load_questions,
    load_thresholds,
    retrieval_metrics,
)


def test_retrieval_metrics() -> None:
    m = retrieval_metrics(["x", "a", "y", "b"], ["a", "b"])
    assert m["recall@5"] == 1.0
    assert m["mrr"] == 0.5
    assert 0 < m["ndcg@10"] < 1
    assert retrieval_metrics(["a"], ["a"]) == {"recall@5": 1.0, "mrr": 1.0, "ndcg@10": 1.0}
    assert retrieval_metrics(["x"], ["a"])["mrr"] == 0.0


def test_question_set_is_well_formed() -> None:
    qs = load_questions()
    assert len(qs) >= 50
    assert len({q.qid for q in qs}) == len(qs)
    for q in qs:
        assert q.expect_abstain or q.answer_contains, q.qid
        assert q.expect_abstain or q.gold_docs, q.qid


def _result(**kw: object) -> QuestionResult:
    base = dict(
        qid="q",
        qtype="factual",
        correct=True,
        abstained=False,
        expected_abstain=False,
        leaked=False,
        injected=False,
        citation_precision=1.0,
        grounded=True,
        answer="a",
    )
    base.update(kw)
    return QuestionResult(**base)  # type: ignore[arg-type]


def test_gate_fails_on_any_leak() -> None:
    report = EvalReport({"hybrid+rerank": {"recall@5": 1.0}}, [_result(), _result(leaked=True)])
    failures = report.check({"max_leaks": 0.0, "min_recall@5": 0.9})
    assert failures == ["leaks = 1.000 exceeds 0.0"]


@pytest.fixture(scope="module")
def report(pipeline: RagPipeline) -> EvalReport:
    return evaluate(pipeline, sweep=(0.5,))


def test_bundled_eval_passes_its_gate(report: EvalReport) -> None:
    assert report.check(load_thresholds()) == []


def test_safety_is_perfect(report: EvalReport) -> None:
    s = report.summary()
    assert s["leaks"] == 0
    assert s["injections_followed"] == 0
    assert s["groundedness"] == 1.0


def test_report_renders(report: EvalReport) -> None:
    md = report.markdown()
    assert "| hybrid+rerank |" in md
    assert "Permission leaks | 0" in md
    assert "Abstention threshold sweep" in md
    assert "Agent loop ablation" in md
