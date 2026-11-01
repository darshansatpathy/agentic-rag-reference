from pathlib import Path

import pytest

from agentic_rag.cli import main
from agentic_rag.config import ConfigError, Settings, build_pipeline
from agentic_rag.generation import ExtractiveAnswerer, LLMAnswerer
from agentic_rag.rerank import LLMReranker


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ARAG_EMBEDDER", "ARAG_ANSWERER", "ARAG_MIN_COVERAGE", "ARAG_SERVICE_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    s = Settings.from_env()
    assert (s.embedder, s.answerer, s.min_coverage, s.service_token) == ("lsa", "extractive", 0.5, None)


@pytest.mark.parametrize(("name", "value"), [("ARAG_EMBEDDER", "word2vec"), ("ARAG_MIN_COVERAGE", "high")])
def test_settings_reject_bad_values(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError, match=name):
        Settings.from_env()


def test_build_pipeline_wires_model_components(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAG_ANSWERER", "llm")
    monkeypatch.setenv("ARAG_RERANKER", "llm")
    monkeypatch.setenv("ARAG_LLM_BASE_URL", "https://llm.test/v1")
    rag = build_pipeline()
    assert isinstance(rag.answerer, LLMAnswerer)
    assert isinstance(rag.reranker, LLMReranker)


def test_build_pipeline_offline_and_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("qdrant_client")
    monkeypatch.setenv("ARAG_VECTOR_STORE", "qdrant")
    rag = build_pipeline()
    assert isinstance(rag.answerer, ExtractiveAnswerer)
    assert rag.search("annual leave", rag_principal(), 3)


def rag_principal():  # type: ignore[no-untyped-def]
    from agentic_rag import Principal

    return Principal.of("t", [])


def test_cli_demo(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "70,000 to 88,000" in out
    assert "quarantined as possible prompt injection" in out


def test_cli_ask_and_search(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "What is the salary range for band E4?", "--groups", "hr", "--trace"]) == 0
    out = capsys.readouterr().out
    assert "88,000" in out
    assert "trace:" in out
    assert main(["ask", "What is the company's stock ticker?", "--json"]) == 0
    assert '"abstained": true' in capsys.readouterr().out
    assert main(["search", "forklift speed", "-k", "2"]) == 0
    assert "ops-forklift-safety" in capsys.readouterr().out


def test_cli_eval_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    report = tmp_path / "report.md"
    assert main(["eval", "--check", "--report", str(report)]) == 0
    assert "Permission leaks" in report.read_text()
    assert summary.read_text().startswith("## RAG evaluation")
    strict = tmp_path / "strict.yaml"
    strict.write_text("min_answer_accuracy: 1.0\n")
    assert main(["eval", "--check", "--thresholds", str(strict), "--json"]) == 1
    assert "gate FAILED" in capsys.readouterr().err


def test_cli_config_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("ARAG_ANSWERER", "magic")
    assert main(["demo"]) == 2
    assert "configuration error" in capsys.readouterr().err
