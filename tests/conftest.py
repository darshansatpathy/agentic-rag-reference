from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from agentic_rag import RagPipeline
from agentic_rag.models import Chunk, Hit


@pytest.fixture(scope="session")
def pipeline() -> RagPipeline:
    """The default offline pipeline over the bundled corpus (built once per test run)."""
    return RagPipeline.from_corpus()


@pytest.fixture
def make_chunk() -> Callable[..., Chunk]:
    def _make(
        text: str,
        *,
        doc_id: str = "doc",
        section: str = "Section",
        title: str = "Title",
        groups: tuple[str, ...] = ("all",),
        position: int = 0,
    ) -> Chunk:
        return Chunk(
            chunk_id=f"{doc_id}#{position}",
            doc_id=doc_id,
            title=title,
            section=section,
            text=text,
            groups=frozenset(groups),
            position=position,
        )

    return _make


@pytest.fixture
def make_hit(make_chunk: Callable[..., Chunk]) -> Callable[..., Hit]:
    def _make(text: str, score: float = 1.0, **kwargs: object) -> Hit:
        return Hit(chunk=make_chunk(text, **kwargs), score=score)

    return _make


@pytest.fixture
def write_corpus(tmp_path: Path) -> Callable[[dict[str, str]], Path]:
    def _write(files: dict[str, str]) -> Path:
        for name, body in files.items():
            (tmp_path / f"{name}.md").write_text(body, encoding="utf-8")
        return tmp_path

    return _write
