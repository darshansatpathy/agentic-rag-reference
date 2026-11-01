"""Load Markdown documents with front matter, and split them into chunks.

Front matter carries the document id, title, access groups and whether the source is
internal or external. Access groups travel with every chunk, so permissions are enforced
at retrieval time rather than trusted to the prompt.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import yaml

from agentic_rag.models import Chunk, Document
from agentic_rag.text import sentences

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)


class CorpusError(ValueError):
    pass


def parse_document(text: str, fallback_id: str = "") -> Document:
    match = _FRONT_MATTER.match(text)
    if not match:
        raise CorpusError(f"document {fallback_id or '?'} has no front matter")
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise CorpusError(f"document {fallback_id or '?'} has invalid front matter: {exc}") from exc
    body = match.group(2)
    doc_id = str(meta.get("id") or fallback_id)
    groups = meta.get("groups") or []
    if not doc_id:
        raise CorpusError("document has no id")
    if not groups:
        raise CorpusError(f"document {doc_id} has no access groups; refusing to index it open")
    sections: list[tuple[str, str]] = []
    heading = "Overview"
    buf: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if buf:
                sections.append((heading, " ".join(buf).strip()))
            heading, buf = line[3:].strip(), []
        elif line.startswith("# "):
            continue
        elif line.strip():
            buf.append(line.strip())
    if buf:
        sections.append((heading, " ".join(buf).strip()))
    return Document(
        doc_id=doc_id,
        title=str(meta.get("title") or doc_id),
        groups=frozenset(str(g) for g in groups),
        source=str(meta.get("source") or "internal"),
        sections=tuple(sections),
    )


def load_corpus(directory: str | Path) -> list[Document]:
    paths = sorted(Path(directory).glob("*.md"))
    if not paths:
        raise CorpusError(f"no .md documents in {directory}")
    docs = [parse_document(p.read_text(encoding="utf-8"), p.stem) for p in paths]
    ids = [d.doc_id for d in docs]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise CorpusError(f"duplicate document ids: {sorted(duplicates)}")
    return docs


def chunk_documents(docs: Iterable[Document], max_words: int = 90, overlap_sentences: int = 1) -> list[Chunk]:
    """Heading-aware chunking.

    Each section becomes one chunk. A section longer than `max_words` is split on
    sentence boundaries, repeating `overlap_sentences` sentences between pieces so a fact
    that spans a boundary is still retrievable whole.
    """
    chunks: list[Chunk] = []
    for doc in docs:
        position = 0
        for section, text in doc.sections:
            for piece in _split(text, max_words, overlap_sentences):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}#{position}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        section=section,
                        text=piece,
                        groups=doc.groups,
                        source=doc.source,
                        position=position,
                    )
                )
                position += 1
    return chunks


def _split(text: str, max_words: int, overlap: int) -> list[str]:
    if len(text.split()) <= max_words:
        return [text]
    sents = sentences(text)
    pieces: list[str] = []
    current: list[str] = []
    for sent in sents:
        if current and len(" ".join([*current, sent]).split()) > max_words:
            pieces.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.append(sent)
    if current:
        pieces.append(" ".join(current))
    return pieces
