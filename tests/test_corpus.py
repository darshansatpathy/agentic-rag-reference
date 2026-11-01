import itertools
from pathlib import Path

import pytest

from agentic_rag.corpus import CorpusError, chunk_documents, load_corpus, parse_document
from agentic_rag.pipeline import default_corpus

DOC = """---
id: policy
title: "Policy: travel"
groups: [finance]
source: internal
---

# Policy: travel

## Per diem

Meals are 45 EUR per day. Hotels are booked centrally.

## Approvals

Managers approve up to 500 EUR.
"""


def test_parse_document_reads_front_matter_and_sections() -> None:
    doc = parse_document(DOC)
    assert doc.doc_id == "policy"
    assert doc.title == "Policy: travel"
    assert doc.groups == frozenset({"finance"})
    assert [s for s, _ in doc.sections] == ["Per diem", "Approvals"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("no front matter", "no front matter"),
        ("---\nid: [unclosed\n---\nbody", "invalid front matter"),
        ("---\ntitle: x\ngroups: [all]\n---\nbody", "no id"),
        ("---\nid: x\ntitle: x\n---\nbody", "no access groups"),
    ],
)
def test_parse_document_rejects_bad_input(text: str, message: str) -> None:
    with pytest.raises(CorpusError, match=message):
        parse_document(text)


def test_load_corpus_rejects_duplicates_and_empty_dirs(write_corpus, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(CorpusError, match=r"no \.md documents"):
        load_corpus(tmp_path)
    write_corpus({"a": DOC, "b": DOC})
    with pytest.raises(CorpusError, match="duplicate document ids"):
        load_corpus(tmp_path)


def test_chunks_carry_section_permissions_and_stable_ids() -> None:
    chunks = chunk_documents([parse_document(DOC)])
    assert [c.chunk_id for c in chunks] == ["policy#0", "policy#1"]
    assert chunks[0].section == "Per diem"
    assert all(c.groups == frozenset({"finance"}) for c in chunks)
    assert chunks[0].context_text.startswith("Policy: travel. Per diem.")


def test_long_sections_split_with_sentence_overlap() -> None:
    body = " ".join(f"Sentence number {i} has several words in it." for i in range(40))
    doc = parse_document(f"---\nid: long\ntitle: Long\ngroups: [all]\n---\n## Body\n\n{body}\n")
    chunks = chunk_documents([doc], max_words=60, overlap_sentences=1)
    assert len(chunks) > 3
    for a, b in itertools.pairwise(chunks):
        assert a.text.split(". ")[-1].rstrip(".") in b.text  # last sentence repeats
        assert len(a.text.split()) <= 70


def test_bundled_corpus_loads() -> None:
    docs = load_corpus(default_corpus())
    assert len(docs) == 30
    restricted = {d.doc_id for d in docs if "all" not in d.groups}
    assert {"hr-salary-bands-2026", "hr-restructuring-plan", "fin-q3-2026-results"} <= restricted
    assert any(d.source == "external" for d in docs)
