from __future__ import annotations

from kbdebugger.compat.langchain import Document
from kbdebugger.extraction.logging import build_chunked_documents_payload
from kbdebugger.extraction.reference_filter import (
    filter_reference_section,
    is_reference_heading,
    normalize_reference_heading,
)


def doc(text: str, *, headings: list[str] | None = None) -> Document:
    metadata = {}
    if headings is not None:
        metadata = {"dl_meta": {"headings": headings}}
    return Document(page_content=text, metadata=metadata)


def body_docs(count: int = 6) -> list[Document]:
    return [doc(f"Body paragraph {idx} with inline citation (Smith et al., 2021).") for idx in range(count)]


def test_reference_heading_normalization_variants() -> None:
    cases = {
        "References": "references",
        "REFERENCES": "references",
        "7 References": "references",
        "7. References": "references",
        "VII References": "references",
        "VII. References": "references",
        "References:": "references",
        "Works Cited": "works cited",
        "Literature Cited": "literature cited",
        "REFERENCES AND NOTES": "references and notes",
    }

    for raw, expected in cases.items():
        assert normalize_reference_heading(raw) == expected
        assert is_reference_heading(raw)


def test_conservative_filter_drops_late_references_from_heading_onward() -> None:
    docs = [
        *body_docs(7),
        doc("References\n[1] Example citation"),
        doc("[2] Another citation"),
    ]

    filtered, metadata = filter_reference_section(docs)

    assert len(filtered) == 7
    assert metadata["reference_section_detected"] is True
    assert metadata["trigger_heading"] == "References"
    assert metadata["trigger_doc_index"] == 7
    assert metadata["num_docs_before_filter"] == 9
    assert metadata["num_docs_after_filter"] == 7
    assert metadata["num_reference_docs_removed"] == 2


def test_conservative_filter_does_not_drop_early_references_heading() -> None:
    docs = [
        doc("References\nThis section explains how we reference prior work."),
        *body_docs(7),
    ]

    filtered, metadata = filter_reference_section(docs)

    assert filtered == docs
    assert metadata["reference_section_detected"] is False
    assert metadata["num_reference_docs_removed"] == 0


def test_conservative_filter_does_not_drop_inline_citations_only() -> None:
    docs = body_docs(8)

    filtered, metadata = filter_reference_section(docs)

    assert filtered == docs
    assert metadata["reference_section_detected"] is False


def test_conservative_filter_does_not_trigger_on_body_phrase() -> None:
    docs = [
        *body_docs(7),
        doc("We reference prior work as evidence for the evaluation design."),
        doc("Final body paragraph."),
    ]

    filtered, metadata = filter_reference_section(docs)

    assert filtered == docs
    assert metadata["reference_section_detected"] is False


def test_conservative_filter_uses_docling_headings_before_text_fallback() -> None:
    docs = [
        *body_docs(6),
        doc("[1] Example citation text", headings=["7. Bibliography"]),
        doc("[2] Another citation"),
    ]

    filtered, metadata = filter_reference_section(docs)

    assert len(filtered) == 6
    assert metadata["trigger_heading"] == "7. Bibliography"


def test_filter_can_be_disabled() -> None:
    docs = [*body_docs(7), doc("References\n[1] Citation")]

    filtered, metadata = filter_reference_section(docs, enabled=False)

    assert filtered == docs
    assert metadata["reference_filter_enabled"] is False
    assert metadata["reference_section_detected"] is False


def test_chunked_payload_includes_reference_filter_metadata() -> None:
    docs = body_docs(2)
    payload = build_chunked_documents_payload(
        docs=docs,
        extra_metadata={
            "reference_filter_enabled": True,
            "reference_section_detected": False,
            "num_docs_before_filter": 2,
            "num_docs_after_filter": 2,
            "num_reference_docs_removed": 0,
        },
    )

    assert payload["num_docs"] == 2
    assert payload["reference_filter_enabled"] is True
    assert payload["num_reference_docs_removed"] == 0
