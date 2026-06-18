from __future__ import annotations

from kbdebugger.comparison.overlap import (
    concept_coverage,
    document_coverage,
    overlap_relations,
)
from kbdebugger.comparison.provenance import ProvenanceEdge, _parse_records


def _edges() -> list[ProvenanceEdge]:
    return [
        ProvenanceEdge(
            source="explainability",
            predicate="contributes_to",
            target="transparency",
            docs=("iso24028.pdf", "fraunhofer.pdf"),
            records=(
                {"doc": "iso24028.pdf", "quality": "Explainability contributes to transparency.", "modality": "RECOMMENDED"},
                {"doc": "fraunhofer.pdf", "quality": "Explainability supports transparency."},
            ),
        ),
        ProvenanceEdge(
            source="provider",
            predicate="requires",
            target="risk assessment",
            docs=("iso24028.pdf",),
            records=(
                {"doc": "iso24028.pdf", "quality": "The provider shall perform a risk assessment.", "modality": "MANDATORY"},
            ),
        ),
    ]


def test_parse_records_handles_json_strings_and_dicts() -> None:
    records = _parse_records(
        ['{"doc": "a.pdf", "quality": "q"}', {"doc": "b.pdf"}, "not json", 42]
    )
    assert records == ({"doc": "a.pdf", "quality": "q"}, {"doc": "b.pdf"})


def test_document_coverage_counts_assertions_concepts_and_normative() -> None:
    coverage = document_coverage(_edges())

    by_doc = {c["doc"]: c for c in coverage}
    assert by_doc["iso24028.pdf"]["assertions"] == 2
    assert by_doc["iso24028.pdf"]["normative_statements"] == 2
    assert by_doc["iso24028.pdf"]["concepts"] == 4
    assert by_doc["fraunhofer.pdf"]["assertions"] == 1
    assert by_doc["fraunhofer.pdf"]["normative_statements"] == 0


def test_overlap_relations_returns_only_multi_doc_edges() -> None:
    overlap = overlap_relations(_edges())

    assert len(overlap) == 1
    assert overlap[0]["source"] == "explainability"
    assert set(overlap[0]["docs"]) == {"iso24028.pdf", "fraunhofer.pdf"}
    # One doc said RECOMMENDED, the other unspecified -> agreement.
    assert overlap[0]["verdict"] == "AGREEMENT"


def test_overlap_relations_flags_modality_tension() -> None:
    edges = [
        ProvenanceEdge(
            source="system",
            predicate="provides",
            target="explanation",
            docs=("a.pdf", "b.pdf"),
            records=(
                {"doc": "a.pdf", "quality": "The system shall provide an explanation.", "modality": "MANDATORY"},
                {"doc": "b.pdf", "quality": "The system may provide an explanation.", "modality": "OPTIONAL"},
            ),
        ),
    ]
    overlap = overlap_relations(edges)

    assert len(overlap) == 1
    assert overlap[0]["verdict"] == "TENSION"
    assert overlap[0]["modalities"] == ["MANDATORY", "OPTIONAL"]
    assert overlap[0]["modality_by_doc"] == {"a.pdf": "MANDATORY", "b.pdf": "OPTIONAL"}


def test_concept_coverage_applies_same_as_canonicalization() -> None:
    edges = _edges() + [
        ProvenanceEdge(
            source="explicability",
            predicate="ensures",
            target="trust",
            docs=("fraunhofer.pdf",),
            records=({"doc": "fraunhofer.pdf", "quality": "Explicability ensures trust."},),
        ),
    ]

    matrix = concept_coverage(edges, canon={"explicability": "explainability"})

    assert matrix["documents"] == ["fraunhofer.pdf", "iso24028.pdf"]
    rows = {r["concept"]: r for r in matrix["rows"]}
    # "explicability" merged into "explainability": counted for both docs.
    assert "explicability" not in rows
    assert rows["explainability"]["docs"] == 2
    assert rows["explainability"]["counts"]["fraunhofer.pdf"] == 2
    # Shared concepts sort before single-doc concepts.
    assert matrix["rows"][0]["docs"] == 2
