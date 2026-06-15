from __future__ import annotations

from kbdebugger.comparison.ambiguity import (
    undefined_normative_terms,
    vague_language_report,
)
from kbdebugger.comparison.provenance import ProvenanceEdge


def _edges() -> list[ProvenanceEdge]:
    return [
        # ISO obligates "human oversight" but never defines it.
        ProvenanceEdge(
            source="provider",
            predicate="requires",
            target="human oversight",
            docs=("iso.pdf",),
            records=(
                {"doc": "iso.pdf", "quality": "The provider shall ensure appropriate human oversight.", "modality": "MANDATORY"},
            ),
        ),
        # ISO defines "transparency" (so transparency is NOT undefined).
        ProvenanceEdge(
            source="transparency",
            predicate="defines",
            target="openness about system behaviour",
            docs=("iso.pdf",),
            records=({"doc": "iso.pdf", "quality": "Transparency: openness about system behaviour."},),
        ),
        # ISO obligates transparency too — defined, so excluded from report.
        ProvenanceEdge(
            source="system",
            predicate="requires",
            target="transparency",
            docs=("iso.pdf",),
            records=(
                {"doc": "iso.pdf", "quality": "The system shall provide transparency where applicable.", "modality": "MANDATORY"},
            ),
        ),
        # Fraunhofer defines human oversight — its own doc is fine.
        ProvenanceEdge(
            source="human oversight",
            predicate="with_description",
            target="monitoring by a competent person",
            docs=("fraunhofer.pdf",),
            records=({"doc": "fraunhofer.pdf", "quality": "Human oversight means monitoring by a competent person."},),
        ),
    ]


def test_undefined_normative_terms_are_per_document() -> None:
    rows = undefined_normative_terms(_edges())

    iso_terms = {r["term"] for r in rows if r["doc"] == "iso.pdf"}
    # "human oversight" is obligated but not defined in iso.pdf,
    # even though fraunhofer.pdf defines it — definitions are per document.
    assert "human oversight" in iso_terms
    # "transparency" is defined in iso.pdf, so it must not be reported there.
    assert "transparency" not in iso_terms

    row = next(r for r in rows if r["doc"] == "iso.pdf" and r["term"] == "human oversight")
    assert row["predicate"] == "requires"
    assert "human oversight" in row["example"]


def test_vague_language_report_counts_hedge_terms_per_document() -> None:
    rows = vague_language_report(_edges())

    by_key = {(r["doc"], r["term"]): r for r in rows}
    assert ("iso.pdf", "appropriate") in by_key
    assert ("iso.pdf", "where applicable") in by_key

    appropriate = by_key[("iso.pdf", "appropriate")]
    assert appropriate["count"] == 1
    assert any("appropriate human oversight" in ex for ex in appropriate["examples"])

    # No vague terms attributed to fraunhofer.pdf statements.
    assert not any(r["doc"] == "fraunhofer.pdf" for r in rows)
