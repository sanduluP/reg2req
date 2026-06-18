from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Optional, Sequence

from .provenance import ProvenanceEdge, fetch_provenance_edges, record_text

# Hedge/vagueness markers typical of standards prose. Each is matched as a
# whole phrase, case-insensitive.
VAGUE_TERMS: tuple[str, ...] = (
    "appropriate",
    "adequate",
    "sufficient",
    "reasonable",
    "as far as possible",
    "where applicable",
    "where appropriate",
    "if necessary",
    "as needed",
    "state of the art",
    "best effort",
    "in a timely manner",
    "to the extent possible",
    "suitable",
    "acceptable",
    "relevant",
)

_NORMATIVE_REL_TYPES = {"requires", "recommends", "permits", "prohibits"}
_DEFINITION_REL_TYPES = {"defines", "with_description", "with_long_description"}

_VAGUE_PATTERNS = {
    term: re.compile(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", re.IGNORECASE)
    for term in VAGUE_TERMS
}


def undefined_normative_terms(
    edges: Sequence[ProvenanceEdge],
    *,
    max_per_doc: int = 50,
) -> list[dict[str, Any]]:
    """
    Terms that a document uses inside obligations (normative predicates or
    MANDATORY/PROHIBITED modality) but never defines in that same document.
    A pure provenance-layer query — the classic ambiguity signal in standards.
    """
    obligated: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)  # doc -> term -> info
    defined: dict[str, set[str]] = defaultdict(set)

    for edge in edges:
        is_normative_pred = edge.predicate in _NORMATIVE_REL_TYPES
        for record in edge.records:
            doc = str(record.get("doc") or "").strip()
            if not doc:
                continue

            if edge.predicate in _DEFINITION_REL_TYPES:
                defined[doc].add(edge.source.lower())
                continue

            modality = str(record.get("modality") or "").strip().upper()
            if is_normative_pred or modality in {"MANDATORY", "PROHIBITED"}:
                example = record_text(dict(record))
                for term in (edge.source, edge.target):
                    obligated[doc].setdefault(
                        term,
                        {"predicate": edge.predicate, "example": example},
                    )

    rows: list[dict[str, Any]] = []
    for doc in sorted(obligated):
        count = 0
        for term in sorted(obligated[doc]):
            if term.lower() in defined[doc]:
                continue
            info = obligated[doc][term]
            rows.append(
                {
                    "doc": doc,
                    "term": term,
                    "predicate": info["predicate"],
                    "example": info["example"],
                }
            )
            count += 1
            if count >= max_per_doc:
                break
    return rows


def vague_language_report(
    edges: Sequence[ProvenanceEdge],
    *,
    max_examples: int = 3,
) -> list[dict[str, Any]]:
    """
    Per-document hedge-term usage: counts and example statements for each
    vague term found in the stored qualities / chunk excerpts.
    """
    seen_texts: dict[str, set[str]] = defaultdict(set)
    hits: dict[tuple[str, str], dict[str, Any]] = {}

    for edge in edges:
        for record in edge.records:
            doc = str(record.get("doc") or "").strip()
            if not doc:
                continue
            text = record_text(dict(record))
            if not text or text in seen_texts[doc]:
                continue
            seen_texts[doc].add(text)

            for term, pattern in _VAGUE_PATTERNS.items():
                if not pattern.search(text):
                    continue
                entry = hits.setdefault(
                    (doc, term), {"doc": doc, "term": term, "count": 0, "examples": []}
                )
                entry["count"] += 1
                if len(entry["examples"]) < max_examples:
                    entry["examples"].append(text)

    rows = list(hits.values())
    rows.sort(key=lambda r: (r["doc"], -r["count"], r["term"]))
    return rows


def build_ambiguity_report(
    graph: Optional[Any] = None,
    *,
    sources: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Full ambiguity report for the Compare tab."""
    from .alignment import rejected_near_synonyms

    edges = fetch_provenance_edges(graph, sources=sources)

    return {
        "undefined_normative_terms": undefined_normative_terms(edges),
        "vague_language": vague_language_report(edges),
        "near_synonyms": rejected_near_synonyms(graph),
    }
