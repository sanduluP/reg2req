from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence

from .provenance import ProvenanceEdge, fetch_provenance_edges


def document_coverage(edges: Sequence[ProvenanceEdge]) -> list[dict[str, Any]]:
    """
    Per-document contribution summary: how many assertions, distinct concepts,
    and normative (modality-carrying) statements each document put in the graph.
    """
    assertions: dict[str, int] = defaultdict(int)
    concepts: dict[str, set[str]] = defaultdict(set)
    normative: dict[str, int] = defaultdict(int)

    for edge in edges:
        for record in edge.records:
            doc = str(record.get("doc") or "").strip()
            if not doc:
                continue
            assertions[doc] += 1
            concepts[doc].add(edge.source)
            concepts[doc].add(edge.target)
            if str(record.get("modality") or "").strip():
                normative[doc] += 1

    return [
        {
            "doc": doc,
            "assertions": assertions[doc],
            "concepts": len(concepts[doc]),
            "normative_statements": normative[doc],
        }
        for doc in sorted(assertions, key=lambda d: -assertions[d])
    ]


def _modality_by_doc(edge: ProvenanceEdge) -> dict[str, str]:
    """Map each document to the modality it asserted for this edge (if any)."""
    out: dict[str, str] = {}
    for record in edge.records:
        doc = str(record.get("doc") or "").strip()
        modality = str(record.get("modality") or "").strip().upper()
        if doc and modality and doc not in out:
            out[doc] = modality
    return out


def overlap_relations(edges: Sequence[ProvenanceEdge]) -> list[dict[str, Any]]:
    """
    Assertions supported by two or more documents — the direct
    inter-document agreement table.

    Each row carries a `verdict` derived from the deontic modality the
    documents attached to the same (subject, predicate, object):
      - AGREEMENT: the documents agree (same modality, or none specified)
      - TENSION:   the same assertion carries different obligation strengths
                   across documents (e.g. MANDATORY in one, OPTIONAL in another)
    This is the core "same dimension + same relation + different strength"
    signal — a contradiction surfacing as a graph pattern.
    """
    out: list[dict[str, Any]] = []
    for edge in edges:
        if len(edge.docs) < 2:
            continue
        modality_by_doc = _modality_by_doc(edge)
        distinct_modalities = {m for m in modality_by_doc.values() if m}
        verdict = "TENSION" if len(distinct_modalities) >= 2 else "AGREEMENT"
        out.append(
            {
                "source": edge.source,
                "predicate": edge.predicate,
                "target": edge.target,
                "docs": list(edge.docs),
                "records": [dict(r) for r in edge.records],
                "modality_by_doc": modality_by_doc,
                "modalities": sorted(distinct_modalities),
                "verdict": verdict,
            }
        )
    # Tensions first, then by breadth of support.
    out.sort(key=lambda item: (item["verdict"] != "TENSION", -len(item["docs"]), item["source"]))
    return out


def concept_coverage(
    edges: Sequence[ProvenanceEdge],
    *,
    canon: Optional[Mapping[str, str]] = None,
    max_concepts: int = 500,
) -> dict[str, Any]:
    """
    Document × concept matrix: how many assertions of each document touch each
    (canonicalized) concept. Concepts mentioned by multiple documents sort first —
    those rows are the overlap; rows with a single document are coverage gaps.
    """
    canon = canon or {}

    def canonical(name: str) -> str:
        return canon.get(name, name)

    docs: set[str] = set()
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for edge in edges:
        for record in edge.records:
            doc = str(record.get("doc") or "").strip()
            if not doc:
                continue
            docs.add(doc)
            for concept in {canonical(edge.source), canonical(edge.target)}:
                counts[concept][doc] += 1

    ordered_docs = sorted(docs)
    rows = [
        {
            "concept": concept,
            "counts": dict(doc_counts),
            "docs": len(doc_counts),
            "total": sum(doc_counts.values()),
        }
        for concept, doc_counts in counts.items()
    ]
    rows.sort(key=lambda r: (-r["docs"], -r["total"], r["concept"]))

    return {"documents": ordered_docs, "rows": rows[:max_concepts]}


def build_overlap_report(
    graph: Optional[Any] = None,
    *,
    sources: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Full overlap/coverage report for the Compare tab.

    Concept canonicalization uses reviewer-accepted SAME_AS clusters so that
    e.g. "explainability" (ISO) and "explicability" (Fraunhofer) count as one
    concept once aligned.
    """
    from .alignment import same_as_clusters
    from .dimensions import dimension_canon_for_names

    edges = fetch_provenance_edges(graph, sources=sources)

    # Curated dimension aliases fill vocabulary gaps; reviewer-accepted SAME_AS
    # clusters always win over them.
    names = {e.source for e in edges} | {e.target for e in edges}
    canon = {**dimension_canon_for_names(names), **same_as_clusters(graph)}

    return {
        "coverage": document_coverage(edges),
        "overlap": overlap_relations(edges),
        "concepts": concept_coverage(edges, canon=canon),
        "num_edges_with_provenance": len(edges),
    }
