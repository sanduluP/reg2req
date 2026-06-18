from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class ProvenanceEdge:
    """One relationship plus its parsed per-document provenance records."""

    source: str
    predicate: str
    target: str
    docs: tuple[str, ...] = ()
    records: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _parse_records(raw: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    parsed: list[dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, dict):
            parsed.append(entry)
            continue
        try:
            obj = json.loads(str(entry))
        except (TypeError, ValueError):
            continue
        if isinstance(obj, dict):
            parsed.append(obj)
    return tuple(parsed)


def fetch_provenance_edges(
    graph: Optional[Any] = None,
    *,
    sources: Optional[list[str]] = None,
) -> list[ProvenanceEdge]:
    """
    Fetch every relationship that carries provenance records and parse the
    JSON entries. This is the single read query all comparison reports
    build on.
    """
    if graph is None:
        from kbdebugger.graph import get_graph

        graph = get_graph()

    rows = graph.query(
        """
        MATCH (s)-[r]->(o)
        WHERE r.provenance_records IS NOT NULL AND size(r.provenance_records) > 0
        RETURN s.name AS source,
               type(r) AS predicate,
               o.name AS target,
               coalesce(r.provenance_docs, []) AS docs,
               r.provenance_records AS records
        """
    )

    edges: list[ProvenanceEdge] = []
    for row in rows:
        source = str(row.get("source") or "").strip()
        target = str(row.get("target") or "").strip()
        predicate = str(row.get("predicate") or "").strip()
        if not source or not target or not predicate:
            continue

        records = _parse_records(row.get("records"))
        docs = tuple(
            dict.fromkeys(
                [str(d).strip() for d in (row.get("docs") or []) if str(d).strip()]
                + [str(r.get("doc") or "").strip() for r in records if str(r.get("doc") or "").strip()]
            )
        )

        edges.append(
            ProvenanceEdge(
                source=source,
                predicate=predicate,
                target=target,
                docs=docs,
                records=records,
            )
        )

    if sources:
        sources_set = set(sources)
        edges = [e for e in edges if sources_set.intersection(e.docs)]

    return edges


def record_text(record: dict[str, Any]) -> str:
    """Best display text for a provenance record: quality, else chunk excerpt."""
    quality = str(record.get("quality") or "").strip()
    if quality:
        return quality
    return str(record.get("chunk_excerpt") or "").strip()
