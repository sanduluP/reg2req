from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from .provenance import ProvenanceEdge, fetch_provenance_edges, record_text

_DEFINITION_PREDICATES = {"defines", "with_description", "with_long_description"}
_TAXONOMY_PREDICATES = {"is_subclass_of", "is_dimension_of"}
_VALUE_PREDICATES = {"with_low", "with_high", "has_threshold", "with_default"}

VERDICTS = ("AGREE", "UNRELATED", "TENSION", "CONTRADICT", "UNJUDGED")


def _get_graph(graph: Optional[Any]):
    if graph is not None:
        return graph
    from kbdebugger.graph import get_graph

    return get_graph()


def _candidate_id(conflict_type: str, side_a: Mapping[str, Any], side_b: Mapping[str, Any]) -> str:
    payload = json.dumps(
        [conflict_type, sorted([str(side_a), str(side_b)])],
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"{conflict_type.lower()}-{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _make_candidate(
    conflict_type: str,
    *,
    summary: str,
    concept: str,
    side_a: dict[str, Any],
    side_b: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": _candidate_id(conflict_type, side_a, side_b),
        "type": conflict_type,
        "summary": summary,
        "concept": concept,
        "side_a": side_a,
        "side_b": side_b,
        "verdict": "UNJUDGED",
        "rationale": "",
    }


def _record_side(doc: str, record: Mapping[str, Any], triple: str, modality: str = "") -> dict[str, Any]:
    return {
        "doc": doc,
        "text": record_text(dict(record)) or triple,
        "chunk_excerpt": str(record.get("chunk_excerpt") or "").strip(),
        "triple": triple,
        "modality": modality or str(record.get("modality") or "").strip(),
    }


def find_conflict_candidates(
    graph: Optional[Any] = None,
    *,
    edges: Optional[Sequence[ProvenanceEdge]] = None,
    canon: Optional[Mapping[str, str]] = None,
    exclude_ids: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """
    Generate typed cross-document conflict candidates from the provenance layer.

    Types:
    - MODALITY_CONFLICT:   aligned triple, different obligation strength per doc
    - DEFINITION_DIVERGENCE: same concept defined differently by different docs
    - TAXONOMY_CONFLICT:   hierarchy edge asserted in both directions
    - VALUE_CONFLICT:      same subject+value-predicate, different values per doc
    """
    if edges is None:
        edges = fetch_provenance_edges(graph)
    if canon is None:
        from .alignment import same_as_clusters

        canon = same_as_clusters(graph)

    exclude_ids = exclude_ids or set()

    def canonical(name: str) -> str:
        return canon.get(name, name)

    candidates: list[dict[str, Any]] = []

    # --- MODALITY_CONFLICT -------------------------------------------------
    by_triple: dict[tuple[str, str, str], list[tuple[str, str, dict]]] = defaultdict(list)
    for edge in edges:
        key = (canonical(edge.source), edge.predicate, canonical(edge.target))
        triple = f"{edge.source} --{edge.predicate}--> {edge.target}"
        for record in edge.records:
            doc = str(record.get("doc") or "").strip()
            modality = str(record.get("modality") or "").strip().upper()
            if doc and modality:
                by_triple[key].append((doc, modality, {**record, "_triple": triple}))

    for key, entries in by_triple.items():
        per_modality: dict[str, tuple[str, dict]] = {}
        for doc, modality, record in entries:
            per_modality.setdefault(modality, (doc, record))
        if len(per_modality) < 2:
            continue
        modalities = sorted(per_modality)
        (mod_a, (doc_a, rec_a)), (mod_b, (doc_b, rec_b)) = (
            (modalities[0], per_modality[modalities[0]]),
            (modalities[1], per_modality[modalities[1]]),
        )
        if doc_a == doc_b:
            continue
        candidate = _make_candidate(
            "MODALITY_CONFLICT",
            summary=f"'{key[0]} {key[1]} {key[2]}' is {mod_a} in {doc_a} but {mod_b} in {doc_b}.",
            concept=key[0],
            side_a=_record_side(doc_a, rec_a, rec_a.get("_triple", ""), mod_a),
            side_b=_record_side(doc_b, rec_b, rec_b.get("_triple", ""), mod_b),
        )
        if candidate["id"] not in exclude_ids:
            candidates.append(candidate)

    # --- DEFINITION_DIVERGENCE ----------------------------------------------
    definitions: dict[str, list[tuple[str, str, dict, str]]] = defaultdict(list)
    for edge in edges:
        if edge.predicate not in _DEFINITION_PREDICATES:
            continue
        concept = canonical(edge.source)
        triple = f"{edge.source} --{edge.predicate}--> {edge.target}"
        for record in edge.records:
            doc = str(record.get("doc") or "").strip()
            if doc:
                definitions[concept].append((doc, edge.target, record, triple))

    for concept, entries in definitions.items():
        per_doc: dict[str, tuple[str, dict, str]] = {}
        for doc, definition, record, triple in entries:
            per_doc.setdefault(doc, (definition, record, triple))
        if len(per_doc) < 2:
            continue
        docs = sorted(per_doc)
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                def_a, rec_a, triple_a = per_doc[docs[i]]
                def_b, rec_b, triple_b = per_doc[docs[j]]
                if def_a.strip().lower() == def_b.strip().lower():
                    continue
                candidate = _make_candidate(
                    "DEFINITION_DIVERGENCE",
                    summary=f"'{concept}' is defined differently by {docs[i]} and {docs[j]}.",
                    concept=concept,
                    side_a=_record_side(docs[i], rec_a, triple_a),
                    side_b=_record_side(docs[j], rec_b, triple_b),
                )
                if candidate["id"] not in exclude_ids:
                    candidates.append(candidate)

    # --- TAXONOMY_CONFLICT ---------------------------------------------------
    taxonomy: dict[tuple[str, str, str], ProvenanceEdge] = {}
    for edge in edges:
        if edge.predicate in _TAXONOMY_PREDICATES:
            taxonomy[(canonical(edge.source), edge.predicate, canonical(edge.target))] = edge

    seen_taxonomy: set[frozenset] = set()
    for (src, pred, tgt), edge in taxonomy.items():
        reverse = taxonomy.get((tgt, pred, src))
        if reverse is None or src == tgt:
            continue
        pair_key = frozenset(((src, pred, tgt), (tgt, pred, src)))
        if pair_key in seen_taxonomy:
            continue
        seen_taxonomy.add(pair_key)

        rec_a = dict(edge.records[0]) if edge.records else {}
        rec_b = dict(reverse.records[0]) if reverse.records else {}
        doc_a = str(rec_a.get("doc") or (edge.docs[0] if edge.docs else "")).strip()
        doc_b = str(rec_b.get("doc") or (reverse.docs[0] if reverse.docs else "")).strip()
        candidate = _make_candidate(
            "TAXONOMY_CONFLICT",
            summary=f"'{src}' and '{tgt}' are placed in opposite {pred} directions.",
            concept=src,
            side_a=_record_side(doc_a, rec_a, f"{edge.source} --{pred}--> {edge.target}"),
            side_b=_record_side(doc_b, rec_b, f"{reverse.source} --{pred}--> {reverse.target}"),
        )
        if candidate["id"] not in exclude_ids:
            candidates.append(candidate)

    # --- VALUE_CONFLICT --------------------------------------------------------
    values: dict[tuple[str, str], list[tuple[str, str, dict, str]]] = defaultdict(list)
    for edge in edges:
        if edge.predicate not in _VALUE_PREDICATES:
            continue
        key = (canonical(edge.source), edge.predicate)
        triple = f"{edge.source} --{edge.predicate}--> {edge.target}"
        for record in edge.records:
            doc = str(record.get("doc") or "").strip()
            if doc:
                values[key].append((doc, edge.target, record, triple))

    for (concept, pred), entries in values.items():
        per_doc: dict[str, tuple[str, dict, str]] = {}
        for doc, value, record, triple in entries:
            per_doc.setdefault(doc, (value, record, triple))
        if len(per_doc) < 2:
            continue
        docs = sorted(per_doc)
        distinct_values = {per_doc[d][0].strip().lower() for d in docs}
        if len(distinct_values) < 2:
            continue
        val_a, rec_a, triple_a = per_doc[docs[0]]
        val_b, rec_b, triple_b = per_doc[docs[1]]
        candidate = _make_candidate(
            "VALUE_CONFLICT",
            summary=f"'{concept}' {pred}: '{val_a}' ({docs[0]}) vs '{val_b}' ({docs[1]}).",
            concept=concept,
            side_a=_record_side(docs[0], rec_a, triple_a),
            side_b=_record_side(docs[1], rec_b, triple_b),
        )
        if candidate["id"] not in exclude_ids:
            candidates.append(candidate)

    return candidates


def judge_conflict_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    batch_size: int = 5,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    """
    LLM adjudication: each candidate pair (with verbatim source texts) gets a
    verdict AGREE / UNRELATED / TENSION / CONTRADICT plus a one-line rationale.
    Failures degrade to UNJUDGED — candidates are never dropped.
    """
    from kbdebugger.llm.model_access import respond
    from kbdebugger.prompts import render_prompt
    from kbdebugger.utils import batched
    from kbdebugger.utils.json import ensure_json_object

    judged = [dict(c) for c in candidates]
    if not judged:
        return []

    by_id = {c["id"]: c for c in judged}

    for batch in batched(judged, batch_size):
        payload = [
            {
                "id": c["id"],
                "type": c["type"],
                "side_a": {"doc": c["side_a"]["doc"], "text": c["side_a"]["text"], "modality": c["side_a"].get("modality", "")},
                "side_b": {"doc": c["side_b"]["doc"], "text": c["side_b"]["text"], "modality": c["side_b"].get("modality", "")},
            }
            for c in batch
        ]
        try:
            prompt = render_prompt(
                "conflict_judge",
                candidates_json=json.dumps(payload, ensure_ascii=False),
            )
            response = respond(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=True,
            )
            parsed = ensure_json_object(response)
            verdicts = parsed.get("verdicts")
            if not isinstance(verdicts, list):
                continue
            for item in verdicts:
                if not isinstance(item, dict):
                    continue
                cid = str(item.get("id") or "").strip()
                verdict = str(item.get("verdict") or "").strip().upper()
                rationale = str(item.get("rationale") or "").strip()
                if cid in by_id and verdict in VERDICTS:
                    by_id[cid]["verdict"] = verdict
                    by_id[cid]["rationale"] = rationale
        except Exception as exc:  # noqa: BLE001 — judge failures must not lose candidates
            for c in batch:
                if c["verdict"] == "UNJUDGED" and not c["rationale"]:
                    c["rationale"] = f"LLM adjudication failed: {exc}"

    return judged


def fetch_recorded_conflict_ids(graph: Optional[Any] = None) -> set[str]:
    graph = _get_graph(graph)
    try:
        rows = graph.query("MATCH (c:Conflict) RETURN c.conflict_id AS id")
    except Exception:
        return set()
    return {str(r.get("id") or "").strip() for r in rows if r.get("id")}


def record_conflict_decision(
    graph: Optional[Any] = None,
    *,
    candidate: Mapping[str, Any],
    accepted: bool,
) -> None:
    """
    Persist the reviewer's decision as a (:Conflict) node so findings are
    queryable graph content and never resurface as candidates. Dismissed
    candidates are stored with status "dismissed".
    """
    graph = _get_graph(graph)
    side_a = candidate.get("side_a") or {}
    side_b = candidate.get("side_b") or {}

    props = {
        "type": str(candidate.get("type") or ""),
        "status": "accepted" if accepted else "dismissed",
        "verdict": str(candidate.get("verdict") or "UNJUDGED"),
        "rationale": str(candidate.get("rationale") or ""),
        "summary": str(candidate.get("summary") or ""),
        "concept": str(candidate.get("concept") or ""),
        "doc_a": str(side_a.get("doc") or ""),
        "text_a": str(side_a.get("text") or ""),
        "triple_a": str(side_a.get("triple") or ""),
        "modality_a": str(side_a.get("modality") or ""),
        "doc_b": str(side_b.get("doc") or ""),
        "text_b": str(side_b.get("text") or ""),
        "triple_b": str(side_b.get("triple") or ""),
        "modality_b": str(side_b.get("modality") or ""),
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }

    graph.query(
        """
        MERGE (c:Conflict {conflict_id: $id})
        SET c += $props
        """,
        params={"id": str(candidate.get("id") or ""), "props": props},
    )

    concept = props["concept"]
    if accepted and concept:
        graph.query(
            """
            MATCH (c:Conflict {conflict_id: $id})
            MATCH (n:Node {name: $concept})
            MERGE (c)-[:INVOLVES]->(n)
            """,
            params={"id": str(candidate.get("id") or ""), "concept": concept},
        )


def fetch_recorded_conflicts(graph: Optional[Any] = None) -> list[dict[str, Any]]:
    """Previously reviewed conflicts (accepted + dismissed) for display."""
    graph = _get_graph(graph)
    try:
        rows = graph.query(
            """
            MATCH (c:Conflict)
            RETURN c.conflict_id AS id, c.type AS type, c.status AS status,
                   c.verdict AS verdict, c.rationale AS rationale, c.summary AS summary,
                   c.concept AS concept, c.doc_a AS doc_a, c.text_a AS text_a,
                   c.doc_b AS doc_b, c.text_b AS text_b, c.decided_at AS decided_at
            ORDER BY c.decided_at DESC
            """
        )
    except Exception:
        return []
    return [dict(r) for r in rows]
