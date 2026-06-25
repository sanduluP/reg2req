"""
Pre-merge comparison: check a document's extracted triples against the current
knowledge graph BEFORE anything is written.

This is the "KB debugger" gate — it lets the reviewer see, per extracted
statement, whether it is NEW, already EXISTING, in TENSION (same fact, different
obligation strength), or in direct CONFLICT with the graph, and shows the actual
SENTENCES on both sides so the difference is visible (not just node names).

Read-only: it only issues MATCH/RETURN queries and computes the verdict in
Python. Nothing is written.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Optional, Sequence

from kbdebugger.extraction.predicate_options import DEFAULT_ALLOWED_PREDICATES, PREDICATE_MODALITY
from kbdebugger.graph.utils import normalize_text, predicate_to_relationship_type

# Relationship types (snake) that express an allow / deny stance — used to flag
# direct contradictions for the same (subject, object) pair.
_POSITIVE_RELTYPES = frozenset(
    predicate_to_relationship_type(p) for p in ("Requires", "Recommends", "Permits", "Ensures", "ShouldEnsure")
)
_NEGATIVE_RELTYPES = frozenset(predicate_to_relationship_type(p) for p in ("Prohibits",))
_RELTYPE_TO_PREDICATE: dict[str, str] = {
    predicate_to_relationship_type(p): p for p in DEFAULT_ALLOWED_PREDICATES
}

# Obligation-strength polarity for modality-tension / modality-conflict checks.
_PROHIBITIVE_MODALITY = "PROHIBITED"


def _pred_label(reltype: str) -> str:
    return _RELTYPE_TO_PREDICATE.get(str(reltype), str(reltype))


def _edge_sentence(props: dict[str, Any]) -> str:
    """Best human-readable sentence attached to a KB edge."""
    sentence = props.get("sentence")
    if isinstance(sentence, str) and sentence.strip():
        return sentence.strip()
    for raw in props.get("provenance_records") or []:
        try:
            rec = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            continue
        if isinstance(rec, dict):
            q = rec.get("quality") or rec.get("chunk_excerpt")
            if isinstance(q, str) and q.strip():
                return q.strip()
    return ""


def _edge_modalities(props: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for raw in props.get("provenance_records") or []:
        try:
            rec = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            continue
        if isinstance(rec, dict):
            m = str(rec.get("modality") or "").strip().upper()
            if m:
                out.add(m)
    return out


def _edge_docs(props: dict[str, Any]) -> list[str]:
    docs = props.get("provenance_docs")
    if isinstance(docs, list):
        return [str(d) for d in docs if str(d).strip()]
    src = props.get("provenance_source")
    return [str(src)] if isinstance(src, str) and src.strip() else []


def _input_modality(triple: dict[str, Any]) -> str:
    m = str(triple.get("modality") or "").strip().upper()
    if m:
        return m
    return PREDICATE_MODALITY.get(str(triple.get("predicate") or "").strip(), "")


def preview_triples_against_kb(
    graph: Any,
    triples: Sequence[dict[str, Any]],
    *,
    source: Optional[str] = None,
) -> dict[str, Any]:
    """
    Compare extracted triples against the current graph (read-only).

    Each input triple: {"subject", "predicate", "object", "sentence"?, "modality"?}

    Returns:
        {
          "summary": {"new":n, "existing":n, "tension":n, "conflict":n, "related":n, "total":n},
          "items": [ {
              "subject","predicate","object","sentence","modality",
              "category": "NEW"|"EXISTING"|"TENSION"|"CONFLICT"|"RELATED",
              "reason": "...",
              "kb_matches": [ {"predicate","sentence","docs","modality"} ]
          } ]
        }
    """
    # One read of every KB edge, indexed by (normalized subject, normalized object).
    rows = graph.query(
        """
        MATCH (s:Node)-[r]->(t:Node)
        RETURN s.name AS s, type(r) AS rel, t.name AS o, properties(r) AS props
        """
    ) or []

    pair_index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    node_names: set[str] = set()
    for row in rows:
        s, o = str(row.get("s") or ""), str(row.get("o") or "")
        node_names.add(normalize_text(s))
        node_names.add(normalize_text(o))
        props = row.get("props") or {}
        pair_index[(normalize_text(s), normalize_text(o))].append({
            "rel": str(row.get("rel") or ""),
            "predicate": _pred_label(str(row.get("rel") or "")),
            "sentence": _edge_sentence(props),
            "docs": _edge_docs(props),
            "modalities": _edge_modalities(props),
        })

    items: list[dict[str, Any]] = []
    summary = {"new": 0, "existing": 0, "tension": 0, "conflict": 0, "related": 0}

    for triple in triples:
        subj = str(triple.get("subject") or "").strip()
        obj = str(triple.get("object") or "").strip()
        pred = str(triple.get("predicate") or "").strip()
        sentence = str(triple.get("sentence") or "").strip()
        in_mod = _input_modality(triple)
        if not subj or not obj or not pred:
            continue

        try:
            in_rel = predicate_to_relationship_type(pred)
        except ValueError:
            in_rel = pred.lower()

        key = (normalize_text(subj), normalize_text(obj))
        matches = pair_index.get(key, [])

        same_pred = [m for m in matches if m["rel"] == in_rel]
        other_pred = [m for m in matches if m["rel"] != in_rel]

        category = "NEW"
        reason = ""
        kb_matches: list[dict[str, Any]] = []

        if same_pred:
            kb_mods = set().union(*(m["modalities"] for m in same_pred)) if same_pred else set()
            kb_matches = same_pred
            polarity_clash = (
                (in_mod == _PROHIBITIVE_MODALITY and kb_mods and _PROHIBITIVE_MODALITY not in kb_mods)
                or (in_mod and in_mod != _PROHIBITIVE_MODALITY and _PROHIBITIVE_MODALITY in kb_mods)
            )
            if polarity_clash:
                category = "CONFLICT"
                reason = f"Same statement, opposite obligation: this doc is {in_mod or '—'}, the KB is {'/'.join(sorted(kb_mods)) or '—'}."
            elif in_mod and kb_mods and in_mod not in kb_mods:
                category = "TENSION"
                reason = f"Same statement, different obligation strength: this doc is {in_mod}, the KB is {'/'.join(sorted(kb_mods))}."
            else:
                category = "EXISTING"
                reason = "This statement is already in the knowledge graph."
        elif other_pred:
            in_is_pos = in_rel in _POSITIVE_RELTYPES
            in_is_neg = in_rel in _NEGATIVE_RELTYPES
            contradicting = [
                m for m in other_pred
                if (in_is_pos and m["rel"] in _NEGATIVE_RELTYPES)
                or (in_is_neg and m["rel"] in _POSITIVE_RELTYPES)
            ]
            if contradicting:
                category = "CONFLICT"
                kb_matches = contradicting
                reason = (
                    f"Direct contradiction: this doc says “{pred}” while the KB says "
                    f"“{contradicting[0]['predicate']}” for the same pair."
                )
            else:
                category = "RELATED"
                kb_matches = other_pred
                reason = "The same concepts are already related in the KB, but via a different relation."
        else:
            category = "NEW"
            subj_known = normalize_text(subj) in node_names
            obj_known = normalize_text(obj) in node_names
            if subj_known or obj_known:
                reason = "New relation between concepts that already exist in the KB."
            else:
                reason = "Brand-new knowledge (concepts not yet in the KB)."

        # Tally
        summary[category.lower()] = summary.get(category.lower(), 0) + 1

        items.append({
            "subject": subj,
            "predicate": pred,
            "object": obj,
            "sentence": sentence,
            "modality": in_mod,
            "category": category,
            "reason": reason,
            "kb_matches": [
                {"predicate": m["predicate"], "sentence": m["sentence"],
                 "docs": m["docs"], "modality": "/".join(sorted(m["modalities"]))}
                for m in kb_matches
            ],
        })

    # Order: conflicts first, then tension, related, new, existing.
    order = {"CONFLICT": 0, "TENSION": 1, "RELATED": 2, "NEW": 3, "EXISTING": 4}
    items.sort(key=lambda it: order.get(it["category"], 9))

    summary["total"] = len(items)
    return {"summary": summary, "items": items, "source": source}
