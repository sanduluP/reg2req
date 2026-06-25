"""
Five-strategy graph verifier (Phase B).

Adapted from the author's prior work (NGAC policy extraction) to KBDebugger's
generic Subject-Predicate-Object knowledge graph:

    S1 Coverage      every input sentence is reflected in >=1 triple
    S2 Correctness   every node/edge is well-formed and uses a declared predicate
    S3 Consistency   no contradictory edges (e.g. Requires vs Prohibits) for a pair
    S4 Completeness  every triple is structurally sound (no missing part / provenance)
    S5 Minimality    no duplicate (subject, predicate, object) edges

Everything here is READ-ONLY: the verifier issues only MATCH/RETURN queries and
computes verdicts in Python. It never writes to the graph.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from kbdebugger.extraction.predicate_options import (
    DEFAULT_ALLOWED_PREDICATES,
    PREDICATE_MODALITY,
)
from kbdebugger.graph.utils import normalize_text, predicate_to_relationship_type

# Map snake_case relationship type -> canonical Predicate name (e.g. "requires" -> "Requires").
_RELTYPE_TO_PREDICATE: dict[str, str] = {
    predicate_to_relationship_type(p): p for p in DEFAULT_ALLOWED_PREDICATES
}
_DECLARED_RELTYPES: frozenset[str] = frozenset(_RELTYPE_TO_PREDICATE)

# Allow/deny pairs that must not coexist for the same (subject, object).
_POSITIVE_RELTYPES: frozenset[str] = frozenset(
    predicate_to_relationship_type(p) for p in ("Requires", "Recommends", "Permits")
)
_NEGATIVE_RELTYPES: frozenset[str] = frozenset(
    predicate_to_relationship_type(p) for p in ("Prohibits",)
)
_NORMATIVE_RELTYPES: frozenset[str] = frozenset(
    predicate_to_relationship_type(p) for p in PREDICATE_MODALITY
)

# Strings that indicate a missing / placeholder node name.
_PLACEHOLDER_NAMES: frozenset[str] = frozenset(
    {"", "n/a", "na", "none", "null", "nil", "unknown", "?", "-", "—", "…", "⌀", "tbd"}
)

STRATEGY_NAMES: dict[str, str] = {
    "S1": "Coverage",
    "S2": "Correctness",
    "S3": "Consistency",
    "S4": "Completeness",
    "S5": "Minimality",
}

# Per-strategy pass thresholds (score must be >= threshold to pass).
DEFAULT_THRESHOLDS: dict[str, float] = {
    "S1": 0.9,
    "S2": 1.0,
    "S3": 1.0,
    "S4": 1.0,
    "S5": 1.0,
}

_MAX_FLAGGED = 50  # cap how many flagged items we return per strategy (UI-friendly)


# ---------------------------------------------------------------------------
# Graph reads (read-only)
# ---------------------------------------------------------------------------
def _fetch_edges(graph: Any) -> list[dict[str, Any]]:
    rows = graph.query(
        """
        MATCH (s:Node)-[r]->(t:Node)
        RETURN s.name AS src, type(r) AS rel, t.name AS tgt, properties(r) AS props
        """
    )
    return rows or []


def _fetch_node_issues(graph: Any) -> list[str]:
    """Return names of nodes whose `name` is missing/blank (data corruption)."""
    rows = graph.query(
        """
        MATCH (n:Node)
        WHERE n.name IS NULL OR trim(n.name) = ''
        RETURN elementId(n) AS id
        """
    )
    return [str(row.get("id")) for row in (rows or [])]


def _fetch_duplicate_groups(graph: Any) -> list[dict[str, Any]]:
    rows = graph.query(
        """
        MATCH (s:Node)-[r]->(t:Node)
        WITH s.name AS src, type(r) AS rel, t.name AS tgt, count(*) AS c
        WHERE c > 1
        RETURN src, rel, tgt, c
        """
    )
    return rows or []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _edge_provenance_sentences(props: dict[str, Any]) -> list[str]:
    """All source sentences/qualities attached to an edge, normalized."""
    out: list[str] = []
    sentence = props.get("sentence")
    if isinstance(sentence, str) and sentence.strip():
        out.append(normalize_text(sentence))
    for raw in props.get("provenance_records") or []:
        try:
            record = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            continue
        if isinstance(record, dict):
            quality = record.get("quality")
            if isinstance(quality, str) and quality.strip():
                out.append(normalize_text(quality))
    return out


def _edge_has_provenance(props: dict[str, Any]) -> bool:
    if isinstance(props.get("sentence"), str) and props["sentence"].strip():
        return True
    if props.get("provenance_records"):
        return True
    if isinstance(props.get("provenance_source"), str) and props["provenance_source"].strip():
        return True
    if props.get("provenance_docs"):
        return True
    return False


def _is_placeholder(name: Any) -> bool:
    return str(name or "").strip().lower() in _PLACEHOLDER_NAMES


def _triple_label(src: Any, reltype: Any, tgt: Any) -> str:
    predicate = _RELTYPE_TO_PREDICATE.get(str(reltype), str(reltype))
    return f"({src or '⌀'}, {predicate}, {tgt or '⌀'})"


def _result(
    sid: str,
    *,
    passed: bool,
    score: Optional[float],
    threshold: float,
    summary: str,
    flagged: Optional[list[dict[str, Any]]] = None,
    skipped: bool = False,
    error: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "id": sid,
        "name": STRATEGY_NAMES[sid],
        "passed": passed,
        "skipped": skipped,
        "score": score,
        "threshold": threshold,
        "summary": summary,
        "flagged": (flagged or [])[:_MAX_FLAGGED],
        "flagged_total": len(flagged or []),
        "error": error,
    }


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
def _s1_coverage(
    edges: list[dict[str, Any]],
    source_sentences: Optional[Sequence[str]],
    threshold: float,
) -> dict[str, Any]:
    inputs = [s for s in (str(x or "").strip() for x in (source_sentences or [])) if s]
    if not inputs:
        return _result(
            "S1", passed=True, score=None, threshold=threshold, skipped=True,
            summary="No source sentences supplied — coverage not checked.",
        )

    covered_norms: set[str] = set()
    for edge in edges:
        props = edge.get("props") or {}
        covered_norms.update(_edge_provenance_sentences(props))

    missing: list[dict[str, Any]] = []
    seen: set[str] = set()
    covered_count = 0
    for sentence in inputs:
        norm = normalize_text(sentence)
        if norm in seen:
            continue
        seen.add(norm)
        if norm in covered_norms:
            covered_count += 1
        else:
            missing.append({"label": sentence, "reason": "no triple references this sentence"})

    total = len(seen)
    score = covered_count / total if total else 1.0
    return _result(
        "S1", passed=score >= threshold, score=score, threshold=threshold,
        summary=f"{covered_count}/{total} source sentences reflected in the graph",
        flagged=missing,
    )


def _s2_correctness(
    edges: list[dict[str, Any]],
    node_issue_ids: list[str],
    allowed_reltypes: frozenset[str],
    threshold: float,
) -> dict[str, Any]:
    flagged: list[dict[str, Any]] = []
    for node_id in node_issue_ids:
        flagged.append({"label": f"node {node_id}", "reason": "node has an empty/missing name"})

    bad_edges = 0
    for edge in edges:
        src, rel, tgt = edge.get("src"), edge.get("rel"), edge.get("tgt")
        reasons = []
        if _is_placeholder(src) or _is_placeholder(tgt):
            reasons.append("empty/placeholder endpoint")
        if str(rel) not in allowed_reltypes:
            reasons.append(f"predicate '{_RELTYPE_TO_PREDICATE.get(str(rel), rel)}' not in declared vocabulary")
        if reasons:
            bad_edges += 1
            flagged.append({"label": _triple_label(src, rel, tgt), "reason": "; ".join(reasons)})

    total_units = len(edges) + len(node_issue_ids)
    bad_units = bad_edges + len(node_issue_ids)
    score = 1.0 if total_units == 0 else (total_units - bad_units) / total_units
    return _result(
        "S2", passed=score >= threshold, score=score, threshold=threshold,
        summary=(
            f"all {len(edges)} triples well-typed"
            if bad_units == 0
            else f"{bad_units} issue(s) across nodes/edges"
        ),
        flagged=flagged,
    )


def _s3_consistency(edges: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    # group reltypes by (subject, object)
    pair_reltypes: dict[tuple[str, str], set[str]] = {}
    for edge in edges:
        key = (normalize_text(str(edge.get("src") or "")), normalize_text(str(edge.get("tgt") or "")))
        pair_reltypes.setdefault(key, set()).add(str(edge.get("rel")))

    flagged: list[dict[str, Any]] = []
    for (src, tgt), reltypes in pair_reltypes.items():
        if reltypes & _POSITIVE_RELTYPES and reltypes & _NEGATIVE_RELTYPES:
            preds = sorted(_RELTYPE_TO_PREDICATE.get(rt, rt) for rt in reltypes)
            flagged.append({
                "label": f"({src}) ⇄ ({tgt})",
                "reason": f"contradictory predicates: {', '.join(preds)}",
            })

    passed = len(flagged) == 0
    return _result(
        "S3", passed=passed, score=1.0 if passed else 0.0, threshold=threshold,
        summary="no contradictory edges" if passed else f"{len(flagged)} contradictory pair(s)",
        flagged=flagged,
    )


def _s4_completeness(edges: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    flagged: list[dict[str, Any]] = []
    incomplete = 0
    for edge in edges:
        src, rel, tgt = edge.get("src"), edge.get("rel"), edge.get("tgt")
        props = edge.get("props") or {}
        reasons = []
        if _is_placeholder(src):
            reasons.append("empty subject")
        if _is_placeholder(tgt):
            reasons.append("empty object")
        if not str(rel or "").strip():
            reasons.append("no predicate")
        if not _edge_has_provenance(props):
            reasons.append("no source sentence / provenance")
        if reasons:
            incomplete += 1
            flagged.append({"label": _triple_label(src, rel, tgt), "reason": "; ".join(reasons)})

    total = len(edges)
    score = 1.0 if total == 0 else (total - incomplete) / total
    return _result(
        "S4", passed=score >= threshold, score=score, threshold=threshold,
        summary=(
            f"all {total} triples complete"
            if incomplete == 0
            else f"{incomplete}/{total} triples missing a part"
        ),
        flagged=flagged,
    )


def _s5_minimality(duplicate_groups: list[dict[str, Any]], total_edges: int, threshold: float) -> dict[str, Any]:
    flagged: list[dict[str, Any]] = []
    redundant = 0
    for group in duplicate_groups:
        count = int(group.get("c") or 0)
        redundant += max(0, count - 1)
        flagged.append({
            "label": _triple_label(group.get("src"), group.get("rel"), group.get("tgt")),
            "reason": f"{count} duplicate edges",
        })

    score = 1.0 if total_edges == 0 else (total_edges - redundant) / total_edges
    passed = len(flagged) == 0
    return _result(
        "S5", passed=passed, score=score, threshold=threshold,
        summary="no duplicate triples" if passed else f"{redundant} redundant edge(s)",
        flagged=flagged,
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def verify_graph(
    graph: Any,
    *,
    source_sentences: Optional[Sequence[str]] = None,
    allowed_predicates: Optional[Sequence[str]] = None,
    thresholds: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """
    Run the five-strategy verification against the current graph.

    Read-only. Each strategy is isolated in its own try/except so a single
    failing check degrades to a per-strategy "verification error" instead of
    aborting the whole verdict.

    Returns a dict:
        {
          "verdict": "PASS" | "FAIL",
          "passed_count": int, "total": 5,
          "strategies": [ {id, name, passed, skipped, score, threshold,
                           summary, flagged:[{label, reason}], error}, ... ]
        }
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    allowed_reltypes = set(_DECLARED_RELTYPES)
    for predicate in allowed_predicates or ():
        try:
            allowed_reltypes.add(predicate_to_relationship_type(predicate))
        except ValueError:
            continue
    allowed_reltypes_fz = frozenset(allowed_reltypes)

    # Single read of the graph topology, reused across strategies.
    edges = _fetch_edges(graph)
    node_issue_ids = _fetch_node_issues(graph)
    duplicate_groups = _fetch_duplicate_groups(graph)

    runners = [
        ("S1", lambda: _s1_coverage(edges, source_sentences, thr["S1"])),
        ("S2", lambda: _s2_correctness(edges, node_issue_ids, allowed_reltypes_fz, thr["S2"])),
        ("S3", lambda: _s3_consistency(edges, thr["S3"])),
        ("S4", lambda: _s4_completeness(edges, thr["S4"])),
        ("S5", lambda: _s5_minimality(duplicate_groups, len(edges), thr["S5"])),
    ]

    strategies: list[dict[str, Any]] = []
    for sid, runner in runners:
        try:
            strategies.append(runner())
        except Exception as exc:  # noqa: BLE001 — never let one check break the verdict
            strategies.append(_result(
                sid, passed=False, score=None, threshold=thr[sid],
                summary="verification error", error=f"{type(exc).__name__}: {exc}",
            ))

    passed_count = sum(1 for s in strategies if s["passed"])
    verdict = "PASS" if all(s["passed"] for s in strategies) else "FAIL"
    return {
        "verdict": verdict,
        "passed_count": passed_count,
        "total": len(strategies),
        "edge_count": len(edges),
        "strategies": strategies,
    }
