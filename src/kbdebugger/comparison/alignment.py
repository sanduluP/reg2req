from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np


def _default_encoder():
    """Build the project's sentence encoder lazily (heavy import)."""
    from kbdebugger.subgraph_similarity.encoder import SentenceTransformerEncoder

    model_name = os.getenv(
        "KB_ENCODER_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    ).strip()
    device = os.getenv("KB_ENCODER_DEVICE", "").strip() or None
    return SentenceTransformerEncoder(model_name=model_name, device=device, normalize=True)


def _get_graph(graph: Optional[Any]):
    if graph is not None:
        return graph
    from kbdebugger.graph import get_graph

    return get_graph()


def fetch_node_names(graph: Optional[Any] = None) -> list[str]:
    graph = _get_graph(graph)
    rows = graph.query(
        "MATCH (n:Node) WHERE n.name IS NOT NULL RETURN DISTINCT n.name AS name"
    )
    return sorted({str(r.get("name") or "").strip() for r in rows} - {""})


def fetch_decided_pairs(graph: Optional[Any] = None) -> set[frozenset[str]]:
    """Pairs the reviewer already accepted (same_as) or rejected (not_same_as)."""
    graph = _get_graph(graph)
    rows = graph.query(
        """
        MATCH (a:Node)-[r:same_as|not_same_as]-(b:Node)
        RETURN DISTINCT a.name AS a, b.name AS b
        """
    )
    decided: set[frozenset[str]] = set()
    for row in rows:
        a = str(row.get("a") or "").strip()
        b = str(row.get("b") or "").strip()
        if a and b and a != b:
            decided.add(frozenset((a, b)))
    return decided


def propose_alignment_candidates(
    graph: Optional[Any] = None,
    *,
    encoder: Optional[Any] = None,
    threshold: float = 0.78,
    max_candidates: int = 200,
) -> list[dict[str, Any]]:
    """
    Propose SAME_AS candidate pairs by embedding all node names and keeping
    pairs above the cosine threshold. Already-decided pairs never resurface.

    Returns [{term_a, term_b, score}] sorted by score (desc).
    """
    graph = _get_graph(graph)
    names = fetch_node_names(graph)
    if len(names) < 2:
        return []

    decided = fetch_decided_pairs(graph)
    encoder = encoder or _default_encoder()

    vectors = np.asarray(encoder.encode(names), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    sims = vectors @ vectors.T

    candidates: list[dict[str, Any]] = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sims[i, j])
            if score < threshold:
                continue
            if frozenset((names[i], names[j])) in decided:
                continue
            candidates.append(
                {"term_a": names[i], "term_b": names[j], "score": round(score, 4)}
            )

    candidates.sort(key=lambda c: -c["score"])
    return candidates[:max_candidates]


def record_alignment_decision(
    graph: Optional[Any] = None,
    *,
    term_a: str,
    term_b: str,
    accept: bool,
    score: Optional[float] = None,
) -> None:
    """
    Persist a reviewer decision as a `same_as` (accepted) or `not_same_as`
    (rejected) edge. A new decision replaces any previous one for the pair.
    Rejected pairs keep their score — high-score rejections are themselves an
    ambiguity finding (near-synonyms the reviewer considers distinct).
    """
    term_a = str(term_a).strip()
    term_b = str(term_b).strip()
    if not term_a or not term_b or term_a == term_b:
        raise ValueError("record_alignment_decision needs two distinct non-empty terms.")

    graph = _get_graph(graph)
    now_iso = datetime.now(timezone.utc).isoformat()
    rel_type = "same_as" if accept else "not_same_as"

    graph.query(
        """
        MATCH (a:Node {name: $a})-[r:same_as|not_same_as]-(b:Node {name: $b})
        DELETE r
        """,
        params={"a": term_a, "b": term_b},
    )

    graph.query(
        f"""
        MERGE (a:Node {{name: $a}})
        MERGE (b:Node {{name: $b}})
        MERGE (a)-[r:`{rel_type}`]->(b)
        SET r.reviewed = true,
            r.decided_at = $decided_at,
            r.score = $score
        """,
        params={"a": term_a, "b": term_b, "decided_at": now_iso, "score": score},
    )


def same_as_clusters(graph: Optional[Any] = None) -> dict[str, str]:
    """
    Union-find over accepted same_as edges → mapping from each term to its
    cluster's canonical label (shortest name, ties broken alphabetically).
    """
    graph = _get_graph(graph)
    try:
        rows = graph.query(
            "MATCH (a:Node)-[:same_as]-(b:Node) RETURN DISTINCT a.name AS a, b.name AS b"
        )
    except Exception:
        return {}

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent.setdefault(x, x)
        parent.setdefault(y, y)
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for row in rows:
        a = str(row.get("a") or "").strip()
        b = str(row.get("b") or "").strip()
        if a and b and a != b:
            union(a, b)

    clusters: dict[str, list[str]] = {}
    for term in list(parent):
        clusters.setdefault(find(term), []).append(term)

    canon: dict[str, str] = {}
    for members in clusters.values():
        canonical = sorted(members, key=lambda m: (len(m), m))[0]
        for member in members:
            canon[member] = canonical
    return canon


def rejected_near_synonyms(
    graph: Optional[Any] = None, *, min_score: float = 0.8
) -> list[dict[str, Any]]:
    """
    Pairs the reviewer rejected despite high embedding similarity — terms the
    corpus treats as near-identical but a human says are distinct concepts.
    """
    graph = _get_graph(graph)
    try:
        rows = graph.query(
            """
            MATCH (a:Node)-[r:not_same_as]->(b:Node)
            WHERE r.score IS NOT NULL AND r.score >= $min_score
            RETURN a.name AS a, b.name AS b, r.score AS score
            ORDER BY r.score DESC
            """,
            params={"min_score": min_score},
        )
    except Exception:
        return []

    return [
        {
            "term_a": str(r.get("a") or ""),
            "term_b": str(r.get("b") or ""),
            "score": float(r.get("score") or 0.0),
        }
        for r in rows
        if r.get("a") and r.get("b")
    ]
