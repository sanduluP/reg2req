from __future__ import annotations

from typing import Any

import numpy as np

from kbdebugger.comparison.alignment import (
    propose_alignment_candidates,
    record_alignment_decision,
    same_as_clusters,
)


class FakeGraph:
    """Routes alignment queries to canned rows and records writes."""

    def __init__(self, *, names=(), decided=(), same_as_rows=()):
        self.names = list(names)
        self.decided = list(decided)
        self.same_as_rows = list(same_as_rows)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def query(self, cypher: str, params: dict[str, Any] | None = None):
        self.calls.append((cypher, params or {}))
        if "RETURN DISTINCT n.name AS name" in cypher:
            return [{"name": n} for n in self.names]
        if "same_as|not_same_as" in cypher:
            return [{"a": a, "b": b} for a, b in self.decided]
        if "[:same_as]" in cypher:
            return [{"a": a, "b": b} for a, b in self.same_as_rows]
        return []


class FakeEncoder:
    """Maps known names to fixed unit vectors so cosine scores are exact."""

    dim = 2

    _VECTORS = {
        "transparency": [1.0, 0.0],
        "transparence": [1.0, 0.0],   # identical direction → cosine 1.0
        "robustness": [0.0, 1.0],     # orthogonal → cosine 0.0
    }

    def encode(self, texts):
        return np.asarray(
            [self._VECTORS.get(t, [0.5, 0.5]) for t in texts], dtype=np.float32
        )


def test_propose_alignment_candidates_returns_high_similarity_pairs() -> None:
    graph = FakeGraph(names=["transparency", "transparence", "robustness"])

    candidates = propose_alignment_candidates(
        graph, encoder=FakeEncoder(), threshold=0.9
    )

    assert len(candidates) == 1
    assert {candidates[0]["term_a"], candidates[0]["term_b"]} == {
        "transparency",
        "transparence",
    }
    assert candidates[0]["score"] >= 0.99


def test_propose_alignment_candidates_skips_decided_pairs() -> None:
    graph = FakeGraph(
        names=["transparency", "transparence", "robustness"],
        decided=[("transparency", "transparence")],
    )

    candidates = propose_alignment_candidates(
        graph, encoder=FakeEncoder(), threshold=0.9
    )

    assert candidates == []


def test_record_alignment_decision_replaces_previous_edge() -> None:
    graph = FakeGraph()

    record_alignment_decision(
        graph, term_a="transparency", term_b="transparence", accept=True, score=0.97
    )

    delete_call, create_call = graph.calls[-2], graph.calls[-1]
    assert "DELETE r" in delete_call[0]
    assert "`same_as`" in create_call[0]
    assert create_call[1]["score"] == 0.97

    record_alignment_decision(
        graph, term_a="transparency", term_b="transparence", accept=False, score=0.97
    )
    assert "`not_same_as`" in graph.calls[-1][0]


def test_same_as_clusters_builds_canonical_mapping() -> None:
    graph = FakeGraph(
        same_as_rows=[
            ("explainability", "explicability"),
            ("explicability", "xai"),
        ]
    )

    canon = same_as_clusters(graph)

    # Shortest member is the canonical label for the whole cluster.
    assert canon["explainability"] == "xai"
    assert canon["explicability"] == "xai"
    assert canon["xai"] == "xai"
