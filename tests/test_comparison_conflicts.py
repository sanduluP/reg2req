from __future__ import annotations

import json
from typing import Any

from kbdebugger.comparison.conflicts import (
    find_conflict_candidates,
    judge_conflict_candidates,
    record_conflict_decision,
)
from kbdebugger.comparison.provenance import ProvenanceEdge


def _edges() -> list[ProvenanceEdge]:
    return [
        # MODALITY_CONFLICT: mandatory in ISO, recommended in Fraunhofer
        ProvenanceEdge(
            source="provider",
            predicate="requires",
            target="risk assessment",
            docs=("iso.pdf", "fraunhofer.pdf"),
            records=(
                {"doc": "iso.pdf", "quality": "The provider shall perform a risk assessment.", "modality": "MANDATORY"},
                {"doc": "fraunhofer.pdf", "quality": "The provider should screen for risks.", "modality": "RECOMMENDED"},
            ),
        ),
        # DEFINITION_DIVERGENCE: same term, different definitions per doc
        ProvenanceEdge(
            source="transparency",
            predicate="defines",
            target="openness about system behaviour",
            docs=("iso.pdf",),
            records=({"doc": "iso.pdf", "quality": "Transparency: openness about system behaviour."},),
        ),
        ProvenanceEdge(
            source="transparency",
            predicate="defines",
            target="traceability of decisions",
            docs=("fraunhofer.pdf",),
            records=({"doc": "fraunhofer.pdf", "quality": "Transparency means traceability of decisions."},),
        ),
        # TAXONOMY_CONFLICT: reversed hierarchy across docs
        ProvenanceEdge(
            source="explainability",
            predicate="is_subclass_of",
            target="transparency",
            docs=("iso.pdf",),
            records=({"doc": "iso.pdf", "quality": "Explainability is a subclass of transparency."},),
        ),
        ProvenanceEdge(
            source="transparency",
            predicate="is_subclass_of",
            target="explainability",
            docs=("fraunhofer.pdf",),
            records=({"doc": "fraunhofer.pdf", "quality": "Transparency is part of explainability."},),
        ),
        # VALUE_CONFLICT: different threshold per doc
        ProvenanceEdge(
            source="accuracy",
            predicate="has_threshold",
            target="0.9",
            docs=("iso.pdf",),
            records=({"doc": "iso.pdf", "quality": "Accuracy threshold shall be 0.9."},),
        ),
        ProvenanceEdge(
            source="accuracy",
            predicate="has_threshold",
            target="0.8",
            docs=("fraunhofer.pdf",),
            records=({"doc": "fraunhofer.pdf", "quality": "Accuracy threshold of 0.8 is sufficient."},),
        ),
    ]


def test_find_conflict_candidates_detects_all_four_types() -> None:
    candidates = find_conflict_candidates(edges=_edges(), canon={})

    by_type = {c["type"] for c in candidates}
    assert by_type == {
        "MODALITY_CONFLICT",
        "DEFINITION_DIVERGENCE",
        "TAXONOMY_CONFLICT",
        "VALUE_CONFLICT",
    }

    modality = next(c for c in candidates if c["type"] == "MODALITY_CONFLICT")
    sides = {modality["side_a"]["doc"], modality["side_b"]["doc"]}
    assert sides == {"iso.pdf", "fraunhofer.pdf"}
    assert {modality["side_a"]["modality"], modality["side_b"]["modality"]} == {
        "MANDATORY",
        "RECOMMENDED",
    }

    # Every candidate carries verbatim evidence and an UNJUDGED default verdict.
    for c in candidates:
        assert c["side_a"]["text"] and c["side_b"]["text"]
        assert c["verdict"] == "UNJUDGED"


def test_find_conflict_candidates_respects_exclusions_and_canon() -> None:
    first = find_conflict_candidates(edges=_edges(), canon={})
    excluded = {c["id"] for c in first if c["type"] == "MODALITY_CONFLICT"}

    second = find_conflict_candidates(edges=_edges(), canon={}, exclude_ids=excluded)
    assert {c["type"] for c in second} == {
        "DEFINITION_DIVERGENCE",
        "TAXONOMY_CONFLICT",
        "VALUE_CONFLICT",
    }


def test_judge_conflict_candidates_applies_llm_verdicts(monkeypatch) -> None:
    import kbdebugger.llm.model_access as model_access

    candidates = find_conflict_candidates(edges=_edges(), canon={})

    def fake_respond(prompt, **kwargs):
        payload_start = prompt.rindex("[")
        items = json.loads(prompt[payload_start:])
        return json.dumps(
            {
                "verdicts": [
                    {"id": item["id"], "verdict": "TENSION", "rationale": "Differs in strength."}
                    for item in items
                ]
            }
        )

    monkeypatch.setattr(model_access, "respond", fake_respond)

    judged = judge_conflict_candidates(candidates, batch_size=2)

    assert len(judged) == len(candidates)
    assert all(c["verdict"] == "TENSION" for c in judged)
    assert all(c["rationale"] == "Differs in strength." for c in judged)


def test_judge_conflict_candidates_degrades_to_unjudged_on_failure(monkeypatch) -> None:
    import kbdebugger.llm.model_access as model_access

    candidates = find_conflict_candidates(edges=_edges(), canon={})

    def boom(prompt, **kwargs):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(model_access, "respond", boom)

    judged = judge_conflict_candidates(candidates, batch_size=10)

    assert all(c["verdict"] == "UNJUDGED" for c in judged)
    assert all("LLM adjudication failed" in c["rationale"] for c in judged)


class FakeGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def query(self, cypher: str, params: dict[str, Any] | None = None):
        self.calls.append((cypher, params or {}))
        return []


def test_record_conflict_decision_writes_conflict_node() -> None:
    graph = FakeGraph()
    candidate = find_conflict_candidates(edges=_edges(), canon={})[0]
    candidate["verdict"] = "TENSION"
    candidate["rationale"] = "Strength differs."

    record_conflict_decision(graph, candidate=candidate, accepted=True)

    merge_call = graph.calls[0]
    assert "MERGE (c:Conflict {conflict_id: $id})" in merge_call[0]
    assert merge_call[1]["id"] == candidate["id"]
    assert merge_call[1]["props"]["status"] == "accepted"
    assert merge_call[1]["props"]["verdict"] == "TENSION"
    assert merge_call[1]["props"]["doc_a"] and merge_call[1]["props"]["doc_b"]

    # Accepted conflicts are linked to their concept node.
    assert any("INVOLVES" in c[0] for c in graph.calls)


def test_record_conflict_decision_dismissed_skips_involves_link() -> None:
    graph = FakeGraph()
    candidate = find_conflict_candidates(edges=_edges(), canon={})[0]

    record_conflict_decision(graph, candidate=candidate, accepted=False)

    assert graph.calls[0][1]["props"]["status"] == "dismissed"
    assert not any("INVOLVES" in c[0] for c in graph.calls)
