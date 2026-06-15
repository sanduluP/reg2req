from __future__ import annotations

from kbdebugger.graph import retriever as retriever_module
from kbdebugger.graph.retriever import KnowledgeGraphRetriever


def relation(sentence: str):
    return {
        "source": {"label": "Explainability", "id": "source-id"},
        "target": {"label": "Requirement", "id": "target-id"},
        "edge": {
            "label": "is_a",
            "properties": {
                "sentence": sentence,
                "source": "Explainability",
                "target": "Requirement",
            },
        },
    }


class FakeGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def query_relations(self, cypher: str, params: dict):
        self.calls.append((cypher, params))
        if 'properties(r)["target"]' in cypher:
            return [relation("Explainability is a Requirement")]
        return []


def test_retriever_rel_props_query_checks_source_target_without_unknown_property_warning(monkeypatch):
    graph = FakeGraph()
    monkeypatch.setattr(retriever_module, "get_graph", lambda: graph)

    hits = KnowledgeGraphRetriever(limit_per_pattern=3).retrieve("Explainability")

    assert len(hits) == 1
    assert hits[0]["match_pattern"] == "rel_props"
    assert hits[0]["relation"]["edge"]["properties"]["target"] == "Requirement"

    rel_props_query = graph.calls[2][0]
    assert 'properties(r)["sentence"]' in rel_props_query
    assert 'properties(r)["source"]' in rel_props_query
    assert 'properties(r)["target"]' in rel_props_query
    assert "r.source" not in rel_props_query
    assert "r.target" not in rel_props_query
    assert graph.calls[2][1] == {"keyword": "explainability", "limit": 3}
