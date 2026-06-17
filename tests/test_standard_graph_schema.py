from __future__ import annotations

from typing import Any, Sequence

import pytest

from kbdebugger.graph.retriever import KnowledgeGraphRetriever
from kbdebugger.graph.store import GraphStore
from kbdebugger.graph.utils import (
    map_extracted_triplets_to_graph_relations,
    predicate_to_relationship_type,
    rows_to_graph_relations,
)
from kbdebugger.types import GraphRelation


def relation(source: str, predicate: str, target: str, sentence: str = "") -> GraphRelation:
    return {
        "source": {"label": source, "id": None, "created_at": None, "last_updated_at": None},
        "target": {"label": target, "id": None, "created_at": None, "last_updated_at": None},
        "edge": {"label": predicate, "properties": {"sentence": sentence, "source": "review.xlsx"}},
    }


def test_predicate_to_relationship_type_converts_allowed_predicates() -> None:
    assert predicate_to_relationship_type("HasParameter") == "has_parameter"
    assert predicate_to_relationship_type("IsA") == "is_a"
    assert predicate_to_relationship_type("SurfacesRisk") == "surfaces_risk"


def test_predicate_to_relationship_type_sanitizes_unknown_predicates() -> None:
    # Non-standard predicates are kept (reviewer-approved) and sanitized into
    # a safe snake_case relationship type instead of being rejected.
    assert predicate_to_relationship_type("invented relationship") == "invented_relationship"
    assert predicate_to_relationship_type("MustDocument") == "must_document"


def test_predicate_to_relationship_type_rejects_unsafe_predicates() -> None:
    with pytest.raises(ValueError, match="Unsafe Neo4j relationship type"):
        predicate_to_relationship_type("123")


def test_map_extracted_triplets_preserves_node_casing_and_pascal_predicate() -> None:
    rels = map_extracted_triplets_to_graph_relations(
        {
            "sentence": "AI Model applies to Human Oversight.",
            "triplets": [("  AI   Model  ", "Human Oversight", "AppliesTo")],
        },
        source="audit.pdf",
    )

    assert rels == [
        {
            "source": {"label": "AI Model"},
            "target": {"label": "Human Oversight"},
            "edge": {
                "label": "AppliesTo",
                "properties": {
                    "sentence": "AI Model applies to Human Oversight.",
                    "source": "audit.pdf",
                },
            },
        }
    ]


def test_rows_to_graph_relations_does_not_inject_legacy_label_property() -> None:
    rels = rows_to_graph_relations(
        [
            {
                "source": "AI Model",
                "target": "Human Oversight",
                "predicate": "applies_to",
                "props": {"sentence": "AI Model applies to Human Oversight."},
                "source_id": "n1",
                "target_id": "n2",
            }
        ]
    )

    assert rels[0]["edge"]["label"] == "applies_to"
    assert rels[0]["edge"]["properties"] == {"sentence": "AI Model applies to Human Oversight."}


def test_graph_store_upsert_uses_node_name_and_typed_relationship() -> None:
    captured: dict[str, Any] = {}
    store = GraphStore(driver=None)  # type: ignore[arg-type]

    def fake_query(cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        captured["cypher"] = cypher
        captured["params"] = params or {}
        return []

    store.query = fake_query  # type: ignore[method-assign]

    store.upsert_relation(relation("AI Model", "HasParameter", "learning_rate"))

    cypher = captured["cypher"]
    params = captured["params"]

    assert "MERGE (s:Node {name: $source_name})" in cypher
    assert "MERGE (t:Node {name: $target_name})" in cypher
    assert "MERGE (s)-[rel:`has_parameter`]->(t)" in cypher
    assert ":REL" not in cypher
    assert "label" not in params["on_create"]
    assert params["source_name"] == "AI Model"
    assert params["target_name"] == "learning_rate"
    # Node-name mirrors are no longer stored as relationship properties.
    assert "source" not in params["on_create"]
    assert "target" not in params["on_create"]
    assert params["on_create"]["provenance_source"] == "review.xlsx"


def test_graph_store_upsert_sanitizes_non_standard_predicate() -> None:
    captured: dict[str, Any] = {}
    store = GraphStore(driver=None)  # type: ignore[arg-type]

    def fake_query(cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        captured["cypher"] = cypher
        captured["params"] = params or {}
        return []

    store.query = fake_query  # type: ignore[method-assign]

    store.upsert_relation(relation("AI Model", "Invented", "Human Oversight"))

    assert "MERGE (s)-[rel:`invented`]->(t)" in captured["cypher"]


def test_graph_store_upsert_rejects_unsafe_predicate_before_query() -> None:
    store = GraphStore(driver=None)  # type: ignore[arg-type]
    called = False

    def fake_query(cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        nonlocal called
        called = True
        return []

    store.query = fake_query  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="Unsafe Neo4j relationship type"):
        store.upsert_relation(relation("AI Model", "123", "Human Oversight"))

    assert called is False


def test_retriever_reads_graph_node_names_and_relationship_types(monkeypatch: pytest.MonkeyPatch) -> None:
    queries: list[str] = []

    class FakeGraph:
        def query_relations(
            self,
            cypher: str,
            params: dict[str, Any] | None = None,
            **_: Any,
        ) -> Sequence[GraphRelation]:
            queries.append(cypher)
            if len(queries) == 1:
                return [relation("Explainability", "applies_to", "Human Oversight")]
            return []

    monkeypatch.setattr("kbdebugger.graph.retriever.get_graph", lambda: FakeGraph())

    hits = KnowledgeGraphRetriever(limit_per_pattern=3).retrieve("Explain")

    assert hits[0]["match_pattern"] == "source_name"
    assert hits[0]["relation"]["source"]["label"] == "Explainability"

    joined = "\n".join(queries)
    assert "MATCH (n:Node)-[r]->(m:Node)" in joined
    assert "n.name AS source" in joined
    assert "m.name AS target" in joined
    assert "type(r) AS predicate" in joined
    assert ":REL" not in joined
    assert "n.label" not in joined
    assert "m.label" not in joined
