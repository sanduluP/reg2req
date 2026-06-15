from __future__ import annotations

import json

from kbdebugger.graph.store import GraphStore
from kbdebugger.graph.utils import map_extracted_triplets_to_graph_relations


def _relation_with_provenance() -> dict:
    return {
        "source": {"label": "explainability"},
        "target": {"label": "transparency"},
        "edge": {
            "label": "ContributesTo",
            "properties": {
                "sentence": "Explainability contributes to transparency.",
                "source": "ISO24028.pdf",
                "provenance": {
                    "doc_name": "ISO24028.pdf",
                    "quality": "Explainability contributes to transparency.",
                    "chunk_index": 41,
                    "chunk_excerpt": "Explainability is one of the elements...",
                },
            },
        },
    }


def test_map_extracted_triplets_carries_provenance() -> None:
    extraction = {
        "sentence": "Explainability contributes to transparency.",
        "triplets": [("explainability", "transparency", "ContributesTo")],
        "provenance": {
            "doc_name": "ISO24028.pdf",
            "quality": "Explainability contributes to transparency.",
            "chunk_index": 41,
            "chunk_excerpt": "Explainability is one of the elements...",
        },
    }

    rels = map_extracted_triplets_to_graph_relations(extraction, source="ISO24028.pdf")

    assert len(rels) == 1
    props = rels[0]["edge"]["properties"]
    assert props["provenance"]["doc_name"] == "ISO24028.pdf"
    assert props["provenance"]["chunk_index"] == 41
    assert props["source"] == "ISO24028.pdf"


def test_upsert_relation_appends_provenance_records(monkeypatch) -> None:
    captured: dict = {}

    store = GraphStore(driver=None)  # type: ignore[arg-type]

    def fake_query(cypher, params=None):
        captured["cypher"] = cypher
        captured["params"] = params or {}
        return []

    monkeypatch.setattr(store, "query", fake_query)

    store.upsert_relation(_relation_with_provenance())

    cypher = captured["cypher"]
    params = captured["params"]

    # Provenance must be APPENDED (list membership check), never overwritten.
    assert "provenance_records" in cypher
    assert "provenance_docs" in cypher
    assert "coalesce(rel.provenance_records, [])" in cypher

    assert params["prov_doc"] == "ISO24028.pdf"
    record = json.loads(params["prov_json"])
    assert record["doc"] == "ISO24028.pdf"
    assert record["quality"] == "Explainability contributes to transparency."
    assert record["chunk_index"] == 41
    assert record["chunk_excerpt"].startswith("Explainability is one")

    # The raw nested dict must NOT leak into Neo4j edge properties.
    assert "provenance" not in params["on_create"]
    assert params["on_create"]["provenance_source"] == "ISO24028.pdf"


def test_upsert_relation_without_provenance_keeps_lists_untouched(monkeypatch) -> None:
    captured: dict = {}

    store = GraphStore(driver=None)  # type: ignore[arg-type]

    def fake_query(cypher, params=None):
        captured["params"] = params or {}
        return []

    monkeypatch.setattr(store, "query", fake_query)

    relation = _relation_with_provenance()
    del relation["edge"]["properties"]["provenance"]
    relation["edge"]["properties"].pop("source")

    store.upsert_relation(relation)

    # Empty markers mean the Cypher CASE keeps existing lists as-is.
    assert captured["params"]["prov_json"] == ""
    assert captured["params"]["prov_doc"] == ""
