from __future__ import annotations

from typing import Any


def test_retrieve_full_graph_cytoscape_queries_all_relations(monkeypatch):
    import kbdebugger.graph.api as graph_api

    captured: dict[str, Any] = {}

    class FakeGraph:
        def query_relations(self, cypher, params=None, **kwargs):
            captured["cypher"] = cypher
            captured["params"] = params or {}
            return [
                {
                    "source": {"label": "Bias", "id": "n1"},
                    "target": {"label": "Fairness", "id": "n2"},
                    "edge": {"label": "is_threat_to", "properties": {"sentence": "Bias is threat to Fairness"}},
                }
            ]

    monkeypatch.setattr(graph_api, "get_graph", lambda: FakeGraph())

    payload = graph_api.retrieve_full_graph_cytoscape()

    # Whole graph, no keyword filter.
    assert "MATCH (n:Node)-[r]->(m:Node)" in captured["cypher"]
    assert "keyword" not in captured["cypher"].lower()
    assert captured["params"].get("limit") == 5000

    elements = payload["elements"]
    labels = {n["data"]["label"] for n in elements["nodes"]}
    assert {"Bias", "Fairness"} <= labels
    assert len(elements["edges"]) == 1


def test_full_graph_route_returns_payload(monkeypatch):
    import kbdebugger.graph.api as graph_api
    from ui.ui_app.factory import create_app

    fake_payload = {"elements": {"nodes": [{"data": {"id": "n1", "label": "Bias"}}], "edges": []}}
    monkeypatch.setattr(graph_api, "retrieve_full_graph_cytoscape", lambda **kwargs: fake_payload)

    app = create_app()
    app.testing = True

    response = app.test_client().get("/api/graph/full")
    assert response.status_code == 200
    assert response.get_json() == fake_payload
