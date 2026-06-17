from __future__ import annotations

from typing import Any

from kbdebugger.graph.seed import (
    SEED_KNOWLEDGE_TYPE,
    SEED_SOURCE_LABEL,
    parse_seed_line,
    parse_seed_triples,
    seed_knowledge_graph,
    seed_triples_to_relations,
)
from kbdebugger.graph.utils import predicate_to_relationship_type


def test_parse_seed_line_maps_surface_phrases_to_standard_predicates() -> None:
    cases = {
        "Transparency is a Requirement": ("Transparency", "IsA", "Requirement"),
        "Interpretability is subclass of Transparency": (
            "Interpretability",
            "IsSubclassOf",
            "Transparency",
        ),
        "Quantus_Consistency implements Explanation Consistency": (
            "Quantus_Consistency",
            "Implements",
            "Explanation Consistency",
        ),
        "Quantus_Consistency applies to Tabular Data": (
            "Quantus_Consistency",
            "AppliesTo",
            "Tabular Data",
        ),
        "Equity is dimension of Fairness": ("Equity", "IsDimensionOf", "Fairness"),
        "Bias is threat to Fairness": ("Bias", "IsThreatTo", "Fairness"),
        "Explanation Consistency contributes to Explanation Robustness": (
            "Explanation Consistency",
            "ContributesTo",
            "Explanation Robustness",
        ),
    }
    for line, (subject, predicate, obj) in cases.items():
        triple = parse_seed_line(line)
        assert triple is not None, line
        assert (triple.subject, triple.predicate, triple.object) == (subject, predicate, obj)


def test_parse_seed_line_skips_comments_and_blanks() -> None:
    assert parse_seed_line("# a comment") is None
    assert parse_seed_line("   ") is None
    assert parse_seed_line("no predicate here") is None


def test_parse_seed_line_preserves_multiword_object_and_casing() -> None:
    triple = parse_seed_line("Algorithmic Bias is threat to Diversity Non-Discrimination Fairness")
    assert triple is not None
    assert triple.subject == "Algorithmic Bias"
    assert triple.object == "Diversity Non-Discrimination Fairness"


def test_seed_file_parses_and_all_predicates_are_standard() -> None:
    triples = parse_seed_triples(_seed_text())
    assert len(triples) > 50
    # Every predicate must convert to a safe Neo4j relationship type.
    for predicate in {t.predicate for t in triples}:
        rel_type = predicate_to_relationship_type(predicate)
        assert rel_type and rel_type.islower()


def test_seed_triples_to_relations_carry_pipeline_structure_and_provenance() -> None:
    triples = parse_seed_triples("Bias is threat to Fairness")
    relations = seed_triples_to_relations(triples)
    assert len(relations) == 1

    rel = relations[0]
    assert rel["source"]["label"] == "Bias"
    assert rel["target"]["label"] == "Fairness"
    assert rel["edge"]["label"] == "IsThreatTo"

    props = rel["edge"]["properties"]
    assert props["sentence"] == "Bias is threat to Fairness"
    assert props["source"] == SEED_SOURCE_LABEL
    assert props["knowledge_type"] == SEED_KNOWLEDGE_TYPE
    assert props["provenance"]["doc_name"] == SEED_SOURCE_LABEL
    assert props["provenance"]["quality"] == "Bias is threat to Fairness"
    assert props["provenance"]["chunk_index"] == 0


class FakeGraph:
    def __init__(self) -> None:
        self.upserted: list[Any] = []
        self.queries: list[tuple[str, dict]] = []
        self.reset_called = False

    def upsert_relations(self, relations, *, pretty_print=True):
        from kbdebugger.graph.types import BatchUpsertSummary

        self.upserted = list(relations)
        n = len(self.upserted)
        return BatchUpsertSummary(attempted=n, succeeded=n, failed=0, errors=[])

    def reset_graph(self) -> None:
        self.reset_called = True

    def query(self, cypher, params=None):
        self.queries.append((cypher, params or {}))
        return []


def test_seed_knowledge_graph_upserts_all_relations() -> None:
    graph = FakeGraph()
    summary = seed_knowledge_graph(graph)

    assert summary["relations"] == len(graph.upserted)
    assert summary["relations"] > 50
    assert summary["succeeded"] == summary["relations"]
    assert summary["failed"] == 0
    assert summary["nodes"] > 0
    assert summary["reset_all"] is False
    assert summary["cleared_existing"] is False
    # No clearing queries when neither reset nor clear is requested.
    assert graph.queries == []
    assert graph.reset_called is False


def test_seed_knowledge_graph_reset_all_clears_whole_graph_first() -> None:
    graph = FakeGraph()
    summary = seed_knowledge_graph(graph, reset_all=True)

    assert graph.reset_called is True
    assert summary["reset_all"] is True
    assert summary["relations"] == len(graph.upserted)
    # Full reset uses reset_graph(), not the selective purge queries.
    assert graph.queries == []


def test_seed_knowledge_graph_clear_existing_purges_only_seed_edges() -> None:
    graph = FakeGraph()
    seed_knowledge_graph(graph, clear_existing=True)

    assert graph.reset_called is False
    joined = "\n".join(c for c, _ in graph.queries)
    assert "knowledge_type = $kt" in joined
    assert "DELETE r" in joined
    # Orphan-node cleanup runs after edge deletion.
    assert any("NOT (n)--()" in c for c, _ in graph.queries)
    assert graph.queries[0][1].get("kt") == SEED_KNOWLEDGE_TYPE


def test_seed_route_runs_as_background_job(monkeypatch):
    import time

    import kbdebugger.graph.seed as seed_mod
    from ui.services.job_store import JOB_STORE
    from ui.ui_app.factory import create_app

    captured = {}

    def fake_seed(graph=None, *, reset_all=False, clear_existing=False, pretty_print=False, **kwargs):
        captured["reset_all"] = reset_all
        return {"relations": 79, "nodes": 60, "sentences": 79}

    monkeypatch.setattr(seed_mod, "seed_knowledge_graph", fake_seed)

    app = create_app()
    app.testing = True

    response = app.test_client().post("/api/graph/seed", json={"reset": True})
    assert response.status_code == 200
    job_id = response.get_json()["job_id"]

    record = None
    for _ in range(50):
        record = JOB_STORE.get(job_id)
        if record and record.state in {"done", "error"}:
            break
        time.sleep(0.02)

    assert record is not None and record.state == "done"
    assert record.result["seed"]["relations"] == 79
    assert captured["reset_all"] is True


def _seed_text() -> str:
    from kbdebugger.graph.seed import load_seed_text

    return load_seed_text()
