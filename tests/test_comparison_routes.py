from __future__ import annotations

import time


def _poll_job(job_id: str):
    from ui.services.job_store import JOB_STORE

    record = None
    for _ in range(50):
        record = JOB_STORE.get(job_id)
        if record and record.state in {"done", "error"}:
            break
        time.sleep(0.02)
    return record


def _app():
    from ui.ui_app.factory import create_app

    app = create_app()
    app.testing = True
    return app


def test_overlap_route_returns_report(monkeypatch):
    import kbdebugger.comparison.overlap as overlap

    fake_report = {
        "coverage": [{"doc": "iso.pdf", "assertions": 3, "concepts": 4, "normative_statements": 1}],
        "overlap": [],
        "concepts": {"documents": ["iso.pdf"], "rows": []},
        "num_edges_with_provenance": 3,
    }
    monkeypatch.setattr(overlap, "build_overlap_report", lambda graph=None, **kwargs: fake_report)

    response = _app().test_client().get("/api/comparison/overlap")

    assert response.status_code == 200
    assert response.get_json() == fake_report


def test_alignment_scan_route_runs_as_job(monkeypatch):
    import kbdebugger.comparison.alignment as alignment

    candidates = [{"term_a": "transparency", "term_b": "transparence", "score": 0.97}]
    monkeypatch.setattr(
        alignment,
        "propose_alignment_candidates",
        lambda graph=None, **kwargs: candidates,
    )

    response = _app().test_client().post("/api/comparison/alignment/scan", json={})

    assert response.status_code == 200
    record = _poll_job(response.get_json()["job_id"])
    assert record is not None and record.state == "done"
    assert record.result == {"candidates": candidates}


def test_alignment_decision_route_validates_and_records(monkeypatch):
    import kbdebugger.comparison.alignment as alignment

    captured = {}

    def fake_record(graph=None, *, term_a, term_b, accept, score=None):
        captured.update(term_a=term_a, term_b=term_b, accept=accept, score=score)

    monkeypatch.setattr(alignment, "record_alignment_decision", fake_record)

    client = _app().test_client()

    ok = client.post(
        "/api/comparison/alignment/decision",
        json={"term_a": "transparency", "term_b": "transparence", "accept": True, "score": 0.97},
    )
    assert ok.status_code == 200
    assert captured == {
        "term_a": "transparency",
        "term_b": "transparence",
        "accept": True,
        "score": 0.97,
    }

    bad = client.post(
        "/api/comparison/alignment/decision",
        json={"term_a": "transparency", "term_b": "transparency", "accept": True},
    )
    assert bad.status_code == 400


def test_conflict_scan_route_judges_candidates(monkeypatch):
    import kbdebugger.comparison.conflicts as conflicts

    candidate = {
        "id": "modality_conflict-abc",
        "type": "MODALITY_CONFLICT",
        "summary": "s",
        "concept": "provider",
        "side_a": {"doc": "iso.pdf", "text": "shall", "modality": "MANDATORY", "triple": "", "chunk_excerpt": ""},
        "side_b": {"doc": "fh.pdf", "text": "should", "modality": "RECOMMENDED", "triple": "", "chunk_excerpt": ""},
        "verdict": "UNJUDGED",
        "rationale": "",
    }

    monkeypatch.setattr(conflicts, "fetch_recorded_conflict_ids", lambda graph=None: set())
    monkeypatch.setattr(
        conflicts,
        "find_conflict_candidates",
        lambda graph=None, **kwargs: [dict(candidate)],
    )
    monkeypatch.setattr(
        conflicts,
        "judge_conflict_candidates",
        lambda cands, **kwargs: [{**c, "verdict": "TENSION", "rationale": "r"} for c in cands],
    )

    response = _app().test_client().post("/api/comparison/conflicts/scan", json={"judge": True})

    assert response.status_code == 200
    record = _poll_job(response.get_json()["job_id"])
    assert record is not None and record.state == "done"
    assert record.result["conflicts"][0]["verdict"] == "TENSION"


def test_conflict_decision_route(monkeypatch):
    import kbdebugger.comparison.conflicts as conflicts

    captured = {}

    def fake_record(graph=None, *, candidate, accepted):
        captured.update(candidate=candidate, accepted=accepted)

    monkeypatch.setattr(conflicts, "record_conflict_decision", fake_record)

    client = _app().test_client()

    ok = client.post(
        "/api/comparison/conflicts/decision",
        json={"candidate": {"id": "x", "type": "VALUE_CONFLICT"}, "accept": True},
    )
    assert ok.status_code == 200
    assert captured["accepted"] is True
    assert captured["candidate"]["id"] == "x"

    bad = client.post("/api/comparison/conflicts/decision", json={"accept": True})
    assert bad.status_code == 400


def test_ambiguity_route_returns_report(monkeypatch):
    import kbdebugger.comparison.ambiguity as ambiguity

    fake_report = {
        "undefined_normative_terms": [{"doc": "iso.pdf", "term": "human oversight", "predicate": "requires", "example": "..."}],
        "vague_language": [],
        "near_synonyms": [],
    }
    monkeypatch.setattr(ambiguity, "build_ambiguity_report", lambda graph=None, **kwargs: fake_report)

    response = _app().test_client().get("/api/comparison/ambiguity")

    assert response.status_code == 200
    assert response.get_json() == fake_report
