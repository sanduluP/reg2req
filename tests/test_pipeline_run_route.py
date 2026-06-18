from __future__ import annotations

import io
import time
from pathlib import Path


def _poll_job(job_id: str):
    from ui.services.job_store import JOB_STORE

    record = None
    for _ in range(50):
        record = JOB_STORE.get(job_id)
        if record and record.state in {"done", "error"}:
            break
        time.sleep(0.02)
    return record


def test_run_route_accepts_multiple_documents(monkeypatch):
    from ui.routes import pipeline_routes
    from ui.ui_app.factory import create_app

    captured = {}

    def fake_run_pipeline(*, job_id, file_paths, keyword, cfg, keywords=None):
        captured["file_names"] = [Path(p).name for p in file_paths]
        captured["keyword"] = keyword
        captured["keywords"] = keywords
        return {"_meta": {"source_names": captured["file_names"], "keyword": keyword}}

    monkeypatch.setattr(pipeline_routes, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(pipeline_routes, "get_pipeline_config", lambda: object())

    app = create_app()
    app.testing = True

    response = app.test_client().post(
        "/api/pipeline/run?keyword=transparency",
        data={
            "documents": [
                (io.BytesIO(b"%PDF-1.4 iso"), "iso24028.pdf"),
                (io.BytesIO(b"%PDF-1.4 fraunhofer"), "fraunhofer_catalog.pdf"),
            ],
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    record = _poll_job(response.get_json()["job_id"])

    assert record is not None
    assert record.state == "done"
    assert captured["file_names"] == ["iso24028.pdf", "fraunhofer_catalog.pdf"]
    assert captured["keyword"] == "transparency"


def test_run_route_accepts_legacy_single_document_field(monkeypatch):
    from ui.routes import pipeline_routes
    from ui.ui_app.factory import create_app

    captured = {}

    def fake_run_pipeline(*, job_id, file_paths, keyword, cfg, keywords=None):
        captured["file_names"] = [Path(p).name for p in file_paths]
        captured["keywords"] = keywords
        return {"_meta": {}}

    monkeypatch.setattr(pipeline_routes, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(pipeline_routes, "get_pipeline_config", lambda: object())

    app = create_app()
    app.testing = True

    response = app.test_client().post(
        "/api/pipeline/run?keyword=fairness",
        data={"document": (io.BytesIO(b"%PDF-1.4 single"), "single.pdf")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    record = _poll_job(response.get_json()["job_id"])

    assert record is not None
    assert record.state == "done"
    assert captured["file_names"] == ["single.pdf"]


def test_run_route_rejects_missing_documents(monkeypatch):
    from ui.routes import pipeline_routes
    from ui.ui_app.factory import create_app

    monkeypatch.setattr(pipeline_routes, "get_pipeline_config", lambda: object())

    app = create_app()
    app.testing = True

    response = app.test_client().post(
        "/api/pipeline/run?keyword=fairness",
        data={},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400


def test_kg_upsert_route_accepts_non_standard_predicate_and_provenance(monkeypatch):
    import kbdebugger.graph.api as graph_api
    from kbdebugger.graph.types import BatchUpsertSummary
    from ui.ui_app.factory import create_app

    captured = {}

    def fake_upsert_extracted_triplets(*, extractions, source=None, pretty_print=True):
        captured["extractions"] = extractions
        captured["source"] = source
        return BatchUpsertSummary(attempted=1, succeeded=1, failed=0, errors=[])

    monkeypatch.setattr(graph_api, "upsert_extracted_triplets", fake_upsert_extracted_triplets)

    app = create_app()
    app.testing = True

    response = app.test_client().post(
        "/api/pipeline/kg-upsert",
        json={
            "extractions": [
                {
                    "sentence": "AI providers shall document model limitations.",
                    "triplets": [["AI provider", "model limitations", "MustDocument"]],
                    "provenance": {
                        "doc_name": "iso42001.pdf",
                        "quality": "AI providers shall document model limitations.",
                        "chunk_index": 7,
                        "chunk_excerpt": "Documentation requirements...",
                    },
                },
            ],
            "source": "iso42001.pdf",
        },
    )

    assert response.status_code == 200
    record = _poll_job(response.get_json()["job_id"])

    assert record is not None
    assert record.state == "done"

    extraction = captured["extractions"][0]
    assert extraction["triplets"] == [("AI provider", "model limitations", "MustDocument")]
    assert extraction["provenance"]["doc_name"] == "iso42001.pdf"
    assert extraction["provenance"]["chunk_index"] == 7
    assert captured["source"] == "iso42001.pdf"
