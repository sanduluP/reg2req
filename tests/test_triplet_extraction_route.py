from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass(frozen=True)
class FakePipelineConfig:
    triplet_extraction_batch_size: int = 2
    triplet_extraction_parallel: bool = True
    triplet_extraction_max_workers: int = 2


def test_triplet_extraction_route_preserves_metadata_and_existing_policy(monkeypatch):
    from kbdebugger.extraction import triplet_extraction_batch
    from ui.routes import pipeline_routes
    from ui.services.job_store import JOB_STORE
    from ui.ui_app.factory import create_app

    captured = {}

    def fake_extract_triplets_batch(qualities, **kwargs):
        captured["qualities"] = list(qualities)
        captured.update(kwargs)
        return [
            {
                "sentence": qualities[0],
                "triplets": [["model", "classifier", "IsA"]],
            },
            {
                "sentence": qualities[1],
                "triplets": [],
                "skipped_reason": "No allowed relationship type fit this quality.",
            },
        ]

    monkeypatch.setattr(triplet_extraction_batch, "extract_triplets_batch", fake_extract_triplets_batch)
    monkeypatch.setattr(pipeline_routes, "get_pipeline_config", lambda: FakePipelineConfig())

    app = create_app()
    app.testing = True

    response = app.test_client().post(
        "/api/pipeline/triplet-extraction",
        json={
            "selected_items": [
                {
                    "quality": "A model is a classifier.",
                    "source_context": {"source_doc_index": 3, "source_text": "Chunk A"},
                    "decision": "NEW",
                    "max_score": 0.42,
                    "matched_neighbor_sentence": "nearest A",
                },
                {
                    "quality": "This sentence does not match the predicate list.",
                    "source_context": {"source_doc_index": 4, "source_text": "Chunk B"},
                    "decision": "EXISTING",
                    "max_score": 0.95,
                    "matched_neighbor_sentence": "nearest B",
                },
            ],
        },
    )

    assert response.status_code == 200
    job_id = response.get_json()["job_id"]

    record = None
    for _ in range(50):
        record = JOB_STORE.get(job_id)
        if record and record.state in {"done", "error"}:
            break
        time.sleep(0.02)

    assert record is not None
    assert record.state == "done"
    assert record.result is not None

    result = record.result
    rows = result["extracted_triplets"]

    assert captured["qualities"] == [
        "A model is a classifier.",
        "This sentence does not match the predicate list.",
    ]
    assert captured["batch_size"] == 2
    assert captured["parallel"] is True
    assert captured["max_workers"] == 2

    assert result["input_count"] == 2
    assert rows[0]["original_quality"] == "A model is a classifier."
    assert rows[0]["source_context"]["source_text"] == "Chunk A"
    assert rows[0]["decision"] == "NEW"
    assert rows[0]["max_score"] == 0.42
    assert rows[0]["matched_neighbor_sentence"] == "nearest A"
    assert rows[0]["upsert_eligible"] is True

    assert rows[1]["skipped_reason"] == "No allowed relationship type fit this quality."
    assert rows[1]["original_quality"] == "This sentence does not match the predicate list."
    assert rows[1]["decision"] == "EXISTING"
    assert rows[1]["upsert_eligible"] is False
