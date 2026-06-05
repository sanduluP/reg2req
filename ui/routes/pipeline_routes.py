from __future__ import annotations
from typing import List

"""
Pipeline API routes.

Responsibilities
---------------
- Accept uploads and start background pipeline jobs.
- Expose job status for polling.

Notes
-----
Flask is the server. Our 'kbdebugger' code runs *inside* Flask
in the background job thread.
"""
import os
import traceback
from pathlib import Path
from threading import Thread

from flask import Blueprint, jsonify, request

from kbdebugger.pipeline.config import PipelineConfig
from kbdebugger.extraction.predicate_options import DEFAULT_ALLOWED_PREDICATES

from ui.services.job_store import JOB_STORE
from ui.services.pipeline_runner import run_pipeline
from ui.services.pipeline_config_service import get_pipeline_config

from uuid import uuid4
import tempfile

pipeline_bp = Blueprint("pipeline", __name__)


# def _save_upload_to_tmp(file_storage) -> Path:
#     """
#     Save uploaded file to a temporary location under ui/temp_uploads/.

#     Returns
#     -------
#     Path
#         Filesystem path to the saved upload.
#     """
#     # uploads_dir = Path("ui/temp_uploads")
#     uploads_dir = Path(tempfile.gettempdir()) / "kbdebugger_uploads"
#     uploads_dir.mkdir(parents=True, exist_ok=True)

#     # Keep original name; we can sanitize further if needed.
#     dst = uploads_dir / file_storage.filename
#     file_storage.save(dst)
#     return dst


def _save_upload_to_tmp(file_storage) -> Path:
    uploads_dir = Path(tempfile.gettempdir()) / "kbdebugger_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file_storage.filename).name
    # dst = uploads_dir / f"{uuid4()}_{safe_name}"
    dst = uploads_dir / f"{safe_name}"
    file_storage.save(dst)
    return dst


@pipeline_bp.get("/triplet-predicates")
def get_triplet_predicates():
    return jsonify({"predicates": list(DEFAULT_ALLOWED_PREDICATES)})


@pipeline_bp.post("/run")
def start_pipeline_run():
    """
    Start a long-running pipeline job (Stage 2 for now).

    Request
    -------
    multipart/form-data:
        document: File
    query:
        keyword: str

    Response
    --------
    JSON:
        {"job_id": "<uuid>"}
    """
    print(f"[RUN] pid={os.getpid()}")

    keyword = (request.args.get("keyword") or "").strip()
    if not keyword:
        return jsonify({"error": "Missing query param: keyword"}), 400

    if "document" not in request.files:
        return jsonify({"error": "Missing file part: document"}), 400

    file = request.files["document"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    job = JOB_STORE.create_job()
    path = _save_upload_to_tmp(file)

    cfg = get_pipeline_config()

    def worker() -> None:
        try:
            result = run_pipeline(job_id=job.job_id, file_path=path, keyword=keyword, cfg=cfg)
            JOB_STORE.set_done(job.job_id, result)
        except Exception as e:
            traceback.print_exc()
            JOB_STORE.set_error(job.job_id, str(e))

    # Fire a thread in the background 
    # then return an immediate response with the job_id
    # so that the other GET API can keep polling the status of this job
    Thread(target=worker, daemon=True).start()

    return jsonify({"job_id": job.job_id})


@pipeline_bp.get("/jobs/<job_id>")
def get_job_status(job_id: str):
    """
    Poll job status.

    Returns
    -------
    JSON
        {
          "state": "...",
          "stage": "...",
          "message": "...",
          "progress": {"current": ..., "total": ...},
          "result": {...} | null,
          "error": "..." | null
        }
    """
    print(f"[POLL] pid={os.getpid()} job_id={job_id}")
    rec = JOB_STORE.get(job_id)
    if rec is None:
        return jsonify({"error": f"Unknown job_id: {job_id}"}), 404

    return jsonify(
        {
            "state": rec.state,
            "stage": rec.progress.stage,
            "message": rec.progress.message,
            "progress": {"current": rec.progress.current, "total": rec.progress.total},
            "result": rec.result,
            "error": rec.error,
            "started_at": rec.started_at
        }
    )


@pipeline_bp.post("/triplet-extraction")
def start_triplet_extraction():
    from kbdebugger.extraction.triplet_extraction_batch import extract_triplets_batch
    """
    Start Stage 6 (Triplet extraction) as a background job.

    Request (JSON)
    --------------
    {
        "selected_items": [
            {"quality": "...", "source_context": {...}},
            ...
        ]
    }

    Backwards-compatible input: {"selected_qualities": ["...", ...]}
    """
    payload = request.get_json(silent=True) or {}
    raw_items = payload.get("selected_items")
    raw_qualities = payload.get("selected_qualities")

    selected_items: list[dict] = []
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, dict):
                quality = str(item.get("quality", "")).strip()
                if quality:
                    selected_items.append({
                        "quality": quality,
                        "source_context": item.get("source_context"),
                        "decision": item.get("decision"),
                        "max_score": item.get("max_score"),
                        "matched_neighbor_sentence": item.get("matched_neighbor_sentence"),
                    })
            elif item is not None:
                quality = str(item).strip()
                if quality:
                    selected_items.append({"quality": quality, "source_context": None})
    elif isinstance(raw_qualities, list):
        for item in raw_qualities:
            quality = str(item).strip() if item is not None else ""
            if quality:
                selected_items.append({"quality": quality, "source_context": None})
    else:
        return jsonify({"error": "Expected non-empty 'selected_items' or 'selected_qualities' list"}), 400

    if not selected_items:
        return jsonify({"error": "No non-empty qualities provided."}), 400

    qualities = [item["quality"] for item in selected_items]
    allowed_predicates = list(DEFAULT_ALLOWED_PREDICATES)
    job = JOB_STORE.create_job()

    def worker() -> None:
        try:
            JOB_STORE.set_running(job.job_id)
            cfg = get_pipeline_config()
            JOB_STORE.update_progress(
                job.job_id,
                stage="TripletExtractionLLM",
                message=(
                    f"Extracting triplets from {len(qualities)} reviewed qualities "
                    f"(batch size={cfg.triplet_extraction_batch_size}, "
                    f"parallel={cfg.triplet_extraction_parallel}, "
                    f"workers={cfg.triplet_extraction_max_workers})..."
                ),
                current=None,
                total=None,
            )

            extracted = extract_triplets_batch(
                qualities,
                batch_size=cfg.triplet_extraction_batch_size,
                allowed_predicates=allowed_predicates,
                parallel=cfg.triplet_extraction_parallel,
                max_workers=cfg.triplet_extraction_max_workers,
            )

            aligned_extracted: list[dict] = []
            for idx, item in enumerate(selected_items):
                if idx < len(extracted) and isinstance(extracted[idx], dict):
                    result = extracted[idx]
                else:
                    result = {
                        "sentence": item["quality"],
                        "triplets": [],
                        "skipped_reason": "The triplet extractor did not return a result for this quality.",
                    }

                result["original_quality"] = item["quality"]
                decision = str(item.get("decision") or "").strip().upper()
                result["upsert_eligible"] = decision != "EXISTING"
                if decision:
                    result["decision"] = decision

                source_context = item.get("source_context")
                if source_context is not None:
                    result["source_context"] = source_context

                max_score = item.get("max_score")
                if max_score is not None:
                    result["max_score"] = max_score

                matched_neighbor_sentence = str(item.get("matched_neighbor_sentence") or "").strip()
                if matched_neighbor_sentence:
                    result["matched_neighbor_sentence"] = matched_neighbor_sentence

                aligned_extracted.append(result)

            result = {
                "extracted_triplets": aligned_extracted,
                "allowed_predicates": allowed_predicates,
                "input_count": len(selected_items),
            }

            JOB_STORE.set_done(job.job_id, result)
        except Exception as e:
            traceback.print_exc()
            JOB_STORE.set_error(job.job_id, str(e))

    Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job.job_id})



@pipeline_bp.post("/kg-upsert")
def start_kg_upsert():
    from kbdebugger.graph.api import upsert_extracted_triplets
    from kbdebugger.types import ExtractionResult
    """
    Start Stage 7 (KG upsert) as a background job.

    Request (JSON)
    --------------
    {
      "extractions": [
        {"sentence": "...", "triplets": [["S","O","P"], ...]},
        ...
      ],
      "source": "ui/temp_uploads/foo.pdf"   # optional, but recommended
    }

    Response
    --------
    { "job_id": "<uuid>" }
    """
    payload = request.get_json(silent=True) or {}
    raw_extractions = payload.get("extractions")
    source = payload.get("source")
    allowed_predicate_set = set(DEFAULT_ALLOWED_PREDICATES)
    invalid_predicates: list[str] = []

    if not isinstance(raw_extractions, list) or not raw_extractions:
        return jsonify({"error": "Expected non-empty 'extractions' list"}), 400

    # Minimal validation (TypedDict runtime checks)
    cleaned: List[ExtractionResult] = []
    for ex in raw_extractions:
        if not isinstance(ex, dict):
            continue
        sentence = str(ex.get("sentence", "")).strip()
        triplets = ex.get("triplets")
        if not sentence or not isinstance(triplets, list):
            continue

        cleaned_triplets = []
        for t in triplets:
            if not isinstance(t, list) and not isinstance(t, tuple):
                continue
            if len(t) != 3:
                continue
            s = str(t[0]).strip()
            o = str(t[1]).strip()
            p = str(t[2]).strip()
            if s and o and p:
                if allowed_predicate_set and p not in allowed_predicate_set:
                    invalid_predicates.append(p)
                    continue
                cleaned_triplets.append((s, o, p))

        cleaned.append({"sentence": sentence, "triplets": cleaned_triplets})

    if invalid_predicates:
        invalid = sorted(set(invalid_predicates))
        return jsonify({"error": f"Unsupported predicate(s): {invalid}"}), 400

    if not cleaned:
        return jsonify({"error": "No valid ExtractionResult items provided."}), 400

    job = JOB_STORE.create_job()

    def worker() -> None:
        try:
            JOB_STORE.set_running(job.job_id)
            JOB_STORE.update_progress(
                job.job_id,
                stage="KnowledgeGraphUpsert",
                message=f"🗃️ Upserting relations from {len(cleaned)} triplets to the knowledge graph...",
                current=None,
                total=None,
            )

            upsert_summary = upsert_extracted_triplets(
                extractions=cleaned,
                source=str(source).strip() if source else None,
                pretty_print=False,
            )

            # dataclass -> dict for JSON
            JOB_STORE.set_done(job.job_id, {
                "summary": {
                    "attempted": upsert_summary.attempted,
                    "succeeded": upsert_summary.succeeded,
                    "failed": upsert_summary.failed,
                    "errors": upsert_summary.errors,
                }
            })
        except Exception as e:
            traceback.print_exc()
            JOB_STORE.set_error(job.job_id, str(e))

    Thread(target=worker, daemon=True).start()
    
    return jsonify({"job_id": job.job_id})
