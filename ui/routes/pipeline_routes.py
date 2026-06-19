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
from kbdebugger.extraction.predicate_options import (
    DEFAULT_ALLOWED_PREDICATES,
    PREDICATE_FAMILIES,
    FAMILY_LABELS,
    PRESETS,
    DEFAULT_PRESET,
    resolve_extraction_vocabulary,
)

from ui.services.job_store import JOB_STORE
from ui.services.pipeline_runner import run_pipeline
from ui.services.pipeline_config_service import (
    get_pipeline_config,
    current_thresholds,
    apply_threshold_overrides,
)

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


@pipeline_bp.get("/thresholds")
def get_thresholds():
    """Current tunable pipeline thresholds (defaults the UI sliders start from)."""
    return jsonify(current_thresholds())


@pipeline_bp.get("/triplet-predicates")
def get_triplet_predicates():
    """Expose the predicate vocabulary, grouped into families, plus presets."""
    return jsonify(
        {
            "predicates": list(DEFAULT_ALLOWED_PREDICATES),
            "families": {key: list(value) for key, value in PREDICATE_FAMILIES.items()},
            "family_labels": dict(FAMILY_LABELS),
            "presets": {
                key: {
                    "label": preset["label"],
                    "families": list(preset["families"]),
                    "edge_mode": preset["edge_mode"],
                    "modality": preset["modality"],
                }
                for key, preset in PRESETS.items()
            },
            "default_preset": DEFAULT_PRESET,
        }
    )


@pipeline_bp.post("/run")
def start_pipeline_run():
    """
    Start a long-running pipeline job over one or more documents.

    Request
    -------
    multipart/form-data:
        documents: File (repeatable — one part per uploaded document)
        document: File (legacy single-file field, still accepted)
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

    # Keyword scenarios (sentinels sent by the UI):
    #   __ALL_DIMENSIONS__ -> complete scan over the curated dimension list
    #   __CUSTOM__         -> user-supplied keyword(s) from `custom_keywords`
    #   __NO_KEYWORD__     -> whole-document extraction, no chunk filtering
    #   <anything else>    -> a single predefined dimension / keyword
    ALL_DIMENSIONS = "__ALL_DIMENSIONS__"
    CUSTOM = "__CUSTOM__"
    NO_KEYWORD = "__NO_KEYWORD__"
    filter_chunks = True

    src = request.form if request.form else request.args

    if keyword == ALL_DIMENSIONS:
        from ..services.search_keywords_service import load_search_keywords
        dimensions = list(load_search_keywords())
        if not dimensions:
            return jsonify({"error": "No trustworthy-AI dimensions are configured for a complete scan."}), 400
        keyword_label = "All dimensions"
    elif keyword == NO_KEYWORD:
        dimensions = []
        filter_chunks = False
        keyword_label = "Whole document"
    elif keyword == CUSTOM:
        raw = src.get("custom_keywords") or ""
        dimensions = [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]
        dimensions = list(dict.fromkeys(dimensions))
        if not dimensions:
            return jsonify({"error": "Custom keyword mode selected but no keywords were provided."}), 400
        keyword_label = "Custom: " + ", ".join(dimensions)
    else:
        dimensions = [keyword]
        keyword_label = keyword

    files = [f for f in request.files.getlist("documents") if f and f.filename]
    if not files:
        legacy = request.files.get("document")
        if legacy and legacy.filename:
            files = [legacy]

    if not files:
        return jsonify({"error": "Missing file part: documents"}), 400

    job = JOB_STORE.create_job()
    paths = [_save_upload_to_tmp(f) for f in files]

    # Per-run threshold overrides from the UI tuning panel (form or query).
    src = request.form if request.form else request.args
    threshold_overrides = {
        key: src.get(key)
        for key in ("para_threshold", "sim_threshold", "top_k", "kg_limit")
        if src.get(key) not in (None, "")
    }
    cfg = apply_threshold_overrides(get_pipeline_config(), threshold_overrides)

    def worker() -> None:
        try:
            result = run_pipeline(
                job_id=job.job_id,
                file_paths=paths,
                keyword=keyword_label,
                keywords=dimensions,
                cfg=cfg,
                filter_chunks=filter_chunks,
            )
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

    # Resolve the extraction vocabulary from the (optional) settings the UI
    # sends. Absent settings -> the default preset (everything), which matches
    # the previous behavior of using the full predicate list.
    raw_settings = payload.get("extraction_settings")
    settings = raw_settings if isinstance(raw_settings, dict) else {}
    vocab = resolve_extraction_vocabulary(
        preset=settings.get("preset"),
        families=settings.get("families"),
        custom_predicates=settings.get("custom_predicates"),
        excluded_predicates=settings.get("excluded_predicates"),
        edge_mode=settings.get("edge_mode"),
        modality=settings.get("modality_on"),
    )
    allowed_predicates = vocab["allowed_predicates"] or list(DEFAULT_ALLOWED_PREDICATES)
    strict_predicates = vocab["edge_mode"] == "constrained"
    derive_modality = bool(vocab["modality"])
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
                schema_grounding_enabled=getattr(cfg, "schema_grounding_enabled", True),
                strict_predicates=strict_predicates,
                derive_modality_from_predicate=derive_modality,
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
                schema_status = str(result.get("schema_status") or "").strip().upper()
                # EXISTING rows are eligible too: re-submitting an existing triple
                # appends this document's provenance to the edge (the cross-document
                # overlap signal) without changing the knowledge itself.
                result["upsert_eligible"] = schema_status in {"", "SCHEMA_VALID"}
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
                "extraction_vocabulary": {
                    "preset": vocab["preset"],
                    "families": vocab["families"],
                    "edge_mode": vocab["edge_mode"],
                    "modality": vocab["modality"],
                    "custom_predicates": vocab["custom_predicates"],
                },
                "input_count": len(selected_items),
            }

            JOB_STORE.set_done(job.job_id, result)
        except Exception as e:
            traceback.print_exc()
            JOB_STORE.set_error(job.job_id, str(e))

    Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job.job_id})



def _clean_provenance(raw) -> dict | None:
    """Sanitize the per-extraction provenance payload from the review UI."""
    if not isinstance(raw, dict):
        return None

    chunk_index = raw.get("chunk_index")
    if not isinstance(chunk_index, int):
        chunk_index = None

    modality = str(raw.get("modality", "")).strip().upper()
    if modality not in {"MANDATORY", "RECOMMENDED", "OPTIONAL", "PROHIBITED"}:
        modality = ""

    provenance = {
        "doc_name": str(raw.get("doc_name", "")).strip(),
        "quality": str(raw.get("quality", "")).strip(),
        "chunk_index": chunk_index,
        "chunk_excerpt": str(raw.get("chunk_excerpt", "")).strip()[:500],
    }
    if modality:
        provenance["modality"] = modality
    return provenance if any(v for v in provenance.values() if v not in ("", None)) else None


@pipeline_bp.post("/kg-upsert")
def start_kg_upsert():
    from kbdebugger.graph.api import upsert_extracted_triplets
    from kbdebugger.graph.utils import predicate_to_relationship_type
    from kbdebugger.types import ExtractionResult
    """
    Start Stage 7 (KG upsert) as a background job.

    Request (JSON)
    --------------
    {
      "extractions": [
        {
          "sentence": "...",
          "triplets": [["S","O","P"], ...],
          "provenance": {"doc_name": "...", "quality": "...", "chunk_index": 3, "chunk_excerpt": "..."}
        },
        ...
      ],
      "source": "foo.pdf"   # optional global fallback provenance
    }

    Non-standard predicates are accepted (the reviewer explicitly included
    them); they only need to sanitize into a safe Neo4j relationship type.

    Response
    --------
    { "job_id": "<uuid>" }
    """
    payload = request.get_json(silent=True) or {}
    raw_extractions = payload.get("extractions")
    source = payload.get("source")
    unsafe_predicates: list[str] = []

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
                try:
                    predicate_to_relationship_type(p)
                except ValueError:
                    unsafe_predicates.append(p)
                    continue
                cleaned_triplets.append((s, o, p))

        item: ExtractionResult = {"sentence": sentence, "triplets": cleaned_triplets}
        provenance = _clean_provenance(ex.get("provenance"))
        if provenance:
            item["provenance"] = provenance
        cleaned.append(item)

    if unsafe_predicates:
        unsafe = sorted(set(unsafe_predicates))
        return jsonify({"error": f"Predicate(s) cannot be converted to a safe relationship type: {unsafe}"}), 400

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
