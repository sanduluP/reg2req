from __future__ import annotations

"""
Verification API routes (Phase B: five-strategy graph verification).

Runs the read-only verifier as a background job, polled via the shared
/api/pipeline/jobs/<id> endpoint exactly like the other pipeline stages.

This stage is intentionally decoupled from the pipeline: the UI triggers it
*after* the graph has been built, so a verification result (or even a
verification error) can never block or roll back a pipeline run.
"""

import traceback
from threading import Thread

from flask import Blueprint, jsonify, request

from ui.services.job_store import JOB_STORE

verification_bp = Blueprint("verification", __name__)


@verification_bp.post("/run")
def start_verification():
    """
    Start a Phase B verification job over the current knowledge graph.

    Request (JSON, all optional)
    ----------------------------
    {
      "sentences": ["...", ...],          # source sentences for the coverage check (S1)
      "allowed_predicates": ["...", ...], # vocabulary used for this run (S2)
      "thresholds": {"S1": 0.9, ...}      # per-strategy pass thresholds
    }

    Response
    --------
    { "job_id": "<uuid>" }   # poll via /api/pipeline/jobs/<job_id>
    """
    payload = request.get_json(silent=True) or {}

    raw_sentences = payload.get("sentences")
    sentences = (
        [str(s).strip() for s in raw_sentences if str(s).strip()]
        if isinstance(raw_sentences, list)
        else None
    )

    raw_allowed = payload.get("allowed_predicates")
    allowed_predicates = (
        [str(p).strip() for p in raw_allowed if str(p).strip()]
        if isinstance(raw_allowed, list)
        else None
    )

    raw_thresholds = payload.get("thresholds")
    thresholds = None
    if isinstance(raw_thresholds, dict):
        thresholds = {}
        for key, value in raw_thresholds.items():
            try:
                thresholds[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

    job = JOB_STORE.create_job()

    def worker() -> None:
        # Import inside the worker so a heavy import never delays the HTTP response.
        from kbdebugger.graph.store import GraphStore
        from kbdebugger.verification import verify_graph

        graph = None
        try:
            JOB_STORE.set_running(job.job_id)
            JOB_STORE.update_progress(
                job.job_id,
                stage="Verification",
                message="🔎 Verifying the knowledge graph (5-strategy Phase B)...",
                current=None,
                total=None,
            )
            graph = GraphStore.connect(verbose=False)
            verdict = verify_graph(
                graph,
                source_sentences=sentences,
                allowed_predicates=allowed_predicates,
                thresholds=thresholds,
            )
            JOB_STORE.set_done(job.job_id, {"verification": verdict})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            JOB_STORE.set_error(job.job_id, str(e))
        finally:
            if graph is not None:
                try:
                    graph.close()
                except Exception:  # noqa: BLE001
                    pass

    Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job.job_id})
