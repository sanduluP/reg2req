from __future__ import annotations

"""
Comparison API routes (Compare tab).

Cross-document analysis over the provenance layer:
- GET  /overlap              — coverage + assertion overlap + concept matrix (sync)
- POST /alignment/scan       — propose SAME_AS candidates (background job; embeds node names)
- POST /alignment/decision   — accept/reject one candidate pair (sync)
- POST /conflicts/scan       — typed conflict candidates + LLM verdicts (background job)
- POST /conflicts/decision   — accept/dismiss one conflict candidate (sync)
- GET  /conflicts/recorded   — previously reviewed conflicts (sync)
- GET  /ambiguity            — undefined normative terms, vague language, near-synonyms (sync)

Heavy work (embeddings, LLM calls) runs as background jobs polled via the
shared /api/pipeline/jobs/<id> endpoint, exactly like pipeline stages.
"""

import traceback
from threading import Thread

from flask import Blueprint, jsonify, request

from ui.services.job_store import JOB_STORE

comparison_bp = Blueprint("comparison", __name__)


@comparison_bp.get("/overlap")
def get_overlap_report():
    from kbdebugger.comparison import overlap

    try:
        report = overlap.build_overlap_report()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Overlap report failed: {e}"}), 500

    return jsonify(report)


@comparison_bp.post("/alignment/scan")
def start_alignment_scan():
    payload = request.get_json(silent=True) or {}
    try:
        threshold = float(payload.get("threshold", 0.78))
    except (TypeError, ValueError):
        threshold = 0.78
    try:
        max_candidates = int(payload.get("max_candidates", 200))
    except (TypeError, ValueError):
        max_candidates = 200

    job = JOB_STORE.create_job()

    def worker() -> None:
        from kbdebugger.comparison import alignment

        try:
            JOB_STORE.set_running(job.job_id)
            JOB_STORE.update_progress(
                job.job_id,
                stage="AlignmentScan",
                message="🔗 Embedding KG node names and proposing SAME_AS candidates...",
                current=None,
                total=None,
            )
            candidates = alignment.propose_alignment_candidates(
                threshold=threshold,
                max_candidates=max_candidates,
            )
            JOB_STORE.set_done(job.job_id, {"candidates": candidates})
        except Exception as e:
            traceback.print_exc()
            JOB_STORE.set_error(job.job_id, str(e))

    Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job.job_id})


@comparison_bp.post("/alignment/decision")
def post_alignment_decision():
    from kbdebugger.comparison import alignment

    payload = request.get_json(silent=True) or {}
    term_a = str(payload.get("term_a", "")).strip()
    term_b = str(payload.get("term_b", "")).strip()
    accept = bool(payload.get("accept"))
    score = payload.get("score")
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None

    if not term_a or not term_b or term_a == term_b:
        return jsonify({"error": "term_a and term_b must be two distinct non-empty terms."}), 400

    try:
        alignment.record_alignment_decision(
            term_a=term_a, term_b=term_b, accept=accept, score=score
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Recording alignment decision failed: {e}"}), 500

    return jsonify({"ok": True, "term_a": term_a, "term_b": term_b, "accept": accept})


@comparison_bp.post("/conflicts/scan")
def start_conflict_scan():
    payload = request.get_json(silent=True) or {}
    use_judge = payload.get("judge", True) is not False

    job = JOB_STORE.create_job()

    def worker() -> None:
        from kbdebugger.comparison import conflicts

        try:
            JOB_STORE.set_running(job.job_id)
            JOB_STORE.update_progress(
                job.job_id,
                stage="ConflictScan",
                message="⚖️ Scanning provenance layer for typed conflict candidates...",
                current=None,
                total=None,
            )
            exclude_ids = conflicts.fetch_recorded_conflict_ids()
            candidates = conflicts.find_conflict_candidates(exclude_ids=exclude_ids)

            if use_judge and candidates:
                JOB_STORE.update_progress(
                    job.job_id,
                    stage="ConflictJudgeLLM",
                    message=f"🧑🏻‍⚖️ LLM judging {len(candidates)} conflict candidates...",
                    current=None,
                    total=None,
                )
                candidates = conflicts.judge_conflict_candidates(candidates)

            JOB_STORE.set_done(job.job_id, {"conflicts": candidates})
        except Exception as e:
            traceback.print_exc()
            JOB_STORE.set_error(job.job_id, str(e))

    Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job.job_id})


@comparison_bp.post("/conflicts/decision")
def post_conflict_decision():
    from kbdebugger.comparison import conflicts

    payload = request.get_json(silent=True) or {}
    candidate = payload.get("candidate")
    accepted = bool(payload.get("accept"))

    if not isinstance(candidate, dict) or not str(candidate.get("id", "")).strip():
        return jsonify({"error": "Expected a 'candidate' object with an 'id'."}), 400

    try:
        conflicts.record_conflict_decision(candidate=candidate, accepted=accepted)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Recording conflict decision failed: {e}"}), 500

    return jsonify({"ok": True, "id": candidate["id"], "accept": accepted})


@comparison_bp.get("/conflicts/recorded")
def get_recorded_conflicts():
    from kbdebugger.comparison import conflicts

    try:
        recorded = conflicts.fetch_recorded_conflicts()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Fetching recorded conflicts failed: {e}"}), 500

    return jsonify({"conflicts": recorded})


@comparison_bp.get("/ambiguity")
def get_ambiguity_report():
    from kbdebugger.comparison import ambiguity

    try:
        report = ambiguity.build_ambiguity_report()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Ambiguity report failed: {e}"}), 500

    return jsonify(report)
