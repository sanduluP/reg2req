"""
Graph-related API routes.

Responsibilities
---------------
- Serve Cytoscape-ready graph payloads.
- Serve curated search keywords for dropdown selection.

Design principles
-----------------
- No Neo4j driver usage here.
- No retrieval internals here.
- Only stage API calls + presentation adapters.
"""

from __future__ import annotations

import os
import traceback
from threading import Thread

from flask import Blueprint, jsonify, request, render_template

from ..services.search_keywords_service import load_search_keywords
from ..services.pipeline_config_service import get_pipeline_config
from ..services.job_store import JOB_STORE

graph_bp = Blueprint("graph", __name__)
# name does not affect the URL path.
# Only the url_prefix in app.register_blueprint(...) controls that.

@graph_bp.get("/")
def index():
    """
    Render the main UI page.
    """
    return render_template("index.html")


@graph_bp.get("/search-keywords")
def api_search_keywords():
    """
    Return the list of allowed Trustworthy-AI keywords for the UI dropdown.

    Returns
    -------
    JSON
        {"keywords": ["Human agency and oversight", ...]}
    """
    try:
        keywords = load_search_keywords()
        return jsonify({"keywords": keywords})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@graph_bp.post("/seed")
def api_seed_graph():
    """
    Initialize the knowledge graph with curated Trustworthy-AI ground truth so
    a fresh Neo4j is not empty. Runs as a background job; poll status via
    /api/pipeline/jobs/<job_id>.

    Request (JSON, optional)
    ------------------------
    { "reset": true }   # clear the whole graph first, then build the baseline

    Response
    --------
    { "job_id": "<uuid>" }
    """
    payload = request.get_json(silent=True) or {}
    # "Initialize graph" clears the database then creates; accept legacy "clear".
    reset_all = bool(payload.get("reset", payload.get("clear", True)))

    job = JOB_STORE.create_job()

    def worker() -> None:
        from kbdebugger.graph.seed import seed_knowledge_graph

        try:
            JOB_STORE.set_running(job.job_id)
            JOB_STORE.update_progress(
                job.job_id,
                stage="SeedGraph",
                message="🌱 Initializing the knowledge graph with curated ground truth...",
                current=None,
                total=None,
            )
            summary = seed_knowledge_graph(reset_all=reset_all, pretty_print=False)
            JOB_STORE.set_done(job.job_id, {"seed": summary})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            JOB_STORE.set_error(job.job_id, str(e))

    Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job.job_id})


@graph_bp.get("/subgraph")
def api_subgraph():
    from kbdebugger.graph.api import retrieve_keyword_subgraph_cytoscape
    """
    Retrieve a keyword-driven KG subgraph.

    Query Parameters
    ----------------
    keyword : str
        Must match one of the curated search keywords.

    Returns
    -------
    JSON
        CytoscapeGraphPayload:
        {
            "elements": {
                "nodes": [...],
                "edges": [...]
            }
        }
    """
    keyword = (request.args.get("keyword") or "").strip()
    # ✅ this reads a query parameter from URL.
    # Example:
    #   /api/subgraph?keyword=Transparency
    if not keyword:
        return jsonify({"error": "Missing query param: keyword"}), 400

    allowed_keywords = set(load_search_keywords())
    if keyword not in allowed_keywords:
        return jsonify({"error": f"Keyword not allowed: {keyword!r}\n Allowed keywords: {sorted(allowed_keywords)}"}), 400

    try:
        cfg = get_pipeline_config()

        payload = retrieve_keyword_subgraph_cytoscape(
            keyword=keyword,
            limit_per_pattern=cfg.kg_limit_per_pattern,
        )

        return jsonify(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
