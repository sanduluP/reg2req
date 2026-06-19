from __future__ import annotations

"""
Incremental quality extraction.

When the reviewer lowers the relevance threshold below the value used at run
time, paragraphs that were never decomposed become "included". This module
decomposes ONLY those newly-included paragraphs and classifies them (similarity
+ novelty), reusing the cached run context — so we pay LLM cost strictly for the
delta, never a full re-run.

The returned novelty results have the SAME shape as ``NoveltyLLM.results`` from a
full run, so the UI can merge them straight into the existing review list.
"""

import math
from typing import Any, Dict

from kbdebugger.extraction.api import decompose_paragraphs_to_qualities
from kbdebugger.subgraph_similarity.api import filter_qualities_by_subgraph_similarity
from kbdebugger.novelty.comparator import classify_qualities_novelty

from ui.services.dimension_scan import (
    attach_chunk_relevance_to_results,
    attach_dimensions_to_results,
    build_quality_dimensions,
    dedup_qualities,
)
from ui.services.extraction_context import ExtractionContext, newly_included_indices
from ui.services.json_sanitize import to_jsonable
from ui.services.progress_callbacks import make_job_progress_callback

# Reuse the run's source-context helpers (single source of truth).
from ui.services.pipeline_runner import _source_context_lookup, _attach_source_context


def extend_extraction(
    context: ExtractionContext,
    threshold: float,
    *,
    job_id: str,
) -> Dict[str, Any]:
    """
    Decompose + classify the paragraphs newly included at ``threshold``.

    Returns a JSON-serializable payload:
        {
          "results": [...],            # new NoveltyLLM-style result items
          "added": <int>,              # number of new reviewed qualities
          "decomposed_paragraphs": <int>,
          "threshold": <float>,
        }
    """
    from ui.services.job_store import JOB_STORE

    cfg = context.cfg
    indices = newly_included_indices(context, threshold)
    empty = {"results": [], "added": 0, "decomposed_paragraphs": 0, "threshold": threshold}
    if not indices:
        return empty

    paragraphs = [context.all_paragraphs[i] for i in indices]
    # decompose position (0..n-1) -> dimension set for that paragraph
    dims_by_decompose_index = {
        pos: context.dims_by_paragraph_index.get(orig, set())
        for pos, orig in enumerate(indices)
    }

    # ---- Decompose the delta ----
    num_batches = math.ceil(len(paragraphs) / cfg.decomposer_batch_size) or 1
    JOB_STORE.update_progress(
        job_id,
        stage="DecomposerLLM",
        message=f"🧷 Decomposing {len(paragraphs)} newly included paragraph(s) into qualities…",
        current=0,
        total=num_batches,
    )
    qualities, decomposer_log = decompose_paragraphs_to_qualities(
        paragraphs=paragraphs,
        batch_size=cfg.decomposer_batch_size,
        parallel=cfg.decomposer_parallel,
        max_workers=cfg.decomposer_max_workers,
        progress=make_job_progress_callback(job_id=job_id, stage="DecomposerLLM"),
    )

    quality_source_lookup = _source_context_lookup(decomposer_log)
    quality_dimensions = build_quality_dimensions(
        decomposer_log.get("quality_sources", []), dims_by_decompose_index
    )
    qualities = dedup_qualities(qualities)
    # Skip qualities already produced by the run or a previous extend.
    new_qualities = [q for q in qualities if str(q).strip() not in context.existing_qualities]

    if not new_qualities:
        context.decomposed_indices.update(indices)
        return {**empty, "decomposed_paragraphs": len(indices)}

    # ---- Similarity filter (reuse cached KG subgraph; empty fallback) ----
    if not context.kg_relations:
        kept = [{"quality": q, "max_score": 0.0, "neighbors": []} for q in new_qualities]
    else:
        (kept, _dropped), _log = filter_qualities_by_subgraph_similarity(
            kg_relations=context.kg_relations,
            qualities=new_qualities,
            cfg=cfg.vector_similarity,
            pretty_print=False,
        )

    # ---- Novelty classification ----
    batch_size = cfg.novelty_batch_size
    JOB_STORE.update_progress(
        job_id,
        stage="NoveltyLLM",
        message=f"🧑🏻‍⚖️ Classifying {len(kept)} new qualities…",
        current=0,
        total=max(math.ceil(len(kept) / batch_size) if kept else 1, 1),
    )
    _, novelty_log = classify_qualities_novelty(
        kept,
        max_tokens=cfg.novelty_llm_max_tokens,
        temperature=cfg.novelty_llm_temperature,
        use_batch=True,
        batch_size=batch_size,
        parallel=cfg.novelty_parallel,
        max_workers=cfg.novelty_max_workers,
        pretty_print=False,
        progress=make_job_progress_callback(job_id=job_id, stage="NoveltyLLM"),
    )

    novelty_results = novelty_log.get("results")
    if not isinstance(novelty_results, list):
        novelty_results = []
    else:
        _attach_source_context(novelty_results, quality_source_lookup)
        attach_dimensions_to_results(novelty_results, quality_dimensions)
        attach_chunk_relevance_to_results(
            novelty_results,
            matched_indices=indices,
            chunk_scores_by_dimension=context.chunk_scores_by_dimension,
        )

    # ---- Bookkeeping so the next extend only handles the next delta ----
    context.decomposed_indices.update(indices)
    context.existing_qualities.update(str(q).strip() for q in new_qualities)

    return {
        "results": to_jsonable(novelty_results),
        "added": len(novelty_results),
        "decomposed_paragraphs": len(indices),
        "threshold": threshold,
    }
