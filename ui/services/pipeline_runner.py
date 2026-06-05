from __future__ import annotations
import math

"""
Pipeline runner for the UI.

This module runs long stages in a background thread and reports progress
into the job store.

We start with Stage 2 (Docling + KeyBERT + Decomposer),
and we design the function so you can extend it to Stage 3/4
without changing the job API contract.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from kbdebugger.pipeline.config import PipelineConfig
from kbdebugger.extraction.api import extract_paragraphs_from_pdf

from kbdebugger.keyword_extraction.api import filter_paragraphs_by_keyword

from kbdebugger.extraction.api import decompose_paragraphs_to_qualities
# Optional next stages (enable when ready):
from kbdebugger.graph.api import retrieve_keyword_subgraph
from kbdebugger.subgraph_similarity.api import filter_qualities_by_subgraph_similarity
from kbdebugger.novelty.comparator import classify_qualities_novelty

from ui.services.job_store import JOB_STORE, JobProgressStage
from ui.services.json_sanitize import to_jsonable
from ui.services.progress_callbacks import init_stage, make_job_progress_callback


def _source_context_lookup(decomposer_log: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build quality text -> source paragraph context lookup for UI popovers.
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    entries = decomposer_log.get("quality_sources")
    if not isinstance(entries, list):
        return lookup

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        quality = str(entry.get("quality") or "").strip()
        source_text = str(entry.get("source_text") or "").strip()
        if not quality or not source_text or quality in lookup:
            continue

        context: Dict[str, Any] = {
            "source_doc_index": entry.get("source_doc_index"),
            "source_text": source_text,
        }
        metadata = entry.get("metadata")
        if isinstance(metadata, dict) and metadata:
            context["metadata"] = metadata

        lookup[quality] = context

    return lookup


def _attach_source_context(items: list[Dict[str, Any]], lookup: Dict[str, Dict[str, Any]]) -> None:
    """
    Mutate novelty result items with source paragraph context for the frontend.
    """
    for item in items:
        quality = str(item.get("quality") or "").strip()
        context = lookup.get(quality)
        if context:
            item["source_context"] = context

def run_pipeline(
    *,
    job_id: str,
    file_path: Path,
    keyword: str,
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    """
    Run Stage 2 end-to-end and return JSON-serializable output.

    Parameters
    ----------
    job_id:
        Job identifier whose progress should be updated.
    file_path:
        Path to the uploaded document.
    keyword:
        User-selected Trustworthy AI pillar keyword.
    cfg:
        Central PipelineConfig (from_env).

    Returns
    -------
    dict
        JSON payload for the UI.
    """
    JOB_STORE.set_running(job_id)

    # ---------------------------
    # Stage 2a: Docling
    # ---------------------------
    init_stage(
        job_id=job_id,
        stage="Docling",
        message="🦆 Parsing document into paragraphs (Docling)...",
        current=None,
        total=None,
    )

    paragraphs, docling_log = extract_paragraphs_from_pdf(
        pdf_path=str(file_path),
        do_ocr=cfg.docling_enable_OCR,
        do_table_structure=cfg.docling_enable_table_recognition,
    )

    # ---------------------------
    # Stage 2b: KeyBERT filter
    # ---------------------------
    total_par = len(paragraphs)
    init_stage(
        job_id=job_id,
        stage="KeyBERT",
        message=f"🔎 Scanning {total_par} paragraphs for keyword '{keyword}'...",
        current=0,
        total=total_par,
    )

    keybert_result, keybert_log = filter_paragraphs_by_keyword(
        paragraphs=paragraphs,
        search_keyword=keyword,
        config=cfg.keybert,
        synonyms_enabled=cfg.keyword_synonyms_enabled,
        progress=make_job_progress_callback(job_id=job_id, stage="KeyBERT"),
    )
        
    matched_docs = keybert_result.matched_docs

    # ---------------------------
    # Stage 2c: LLM Decomposer
    # ---------------------------
    # NOTE: total here depends on our decomposer loop granularity:
    # - if progress reports batches: total = num_batches
    # - if progress reports paragraphs: total = len(matched_docs)
    num_batches = math.ceil(len(matched_docs) / cfg.decomposer_batch_size)
    init_stage(
        job_id=job_id,
        stage="DecomposerLLM",
        message=f"🧷 LLM Decomposer: Decomposing {len(matched_docs)} matched paragraphs into qualities..",
        current=0,
        total=num_batches,
    )

    qualities, decomposer_log = decompose_paragraphs_to_qualities(
        paragraphs=list(matched_docs),
        batch_size=cfg.decomposer_batch_size,
        parallel=cfg.decomposer_parallel,
        max_workers=cfg.decomposer_max_workers,
        progress=make_job_progress_callback(job_id=job_id, stage="DecomposerLLM")
    )
    quality_source_lookup = _source_context_lookup(decomposer_log)

    # ---------------------------------------------------------------------
    # Stage 3: Quality-to-Subgraph similarity filter (needs KG relations)
    # ---------------------------------------------------------------------
    init_stage(
        job_id=job_id,
        stage="SubgraphSimilarity",
        message="🧠 Filtering qualities by similarity to KG subgraph...",
        current=0,
        total=3,  # 1. 📚 Building KG vector index, 2. 📊 Running similarity search, 3. ✍️ Finalizing logs
    )

    kg_relations = retrieve_keyword_subgraph(
        keyword=keyword,
        limit_per_pattern=cfg.kg_limit_per_pattern,
    )

    # If kg_relations is empty, SubgraphSimilarityFilter.build_index() will crash
    if not kg_relations:
        raise ValueError(f"No KG relations retrieved for keyword {keyword!r}.")

    (kept, dropped), subgraph_similarity_log = filter_qualities_by_subgraph_similarity(
        kg_relations=kg_relations,
        qualities=qualities,
        cfg=cfg.vector_similarity,  # assumes PipelineConfig has vector_similarity field
        pretty_print=False,
        progress=make_job_progress_callback(job_id=job_id, stage="SubgraphSimilarity"),
    )

    # ---------------------------------------------------------------------
    # Stage 4: Novelty decision (LLM comparator)
    # ---------------------------------------------------------------------
    batch_size = cfg.novelty_batch_size
    num_batches = math.ceil(len(kept) / batch_size) if kept else 0

    init_stage(
        job_id=job_id,
        stage="NoveltyLLM",
        message=(
            f"🧑🏻‍⚖️ Novelty comparator: classifying {len(kept)} kept qualities "
            f"(batch size={batch_size}, parallel={cfg.novelty_parallel}, workers={cfg.novelty_max_workers})..."
        ),
        current=0,
        total=max(num_batches, 1),  # avoid total=0 in UI
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
    if isinstance(novelty_results, list):
        _attach_source_context(novelty_results, quality_source_lookup)


    response: Dict[JobProgressStage | str, Dict] = {
        "Docling": docling_log,
        "KeyBERT": keybert_log,
        "DecomposerLLM": decomposer_log,
        "SubgraphSimilarity": subgraph_similarity_log,
        "NoveltyLLM": novelty_log,
    }

    # ✅ Add pipeline metadata so UI can keep provenance across stages
    response["_meta"] = {
        "source": str(file_path),            # e.g., "ui/temp_uploads/foo.pdf"
        "source_name": file_path.name,       # e.g., "foo.pdf" (nice for UI)
        "keyword": keyword,
    }

    return to_jsonable(response)
