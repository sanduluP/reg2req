from __future__ import annotations
import math

"""
Pipeline runner for the UI.

This module runs long stages in a background thread and reports progress
into the job store.

The runner accepts one or more uploaded documents per job. Each document is
parsed and decomposed individually (so every quality keeps its document
provenance), then similarity filtering and novelty classification run once
over the combined pool of qualities.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from kbdebugger.compat.langchain import Document
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

    The context carries document provenance (doc_name, doc_id) alongside the
    source chunk so downstream stages can attach it to triplets and KG writes.
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
            doc_name = str(metadata.get("source") or "").strip()
            if doc_name:
                context["doc_name"] = doc_name
            doc_id = str(metadata.get("doc_id") or "").strip()
            if doc_id:
                context["doc_id"] = doc_id

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
    file_paths: Sequence[Path],
    keyword: str,
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    """
    Run Stage 2 end-to-end over one or more documents and return
    JSON-serializable output.

    Parameters
    ----------
    job_id:
        Job identifier whose progress should be updated.
    file_paths:
        Paths to the uploaded documents (one or more).
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

    paths = [Path(p) for p in file_paths]
    if not paths:
        raise ValueError("run_pipeline requires at least one document path.")

    num_docs = len(paths)

    # ---------------------------
    # Stage 2a: Docling (per document, so provenance stays per-doc)
    # ---------------------------
    all_paragraphs: List[Document] = []
    docling_logs: list[Dict[str, Any]] = []

    for doc_idx, path in enumerate(paths, start=1):
        init_stage(
            job_id=job_id,
            stage="Docling",
            message=f"🦆 Parsing document {doc_idx}/{num_docs} into paragraphs (Docling): {path.name}",
            current=doc_idx - 1,
            total=num_docs,
        )

        paragraphs, docling_log = extract_paragraphs_from_pdf(
            pdf_path=str(path),
            do_ocr=cfg.docling_enable_OCR,
            do_table_structure=cfg.docling_enable_table_recognition,
            drop_reference_section=cfg.drop_reference_section,
            reference_filter_mode=cfg.reference_section_filter_mode,
        )

        # Tag every paragraph with its document identity so qualities,
        # triplets, and KG provenance can always be traced back to the file.
        doc_id = f"doc-{doc_idx}"
        for paragraph in paragraphs:
            metadata = getattr(paragraph, "metadata", None)
            if isinstance(metadata, dict):
                metadata["source"] = path.name
                metadata["doc_id"] = doc_id

        all_paragraphs.extend(paragraphs)
        docling_logs.append({"doc_name": path.name, "doc_id": doc_id, **(docling_log or {})})

    docling_log_combined: Dict[str, Any] = {
        "num_documents": num_docs,
        "documents": docling_logs,
    }

    # ---------------------------
    # Stage 2b: KeyBERT filter (combined paragraph pool)
    # ---------------------------
    total_par = len(all_paragraphs)
    init_stage(
        job_id=job_id,
        stage="KeyBERT",
        message=f"🔎 Scanning {total_par} paragraphs from {num_docs} document(s) for keyword '{keyword}'...",
        current=0,
        total=total_par,
    )

    keybert_result, keybert_log = filter_paragraphs_by_keyword(
        paragraphs=all_paragraphs,
        search_keyword=keyword,
        config=cfg.keybert,
        synonyms_enabled=cfg.keyword_synonyms_enabled,
        synonym_cache_enabled=cfg.keyword_synonym_cache_enabled,
        synonym_cache_path=cfg.keyword_synonym_cache_path,
        synonym_defaults_path=cfg.keyword_synonym_defaults_path,
        synonym_cache_write=cfg.keyword_synonym_cache_write,
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
        "Docling": docling_log_combined,
        "KeyBERT": keybert_log,
        "DecomposerLLM": decomposer_log,
        "SubgraphSimilarity": subgraph_similarity_log,
        "NoveltyLLM": novelty_log,
    }

    # ✅ Add pipeline metadata so UI can keep provenance across stages
    source_names = [p.name for p in paths]
    response["_meta"] = {
        "source": ", ".join(str(p) for p in paths),   # joined paths (legacy display)
        "source_name": ", ".join(source_names),       # joined names (legacy display)
        "sources": [str(p) for p in paths],
        "source_names": source_names,
        "num_documents": num_docs,
        "keyword": keyword,
    }

    return to_jsonable(response)
