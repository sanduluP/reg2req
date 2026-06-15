from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from kbdebugger.types.ui import ProgressCallback
from rich.progress import track

from kbdebugger.compat.langchain import Document
from kbdebugger.utils import batched
from .sentence_to_qualities import build_sentence_decomposer
from .chunk_to_qualities import build_chunk_decomposer, build_chunk_batch_decomposer
from .types import Qualities, TextDecomposer, BatchTextDecomposer, DecomposeMode
from .logging import save_qualities_json

# ---------------------------------------------------------------------------
# Module-level decomposer singletons
# ---------------------------------------------------------------------------
# These are initialized once at import time to avoid re-loading prompt resources
# and few-shot examples repeatedly inside tight loops.
_sentence_to_qualities_decomposer: TextDecomposer = build_sentence_decomposer()
_chunk_to_qualities_decomposer: TextDecomposer = build_chunk_decomposer()
_chunk_batch_to_qualities_decomposer: BatchTextDecomposer = build_chunk_batch_decomposer()


def decompose(
    text: str,
    *,
    mode: DecomposeMode
) -> Qualities:
    """
    Decompose a single input text into "qualities" under the selected mode.

    Parameters
    ----------
    text:
        Input text: either a single sentence or a larger chunk.
    mode:
        - DecomposeMode.SENTENCES:
            Use when `text` is already sentence-like but may contain
            multiple atomic statements that should be split.
            Example:
                "The cat sat on the mat and looked at the dog."
                → ["The cat sat on the mat.", "The cat looked at the dog."]

        - DecomposeMode.CHUNKS:
            Use when `text` is a larger paragraph or chunk and you want to
            extract key qualities / statements.
            Example:
                "Cats are great pets. They are independent and curious animals..."
                → ["Cats are great pets.",
                   "Cats are independent animals.",
                   "Cats are curious animals."]

    Returns
    -------
    list[str]
        A list of short, atomic sentences (qualities).
    """
    match mode:
        case DecomposeMode.SENTENCES:
            return _sentence_to_qualities_decomposer(text)
        case DecomposeMode.CHUNKS:
            return _chunk_to_qualities_decomposer(text)
        case _:
            pass

    # Defensive: this should never happen with the Enum, but keeps mypy happy
    raise ValueError(f"Unsupported DecomposeMode: {mode}")


def _decompose_one_batch(
    batch_id: int,
    group: List[str],
) -> Tuple[int, List[Qualities]]:
    """
    Worker wrapper for parallel batch decomposition.

    Returns
    -------
    (batch_id, batch_results)
        batch_id is used to optionally re-order results deterministically.
    """
    return batch_id, _chunk_batch_to_qualities_decomposer(group)


def _safe_chunk_batch_to_qualities_decomposer(
    group: List[str],
    failures: Optional[List[str]] = None,
) -> List[Qualities]:
    """
    Safe wrapper around the batched decomposer.

    Why this exists
    ---------------
    `ThreadPoolExecutor.map()` will propagate exceptions and stop iteration
    on the first failure. For a pipeline stage, it's usually better to be
    *best-effort* and preserve output alignment.

    Contract
    --------
    Returns `List[Qualities]` aligned with `group` length:
      - one Qualities list per input chunk text
      - on failure: returns `[[], [], ...]` (same length as group) and appends
        the error message to `failures` so the caller can distinguish
        "LLM produced nothing" from "LLM call failed"
    """
    try:
        return _chunk_batch_to_qualities_decomposer(group)
    except Exception as e:  # noqa: BLE001 (intentionally broad in pipeline boundary)
        print(f"[decompose_documents] Batch failed (size={len(group)}): {e}")
        if failures is not None:
            failures.append(str(e))
        return [[] for _ in range(len(group))]


def _raise_if_all_batches_failed(failures: List[str], num_batches: int) -> None:
    """
    A run where EVERY batch raised is an infrastructure failure (LLM backend
    unreachable, bad credentials, decommissioned model), not "no knowledge in
    the document" — surface it as a job error instead of 0 qualities.
    """
    if num_batches > 0 and len(failures) >= num_batches:
        backend = os.getenv("MODEL_BACKEND", "http")
        service_url = os.getenv("MODEL_SERVICE_URL", "")
        raise RuntimeError(
            f"LLM decomposition failed for all {num_batches} batch(es) — the LLM backend "
            f"is likely unreachable or misconfigured (MODEL_BACKEND={backend!r}"
            + (f", MODEL_SERVICE_URL={service_url!r}" if service_url else "")
            + f"). Last error: {failures[-1]}"
        )


def _doc_source_context(doc: Document, index: int) -> dict[str, Any]:
    """
    Build the source context shown by the UI for qualities from this document.
    """
    metadata = dict(getattr(doc, "metadata", {}) or {})
    dl_meta = metadata.get("dl_meta") if isinstance(metadata, dict) else None

    headings = None
    if isinstance(dl_meta, dict):
        raw_headings = dl_meta.get("headings")
        if isinstance(raw_headings, list) and raw_headings:
            headings = [str(h) for h in raw_headings if str(h).strip()]

    compact_metadata: dict[str, Any] = {}
    for key in ("source", "doc_id", "page", "page_no", "page_number"):
        value = metadata.get(key)
        if isinstance(value, (str, int, float, bool)):
            compact_metadata[key] = value

    if headings:
        compact_metadata["headings"] = headings

    return {
        "source_doc_index": index,
        "source_text": getattr(doc, "page_content", "") or "",
        **({"metadata": compact_metadata} if compact_metadata else {}),
    }

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def decompose_documents(
    docs: Sequence[Document],
    *,
    mode: DecomposeMode,
    batch_size: int = 5,
    use_batch_decomposer: bool = True,
    parallel: bool = False,
    max_workers: Optional[int] = 2,
    progress: Optional[ProgressCallback] = None
) -> Tuple[Qualities, dict]:
    """
    Decompose a list of LangChain Documents into a flat list of qualities.

    Returns
    -------
    (qualities, log_payload)
        - qualities: flat list of atomic qualities
        - log_payload: the same payload that was written to disk
    """
    all_qualities: Qualities = []
    quality_sources: list[dict[str, Any]] = []

    if not docs:
        log_payload = save_qualities_json(
            qualities=all_qualities,
            quality_sources=quality_sources,
            mode=mode,
            num_input_docs=0,
            use_batch_decomposer=use_batch_decomposer,
            batch_size=batch_size if use_batch_decomposer else None,
            num_batches=0 if use_batch_decomposer else None,
            parallel=parallel,
            max_workers=max_workers if parallel else None,
        )
        return all_qualities, log_payload

    texts: List[str] = [getattr(doc, "page_content", "") for doc in docs]
    source_contexts = [_doc_source_context(doc, idx) for idx, doc in enumerate(docs)]

    # --- Fast path: batched chunk decomposition ---
    if mode == DecomposeMode.CHUNKS and use_batch_decomposer:
        num_batches = math.ceil(len(texts) / batch_size)
        batch_failures: List[str] = []

        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                results_iter = pool.map(
                    lambda group: _safe_chunk_batch_to_qualities_decomposer(group, batch_failures),
                    batched(texts, batch_size=batch_size),
                )

                for batch_idx, group_results in track(
                    enumerate(results_iter),
                    total=num_batches,
                    description=(
                        f"🧷 LLM Decomposer (parallel): paragraphs → qualities "
                        f"(num_batches={num_batches}, batch size={batch_size})"
                    ),
                ):
                    if progress:
                        progress(
                            batch_idx + 1,  # nicer: 1-based progress
                            num_batches,
                            f"🧷 LLM Decomposer (parallel): Processing batch ({batch_idx+1}/{num_batches}) ..."
                        )

                    start_idx = batch_idx * batch_size
                    for offset, qualities in enumerate(group_results):
                        source_context = source_contexts[start_idx + offset]
                        all_qualities.extend(qualities)
                        quality_sources.extend(
                            {
                                "quality": quality,
                                **source_context,
                            }
                            for quality in qualities
                        )

        else:
            for batch_idx, group in track(
                enumerate(batched(texts, batch_size=batch_size)),
                total=num_batches,
                description=(
                    f"🧷 LLM Decomposer: paragraphs → qualities "
                    f"(num_batches={num_batches}, batch size={batch_size})"
                ),
            ):
                if progress:
                    progress(
                        batch_idx + 1,
                        num_batches,
                        f"🧷 LLM Decomposer: Processing batch ({batch_idx+1}/{num_batches}) ..."
                    )

                group_results: List[Qualities] = _safe_chunk_batch_to_qualities_decomposer(
                    group, batch_failures
                )
                start_idx = batch_idx * batch_size
                for offset, qualities in enumerate(group_results):
                    source_context = source_contexts[start_idx + offset]
                    all_qualities.extend(qualities)
                    quality_sources.extend(
                        {
                            "quality": quality,
                            **source_context,
                        }
                        for quality in qualities
                    )

        _raise_if_all_batches_failed(batch_failures, num_batches)

        log_payload = save_qualities_json(
            qualities=all_qualities,
            quality_sources=quality_sources,
            mode=mode,
            num_input_docs=len(docs),
            use_batch_decomposer=True,
            batch_size=batch_size,
            num_batches=num_batches,
            parallel=parallel,
            max_workers=max_workers if parallel else None,
        )
        if batch_failures:
            log_payload["num_failed_batches"] = len(batch_failures)
            log_payload["batch_failure_messages"] = batch_failures[:5]
        return all_qualities, log_payload

    # --- Default path: one document -> one decompose() call ---
    for doc_idx, text in enumerate(texts):
        qualities = decompose(text, mode=mode)
        all_qualities.extend(qualities)
        source_context = source_contexts[doc_idx]
        quality_sources.extend(
            {
                "quality": quality,
                **source_context,
            }
            for quality in qualities
        )

    log_payload = save_qualities_json(
        qualities=all_qualities,
        quality_sources=quality_sources,
        mode=mode,
        num_input_docs=len(docs),
        use_batch_decomposer=False,
        batch_size=None,
        num_batches=None,
        parallel=False,
        max_workers=None,
    )
    return all_qualities, log_payload
