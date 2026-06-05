"""
Novelty comparator for extracted "quality" sentences.

Novelty comparator for extracted "quality" sentences.

This module supports two execution modes:

1) Single-item classification
   - easiest to debug
   - one LLM call per kept quality

2) Batched classification (recommended for throughput)
   - reduces LLM call overhead by grouping items into batches
   - still keeps strict alignment using stable integer ids
   - validates that the model returns exactly one result per input item

The comparator decides whether a candidate quality is:
- EXISTING: no meaningful new semantic value compared to its neighbors
- PARTIALLY_NEW: overlaps strongly but adds meaningful details
- NEW: introduces a new claim/aspect not covered by neighbors
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

import math
import os
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from kbdebugger.llm.model_access import respond
from kbdebugger.prompts import build_prompt, build_prompt_batch
from kbdebugger.subgraph_similarity.types import KeptQuality
from kbdebugger.types.ui import ProgressCallback
from kbdebugger.utils import batched
from rich.progress import track
from .types import (
    QualityNoveltyResult,
    QualityNoveltyInput,
)
from kbdebugger.utils.json import ensure_json_object
from .utils import (
    coerce_quality_novelty_result, 
    kept_quality_to_novelty_input,
    coerce_batched_novelty_response, 
)
from .logging import (
    save_novelty_results_json,
    pretty_print_novelty_results
)

# -------------------------
# Public API
# -------------------------
def classify_quality_novelty(
    kept: KeptQuality,
    *,
    max_tokens: int = 700,
    temperature: float = 0.0,
) -> QualityNoveltyResult:
    """
    Classify novelty for a single kept quality (one LLM call).

    Kept as a debugging-friendly baseline.

    Parameters
    ----------
    kept:
        A single kept quality with its nearest neighbors.

    max_tokens:
        Maximum generation tokens for the LLM output for this one response.

    temperature:
        Decoding temperature.

    Returns:
    -----------
    QualityNoveltyResult
        Typed novelty decision including decision label, rationale, novel spans, etc.

    """
    # Map KeptQuality to the minimal input schema expected by the prompt.
    novelty_input = kept_quality_to_novelty_input(kept)

    prompt = build_prompt(
        prompt_name="quality_novelty_comparator",
        examples_name="quality_novelty_comparator",
        input_obj=novelty_input,
    )
    response = respond(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=True,
    )
    parsed = ensure_json_object(response)
    result = coerce_quality_novelty_result(parsed, novelty_input=novelty_input)
    return result


def classify_qualities_novelty(
    kept_qualities: Sequence[KeptQuality],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    use_batch: bool = True,
    batch_size: int = 5,
    parallel: bool = False,
    max_workers: int = 2,
    pretty_print: bool = True,
    progress: Optional[ProgressCallback] = None,
) -> Tuple[
        Sequence[QualityNoveltyResult], 
        Dict
    ]:
    """
    Classify novelty for a list of kept qualities.

    This function supports:
    - sequential mode (use_batch=False): easiest to debug
    - ⚡️ batched mode (use_batch=True): fewer LLM calls, much faster

    Parameters
    ----------
    kept_qualities:
        List of kept qualities (each includes neighbor relations and similarity scores).

    max_tokens:
        Token budget for the LLM output.

        IMPORTANT:
        - In sequential mode, this is "per item".
        - In batched mode, this is "per batch".
        ⚡️ Increase it when you increase `batch_size` or when rationales get long.

    temperature:
        Decoding temperature.

    use_batch:
        If True, run batched LLM calls. If False, run sequential.

    batch_size:
        Number of kept items per LLM call (batched mode only).

    parallel:
        If True, run batch LLM calls concurrently.

    max_workers:
        Maximum worker threads when parallel is True.

    Returns
    -------
    list[QualityNoveltyResult]
        Typed novelty results aligned with the input order.

    Raises
    ------
    ValueError
        If the batched LLM response does not return exactly one result per input id.
    """
    if not kept_qualities:
        return [], {}

    # -------------------------
    # Sequential mode
    # -------------------------
    if not use_batch:
        results: List[QualityNoveltyResult] = []
        for idx, kept in enumerate(kept_qualities, start=1):
            if progress:
                progress(
                    idx,
                    len(kept_qualities),
                    f"🧑🏻‍⚖️ determining novelty for quality",
                )
            results.append(
                classify_quality_novelty(
                    kept,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            )
        pretty_print_novelty_results(kept=kept_qualities, results=results)
        save_novelty_results_json(results)
        return results, {}

    # -------------------------
    # Batched mode
    # -------------------------
    all_results: List[QualityNoveltyResult] = []
    kept_list = list(kept_qualities)
    num_batches = math.ceil(len(kept_list) / batch_size)
    max_workers = max(1, int(max_workers or 1))
    groups = list(enumerate(batched(kept_list, batch_size=batch_size), start=1))

    if parallel and num_batches > 1:
        results_by_batch: Dict[int, List[QualityNoveltyResult]] = {}
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _classify_novelty_batch,
                    batch_idx=batch_idx,
                    group=group,
                    global_offset=(batch_idx - 1) * batch_size,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    num_batches=num_batches,
                    worker_count=max_workers,
                ): batch_idx
                for batch_idx, group in groups
            }

            for future in as_completed(futures):
                batch_idx, batch_results = future.result()
                results_by_batch[batch_idx] = batch_results
                completed += 1

                if progress:
                    progress(
                        completed,
                        num_batches,
                        f"🧑🏻‍⚖️ completed novelty batch ({completed}/{num_batches})...",
                    )

        for batch_idx in sorted(results_by_batch):
            all_results.extend(results_by_batch[batch_idx])

        if pretty_print:
            pretty_print_novelty_results(kept=kept_qualities, results=all_results)

        log_payload = save_novelty_results_json(all_results)
        return all_results, log_payload

    # Use rich.track only when no UI progress callback is given
    sequential_groups = groups
    if progress is None:
        sequential_groups = track(
            sequential_groups,
            description=(
                f"🧑🏻‍⚖️ LLM Novelty Comparator: batch_size={batch_size}, "
                f"num_batches={num_batches}, parallel={parallel}, workers={max_workers}"
            ),
            total=num_batches,
        )


    for batch_idx, group in sequential_groups:
        if progress:
            progress(
                batch_idx,
                num_batches,
                f"🧑🏻‍⚖️ determining novelty for a batch of qualities (batch size={len(group)})…",
            )

        _, batch_results = _classify_novelty_batch(
            batch_idx=batch_idx,
            group=group,
            global_offset=(batch_idx - 1) * batch_size,
            max_tokens=max_tokens,
            temperature=temperature,
            num_batches=num_batches,
            worker_count=1,
        )
        all_results.extend(batch_results)

    # all_results is in ascending id order, which matches original kept order.

    if pretty_print:
        pretty_print_novelty_results(kept=kept_qualities, results=all_results)
    
    log_payload = save_novelty_results_json(all_results)
    return all_results, log_payload


def _classify_novelty_batch(
    *,
    batch_idx: int,
    group: List[KeptQuality],
    global_offset: int,
    max_tokens: int,
    temperature: float,
    num_batches: int,
    worker_count: int,
) -> Tuple[int, List[QualityNoveltyResult]]:
    model_label = _llm_model_label()
    novelty_inputs: List[QualityNoveltyInput] = [
        kept_quality_to_novelty_input(k) for k in group
    ]

    items_for_prompt: List[Dict[str, Any]] = []
    id_to_input: Dict[int, QualityNoveltyInput] = {}

    for i, ni in enumerate(novelty_inputs):
        rid = global_offset + i
        id_to_input[rid] = ni
        d = asdict(ni)
        d["id"] = rid
        items_for_prompt.append(d)

    prompt = build_prompt_batch(
        prompt_name="quality_novelty_comparator_batch",
        examples_name="quality_novelty_comparator",
        items=items_for_prompt,
    )

    started = perf_counter()
    try:
        response = respond(prompt, max_tokens=max_tokens, temperature=temperature, json_mode=True)
        parsed = ensure_json_object(response)
        batch_results = coerce_batched_novelty_response(parsed, id_to_input=id_to_input)
    except Exception:
        elapsed = perf_counter() - started
        print(
            "[NoveltyLLM] "
            f"model={model_label} batch={batch_idx}/{num_batches} size={len(group)} workers={worker_count} "
            f"prompt_chars={len(prompt)} elapsed_s={elapsed:.2f} failures=1"
        )
        raise

    elapsed = perf_counter() - started
    print(
        "[NoveltyLLM] "
        f"model={model_label} batch={batch_idx}/{num_batches} size={len(group)} workers={worker_count} "
        f"prompt_chars={len(prompt)} elapsed_s={elapsed:.2f} failures=0"
    )
    return batch_idx, batch_results


def _llm_model_label() -> str:
    return (
        os.getenv("MODEL_SERVICE_NAME")
        or os.getenv("GROQ_MODEL")
        or os.getenv("HF_LOCAL_MODEL")
        or "unknown"
    )
