from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
from time import perf_counter
from typing import Iterable, List, Sequence
from kbdebugger.llm.hf_backend import use_hf_local, get_hf_causal_model
from kbdebugger.llm.model_access import respond
from kbdebugger.novelty.types import QualityNoveltyResult
from kbdebugger.prompts import load_json_resource, render_prompt
from kbdebugger.utils.json import ensure_json_object
from kbdebugger.types import ExtractionResult
from kbdebugger.utils import batched
from kbdebugger.extraction.utils import (
    coerce_triplets_batch, 
    save_results_json,
    load_triplet_qualifying_decisions,
)
import json
from kbdebugger.subgraph_similarity.types import KeptQuality
from kbdebugger.extraction.predicate_options import sanitize_allowed_predicates
from kbdebugger.standard_schema import (
    SchemaGroundingContext,
    build_schema_grounding_contexts,
    load_standard_schema_profile,
    validate_extraction_with_standard_schema,
)
import torch # type: ignore
import rich
from rich.progress import track

def build_triplet_extraction_prompt_batch(
    sentences: list[str],
    allowed_predicates: Sequence[str] | None = None,
    schema_contexts: Sequence[SchemaGroundingContext] | None = None,
) -> str:
    """
    Build a prompt that asks the LLM to extract schema-shaped triplets for
    multiple sentences in one call, returning a single JSON object.
    """
    examples = load_json_resource("triplets_batch")
    examples_json = json.dumps(examples, ensure_ascii=False)

    payload = []
    for i, sentence in enumerate(sentences):
        clean = sentence.strip()
        if not clean:
            continue
        item = {"id": i, "sentence": clean}
        if schema_contexts is not None and i < len(schema_contexts):
            item["standard_schema"] = schema_contexts[i].to_prompt_dict()
        payload.append(item)

    payload_json = json.dumps(payload, ensure_ascii=False)
    predicates = sanitize_allowed_predicates(allowed_predicates)
    predicates_json = json.dumps(predicates, ensure_ascii=False)

    return render_prompt(
        "triplets_batch",
        examples_json=examples_json,
        predicates_json=predicates_json,
        payload_json=payload_json,
    )


def _extract_batch_via_llm(
    sentences: list[str],
    allowed_predicates: Sequence[str],
    schema_contexts: Sequence[SchemaGroundingContext] | None = None,
) -> list[ExtractionResult]:
    if not sentences:
        return []

    prompt = build_triplet_extraction_prompt_batch(sentences, allowed_predicates, schema_contexts)
    response = respond(
        prompt,
        max_tokens=4096,
        temperature=0.0,
        json_mode=True
    )

    parsed = ensure_json_object(response)
    triplets = coerce_triplets_batch(parsed, sentences, allowed_predicates=allowed_predicates)

    return triplets


@torch.no_grad()
def _extract_batch_via_hf(
    sentences: list[str],
    allowed_predicates: Sequence[str],
    schema_contexts: Sequence[SchemaGroundingContext] | None = None,
) -> list[ExtractionResult]:
    if not sentences:
        return []

    try:
        model, tokenizer, device = get_hf_causal_model()
    except Exception as e:
        rich.print(f"[extract_triplets_batch] ❌ Could not load HF model: {e}")
        return [{"sentence": s, "triplets": []} for s in sentences]

    prompt = build_triplet_extraction_prompt_batch(sentences, allowed_predicates, schema_contexts)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_tokens = model.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=512,
    )[0]

    res_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    res_text = res_text.replace(prompt, "")

    parsed = ensure_json_object(res_text)
    return coerce_triplets_batch(parsed, sentences, allowed_predicates=allowed_predicates)


def extract_triplets_batch(
    sentences: Iterable[str],
    *,
    batch_size: int = 5,
    allowed_predicates: Sequence[str] | None = None,
    parallel: bool = False,
    max_workers: int = 2,
    schema_grounding_enabled: bool = True,
) -> List[ExtractionResult]:
    sent_list = [s.strip() for s in sentences if s and s.strip()]
    if not sent_list:
        return []

    predicates = sanitize_allowed_predicates(allowed_predicates)
    if not predicates:
        return [
            {
                "sentence": sentence,
                "triplets": [],
                "skipped_reason": "No relationship types were provided for triplet extraction.",
            }
            for sentence in sent_list
        ]

    all_results: List[ExtractionResult] = []

    schema_contexts: list[SchemaGroundingContext] | None = None
    if schema_grounding_enabled:
        schema_contexts = build_schema_grounding_contexts(sent_list)
        profile = load_standard_schema_profile()
    else:
        profile = None

    num_batches = math.ceil(len(sent_list) / batch_size) # No iterator materialization
    max_workers = max(1, int(max_workers or 1))
    sentence_batches = list(batched(sent_list, batch_size))
    context_batches = list(batched(schema_contexts, batch_size)) if schema_contexts is not None else [None] * len(sentence_batches)
    batches = list(enumerate(zip(sentence_batches, context_batches), start=1))

    if parallel and num_batches > 1:
        results_by_batch: dict[int, list[ExtractionResult]] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _safe_extract_batch_via_llm,
                    batch_idx=batch_idx,
                    batch=batch,
                    predicates=predicates,
                    num_batches=num_batches,
                    worker_count=max_workers,
                    schema_contexts=batch_contexts,
                    schema_profile=profile,
                ): batch_idx
                for batch_idx, (batch, batch_contexts) in batches
            }

            for future in as_completed(futures):
                batch_idx, batch_results = future.result()
                results_by_batch[batch_idx] = batch_results

        for batch_idx in sorted(results_by_batch):
            all_results.extend(results_by_batch[batch_idx])

        save_results_json(all_results)
        return all_results

    for batch_idx, (batch, batch_contexts) in track(
        batches,
        description=(
            f"🧬 Triplet extraction: sentences → S-P-O. "
            f"(batch size={batch_size}, num_batches={num_batches}, parallel={parallel}, workers={max_workers})"
        ),
        total=num_batches,
    ):
        # if use_hf_local():
        #     batch_results = _extract_batch_via_hf(batch)
        # else:
        #     batch_results = _extract_batch_via_llm(batch, predicates)
        
        _, batch_results = _safe_extract_batch_via_llm(
            batch_idx=batch_idx,
            batch=batch,
            predicates=predicates,
            num_batches=num_batches,
            worker_count=1,
            schema_contexts=batch_contexts,
            schema_profile=profile,
        )
        all_results.extend(batch_results)

    save_results_json(all_results)
    
    return all_results


def extract_triplets_from_novelty_results(
    results: Sequence[QualityNoveltyResult],
    *,
    batch_size: int = 5,
    allowed_predicates: Sequence[str] | None = None,
    parallel: bool = False,
    max_workers: int = 2,
) -> List[ExtractionResult]:
    """
    Extract KG triplets from novelty results based on decision policy.

    This function:
    1) Reads KB_TRIPLET_QUALIFY_DECISIONS from env
    2) Filters QualityNoveltyResult by decision
    3) Extracts the corresponding quality sentences
    4) Calls extract_triplets_batch on them

    Args:
        results:
            Novelty comparator results.
        batch_size:
            Batch size for LLM triplet extraction.

    Returns:
        List of ExtractionResult.
    """
    # qualifying_decisions = load_triplet_qualifying_decisions()

    sentences: List[str] = [
        r.quality
        for r in results
        # if r.decision in qualifying_decisions
    ]

    if not sentences:
        return []

    return extract_triplets_batch(
        sentences,
        batch_size=batch_size,
        allowed_predicates=allowed_predicates,
        parallel=parallel,
        max_workers=max_workers,
    )


def extract_triplets_from_kept_qualities(
    kept_qualities: Sequence[KeptQuality],
    *,
    batch_size: int = 5,
    allowed_predicates: Sequence[str] | None = None,
    parallel: bool = False,
    max_workers: int = 2,
) -> List[ExtractionResult]:

    sentences: List[str] = [
        q["quality"]
        for q in kept_qualities
    ]

    if not sentences:
        return []
    
    return extract_triplets_batch(
        sentences,
        batch_size=batch_size,
        allowed_predicates=allowed_predicates,
        parallel=parallel,
        max_workers=max_workers,
    )


def _safe_extract_batch_via_llm(
    *,
    batch_idx: int,
    batch: list[str],
    predicates: Sequence[str],
    num_batches: int,
    worker_count: int,
    schema_contexts: Sequence[SchemaGroundingContext] | None = None,
    schema_profile=None,
) -> tuple[int, list[ExtractionResult]]:
    model_label = _llm_model_label()
    prompt = build_triplet_extraction_prompt_batch(batch, predicates, schema_contexts)
    started = perf_counter()

    try:
        response = respond(
            prompt,
            max_tokens=4096,
            temperature=0.0,
            json_mode=True,
        )
        parsed = ensure_json_object(response)
        results = coerce_triplets_batch(parsed, batch, allowed_predicates=predicates)
        if schema_contexts is not None and schema_profile is not None:
            results = [
                validate_extraction_with_standard_schema(
                    result,
                    schema_contexts[i],
                    profile=schema_profile,
                    allowed_predicates=predicates,
                )
                for i, result in enumerate(results)
            ]
    except Exception as exc:
        elapsed = perf_counter() - started
        print(
            "[TripletExtractionLLM] "
            f"model={model_label} batch={batch_idx}/{num_batches} size={len(batch)} workers={worker_count} "
            f"prompt_chars={len(prompt)} elapsed_s={elapsed:.2f} failures=1"
        )
        return batch_idx, [
            {
                "sentence": sentence,
                "triplets": [],
                "skipped_reason": f"Triplet extraction failed for this batch: {exc}",
            }
            for sentence in batch
        ]

    elapsed = perf_counter() - started
    print(
        "[TripletExtractionLLM] "
        f"model={model_label} batch={batch_idx}/{num_batches} size={len(batch)} workers={worker_count} "
        f"prompt_chars={len(prompt)} elapsed_s={elapsed:.2f} failures=0"
    )
    return batch_idx, results


def _llm_model_label() -> str:
    return (
        os.getenv("MODEL_SERVICE_NAME")
        or os.getenv("GROQ_MODEL")
        or os.getenv("HF_LOCAL_MODEL")
        or "unknown"
    )
