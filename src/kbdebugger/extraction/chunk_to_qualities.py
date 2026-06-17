from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Dict, List, TypeVar

import rich

from kbdebugger.extraction.types import BatchTextDecomposer, TextDecomposer, Qualities
from kbdebugger.llm.model_access import respond
from kbdebugger.utils import ensure_json_object
from kbdebugger.prompts import render_prompt, load_json_resource
from .utils import _call_with_rate_limit_retries, coerce_batch_qualities, coerce_qualities, sanitize_chunk

@dataclass(frozen=True)
class ChunkDecomposeConfig:
    # chunks can be longer, so allow more newlines than for single sentences
    prompt_max_newlines: int = 20


@dataclass(frozen=True)
class ChunkBatchDecomposeConfig:
    """
    Configuration for batched chunk decomposition.

    This controls the *output shape* (per-chunk qualities) and the size of a
    single batched request (handled in the caller that groups chunks). This
    builder itself accepts a list of texts and returns a list of qualities with
    matching order.

    Parameters
    ----------
    max_qualities_per_chunk:
        Soft cap enforced in the prompt and hard-capped again in post-processing.
        Keeps output bounded and stable.
    max_tokens:
        LLM output limit for the batched call.
        If you increase batch size substantially, you may need to raise this.
    temperature:
        Should generally remain 0.0 for deterministic, parseable JSON.
    """
    max_qualities_per_chunk: int = 12
    # Output ceiling for the batched call. Kept generous because a reasoning
    # model (e.g. deepseek-r1) spends part of the budget on <think> tokens; too
    # small a ceiling truncates the JSON body and the whole batch is lost.
    max_tokens: int = 4096
    temperature: float = 0.0
    # Retry policy (kept simple and explicit)
    max_retries: int = 8
    # Upper bound when escalating max_tokens across retries (a truncated batch
    # gets more room each attempt, but never beyond this).
    max_tokens_cap: int = 16384


def build_chunk_decomposer(
    config: ChunkDecomposeConfig | None = None,
) -> TextDecomposer:
    cfg = config or ChunkDecomposeConfig()

    # Load few-shot examples once from JSON
    examples = load_json_resource("chunk_decompose")
    examples_json = json.dumps(examples, ensure_ascii=False)

    def decompose_chunk(text: str) -> Qualities:
        """
        Extract ordered, atomic 'qualities' from a larger paragraph/chunk.
        Returns a list of short statements.
        """
        # 1. Light sanitization: collapse all whitespace, keep full content
        # i.e., newlines/tabs → spaces, multiple spaces → single space
        s = re.sub(r"\s+", " ", text).strip()
        if not s:
            return []
        
        # We embed the text as a JSON string literal in the prompt
        text_json = json.dumps(s, ensure_ascii=False)

        # 2. Build prompt from template + JSON examples
        prompt_str = render_prompt(
            "chunk_decompose",
            examples_json=examples_json,
            text_json=text_json,
        )

        # 3. Call LLM
        # raw_response = respond(
        #     prompt_str,
        #     max_tokens=2048,
        #     temperature=0.0,
        #     json_mode=True,
        # )

        # ✅ This ensures: 429 never silently kills a batch.
        # It will wait and retry as Groq instructs.
        raw_response = _call_with_rate_limit_retries(
            lambda: respond(
                prompt_str,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                json_mode=True,
            ),
            max_retries=cfg.max_retries,
        )

        # 4. Parse JSON into Python object
        # qualities = parse_response(response, coercer=coerce_qualities, default=[])
        obj = ensure_json_object(raw_response)
        qualities = coerce_qualities(obj)

        # # 5. Coerce into list[str] (qualities)
        if qualities:
            return qualities


        # 6. Fallback: salvage any plain-text lines from the *raw text*
        fallback = [
            line.strip("-• ").strip()  # strip common bullet prefixes
            for line in str(raw_response).splitlines()
            if line.strip()
        ]
        return fallback if fallback else []
    

    return decompose_chunk


def build_chunk_batch_decomposer(
    config: ChunkBatchDecomposeConfig | None = None,
) -> BatchTextDecomposer:
    """
    Build a decomposer for a BATCH of chunks.

    This is the scalable alternative to calling the single-chunk decomposer in a
    tight loop. It issues *one* LLM call per batch and returns per-chunk results.

    Input/Output contract
    ---------------------
    - Input:  List[str] texts (each treated independently)
    - Output: List[Qualities] in the SAME order and SAME length as input

    Notes
    -----
    - This function does NOT decide batch size. The caller (e.g. `decompose_documents`)
      should group documents into batches and call this decomposer per batch.
    - This decomposer is robust: if the model response is malformed or missing
      entries, the missing chunks receive [] rather than failing the pipeline.
    """
    cfg = config or ChunkBatchDecomposeConfig()

    # Reuse the exact same few-shot examples used by the single-chunk prompt.
    examples = load_json_resource("chunk_decompose")
    examples_json = json.dumps(examples, ensure_ascii=False)

    def decompose_chunks(texts: List[str]) -> List[Qualities]:
        """
        Decompose many chunk texts in one LLM call.

        Parameters
        ----------
        texts:
            Chunk texts (raw). Each text is sanitized to a single line for stability.

        Returns
        -------
        list[Qualities]
            A list aligned with `texts`, where each entry is the extracted qualities
            for the corresponding chunk.
        """
        if not texts:
            return []

        sanitized: List[str] = [sanitize_chunk(t) for t in texts]
        # If some chunks are empty, we still preserve alignment.
        # We'll send them as empty strings; model should return empty qualities.
        chunks_payload = {
            "chunks": [{"id": i, "text": sanitized[i]} for i in range(len(sanitized))]
        }
        chunks_json = json.dumps(chunks_payload, ensure_ascii=False)

        prompt_str = render_prompt(
            "chunk_decompose_batch",
            examples_json=examples_json,
            chunks_json=chunks_json,
            max_qualities_per_chunk=str(cfg.max_qualities_per_chunk),
        )

        # Empty chunks legitimately yield no qualities, so they must not count
        # as a "failed" batch when we decide whether to retry.
        non_empty_chunks = sum(1 for s in sanitized if s)

        # Retry when the batch result is structurally unparseable. The usual
        # cause is a *truncated* JSON body: a reasoning model (deepseek-r1)
        # spends much of its token budget on <think> tokens and runs out before
        # closing the JSON, so the whole batch would otherwise be silently
        # dropped as []. Each attempt raises the token ceiling to give the body
        # room to complete. (A 429 inside respond() is handled separately by
        # _call_with_rate_limit_retries.)
        id_to_qualities: Dict[int, Qualities] = {}
        for attempt in range(1, cfg.max_retries + 1):
            max_tokens = min(cfg.max_tokens * attempt, cfg.max_tokens_cap)
            raw_response = _call_with_rate_limit_retries(
                lambda mt=max_tokens: respond(
                    prompt_str,
                    max_tokens=mt,
                    temperature=cfg.temperature,
                    json_mode=True,
                ),
                max_retries=cfg.max_retries,
            )

            obj = ensure_json_object(raw_response)
            parsed_ok = isinstance(obj.get("results"), list)
            id_to_qualities = coerce_batch_qualities(obj, expected_n=len(texts))

            # Stop once the JSON parsed structurally (even if some chunks are
            # legitimately empty), or there was nothing to decompose anyway.
            if parsed_ok or non_empty_chunks == 0:
                break

            rich.print(
                f"[chunk_batch_decomposer] ⚠️ unparseable batch result "
                f"(attempt {attempt}/{cfg.max_retries}, max_tokens={max_tokens}, "
                f"non_empty_chunks={non_empty_chunks}) — retrying with a higher token budget."
            )
        else:
            rich.print(
                f"[chunk_batch_decomposer] ❌ batch still unparseable after "
                f"{cfg.max_retries} attempts; {non_empty_chunks} chunk(s) will be "
                f"dropped. Lower the batch size or raise max_tokens / max_tokens_cap."
            )

        # Reconstruct a dense, ordered list, applying a hard cap for safety.
        out: List[Qualities] = []
        for i in range(len(texts)):
            q = id_to_qualities.get(i, [])
            if q and cfg.max_qualities_per_chunk > 0:
                q = q[: cfg.max_qualities_per_chunk]
            out.append(q)

        return out

    return decompose_chunks
