from __future__ import annotations

import rich

from typing import Any, Dict, Mapping, Sequence, cast, List
from dataclasses import asdict

from kbdebugger.subgraph_similarity.types import KeptQuality, NeighborHit

from .types import (
    NoveltyDecision,
    NeighborView,
    QualityNoveltyResult,
    QualityNoveltyResultRaw,
    QualityNoveltyInput,
)


def neighbor_hit_to_view(hit: NeighborHit) -> NeighborView | None:
    """
    Convert a rich NeighborHit into a slim NeighborView.

    Returns None if the KG sentence cannot be extracted.
    """
    relation = hit.get("relation")
    if not isinstance(relation, dict):
        return None

    edge = relation.get("edge")
    if not isinstance(edge, dict):
        return None

    props = edge.get("properties")
    if not isinstance(props, dict):
        return None

    sentence = props.get("sentence")
    if not isinstance(sentence, str) or not sentence.strip():
        return None

    score = float(hit.get("score", 0.0))
    return NeighborView(score=score, sentence=sentence.strip())


def kept_quality_to_novelty_input(
    kept: KeptQuality,
    *,
    top_k: int = 3,
) -> QualityNoveltyInput:
    """
    Convert SubgraphSimilarityFilter output (KeptQuality) into Novelty stage input
    (QualityNoveltyInput) with slim neighbors.

    Args:
        kept: KeptQuality from vector filter.
        top_k: How many neighbors to keep (default 3).

    Returns:
        QualityNoveltyInput ready to be fed to the novelty comparator.
    """
    quality = str(kept["quality"]).strip()
    max_score = float(kept["max_score"])

    views: List[NeighborView] = []
    for hit in kept["neighbors"][: max(1, top_k)]:
        view = neighbor_hit_to_view(hit)
        if view is not None:
            views.append(view)

    return QualityNoveltyInput(quality=quality, neighbors=views, max_score=max_score)


def _coerce_float_0_1(value: Any, *, field: str) -> float:
    """Parse float and validate [0,1]."""
    try:
        f = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Field '{field}' must be a number, got: {value!r}") from e
    if not (0.0 <= f <= 1.0):
        raise ValueError(f"Field '{field}' must be within [0,1], got: {f}")
    return f


def _coerce_optional_confidence(value: Any) -> float:
    if value is None or str(value).strip() == "":
        return 0.5
    try:
        return _coerce_float_0_1(value, field="confidence")
    except ValueError as exc:
        rich.print(f"[coerce_quality_novelty_result] ⚠️ {exc}; defaulting confidence to 0.5.")
        return 0.5


def coerce_quality_novelty_result(
        parsed: Mapping[str, Any],
        *,
        novelty_input: QualityNoveltyInput,
    ) -> QualityNoveltyResult:
    """
    Coerce parsed model JSON into a typed QualityNoveltyResult, enriched with
    the original quality + max_score from novelty_input.

    This function is intentionally forgiving:
    - If fields are missing, it falls back to safe defaults.
    - If decision is invalid, it defaults to PARTIALLY_NEW (safe behavior: keep signal).
    - Ensures confidence is in [0,1].
    """
    obj = cast(QualityNoveltyResultRaw, dict(parsed))

    decision_raw = str(obj.get("decision", "")).strip().upper()
    if decision_raw not in {d.value for d in NoveltyDecision}:
        rich.print(
            f"[coerce_quality_novelty_result] ⚠️ Invalid decision '{decision_raw}', defaulting to PARTIALLY_NEW."
        )
        # Safe default: treat as PARTIALLY_NEW to avoid dropping potential signal.
        decision = NoveltyDecision.PARTIALLY_NEW
    else:
        decision = NoveltyDecision(decision_raw)

    rationale = str(parsed.get("rationale", "No rationale provided.")).strip()

    novel_spans = obj.get("novel_spans")
    if not isinstance(novel_spans, list):
        novel_spans = []
    novel_spans = [str(s).strip() for s in novel_spans if str(s).strip()]

    matched = obj.get("matched_neighbor_sentence", None)
    if matched is not None and str(matched).strip():
        matched = str(matched).strip()
    else:
        matched = None

    confidence = _coerce_optional_confidence(obj.get("confidence"))

    # Small consistency guard:
    # EXISTING should not claim novel spans; if it does, downgrade to PARTIALLY_NEW.
    if decision == NoveltyDecision.EXISTING and novel_spans:
        rich.print(
            "[coerce_quality_novelty_result] ⚠️ EXISTING result has novel_spans; downgrading to PARTIALLY_NEW."
        )
        decision = NoveltyDecision.PARTIALLY_NEW

    return QualityNoveltyResult(
        quality=novelty_input.quality,
        max_score=novelty_input.max_score,

        decision=decision,
        rationale=rationale,
        novel_spans=novel_spans,
        matched_neighbor_sentence=matched,
        confidence=confidence,
    )


# ============================
# Batching utilities
# ============================
def kept_batch_to_prompt_items(
        batch: Sequence[KeptQuality], 
        *, 
        id_offset: int
    ) -> List[Dict[str, Any]]:
    """
    Convert a batch of KeptQuality objects into the prompt JSON schema for the batched comparator.

    Parameters
    ----------
    batch:
        Batch of kept qualities.

    id_offset:
        Integer offset added to each item's index to create stable ids across batches.

    Returns
    -------
    list[dict[str, Any]]
        Items compatible with the batched prompt contract.

    Notes
    -----
    We use explicit integer ids to enforce alignment between:
    - input items (kept qualities)
    - output results produced by the LLM

    This is crucial: batched calls must be order-robust.
    """
    items: List[Dict[str, Any]] = []
    for i, kept in enumerate(batch):
        novelty_input = kept_quality_to_novelty_input(kept)
        d = asdict(novelty_input)
        d["id"] = id_offset + i
        items.append(d)
    return items


_BATCH_RESULT_KEYS = ("results", "items", "classifications", "novelty_results", "decisions")
_ID_KEYS = ("id", "item_id", "index")


def _coerce_result_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _payload_without_id(entry: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(entry)
    for key in _ID_KEYS:
        payload.pop(key, None)
    return payload


def _extract_from_mapping_by_id(value: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}
    for key, payload in value.items():
        rid = _coerce_result_id(key)
        if rid is None or not isinstance(payload, Mapping):
            continue
        out[rid] = dict(payload)
    return out


def _extract_from_result_list(
    value: Sequence[Any],
    *,
    expected_ids: Sequence[int] | None = None,
) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}
    positional_payloads: list[Mapping[str, Any]] = []

    for entry in value:
        if not isinstance(entry, Mapping):
            continue

        rid = None
        for key in _ID_KEYS:
            rid = _coerce_result_id(entry.get(key))
            if rid is not None:
                break

        payload = _payload_without_id(entry)
        if rid is None:
            positional_payloads.append(payload)
        else:
            out[rid] = payload

    if not out and expected_ids and len(positional_payloads) == len(expected_ids):
        return {rid: payload for rid, payload in zip(expected_ids, positional_payloads)}

    return out


def _extract_batched_results_by_id(
    parsed: Mapping[str, Any],
    *,
    expected_ids: Sequence[int] | None = None,
) -> Dict[int, Mapping[str, Any]]:
    """Extract batched novelty results from strict and common alternate JSON shapes."""
    expected_ids = tuple(expected_ids or ())

    for key in _BATCH_RESULT_KEYS:
        value = parsed.get(key)
        if isinstance(value, list):
            found = _extract_from_result_list(value, expected_ids=expected_ids)
            if found:
                return found
        if isinstance(value, Mapping):
            found = _extract_from_mapping_by_id(value)
            if found:
                return found

    found = _extract_from_mapping_by_id(parsed)
    if found:
        return found

    direct_id = None
    for key in _ID_KEYS:
        direct_id = _coerce_result_id(parsed.get(key))
        if direct_id is not None:
            break
    if direct_id is not None:
        return {direct_id: _payload_without_id(parsed)}

    if len(expected_ids) == 1 and "decision" in parsed:
        return {expected_ids[0]: dict(parsed)}

    return {}


def _fallback_novelty_result(
    novelty_input: QualityNoveltyInput,
    *,
    reason: str,
) -> QualityNoveltyResult:
    matched = novelty_input.neighbors[0].sentence if novelty_input.neighbors else None
    return QualityNoveltyResult(
        quality=novelty_input.quality,
        max_score=novelty_input.max_score,
        decision=NoveltyDecision.PARTIALLY_NEW,
        rationale=reason,
        novel_spans=[],
        matched_neighbor_sentence=matched,
        confidence=0.0,
    )


def coerce_batched_novelty_response(
    parsed: Mapping[str, Any],
    *,
    id_to_input: Mapping[int, QualityNoveltyInput],
) -> List[QualityNoveltyResult]:
    """
    Parse + validate batched LLM response and coerce into typed results.

    The preferred contract is {"results": [{"id": ..., ...}]}. For robustness
    with reasoning models, common wrapper variations are accepted. Missing items
    are kept for review as PARTIALLY_NEW instead of failing the whole pipeline.
    """
    expected_order = sorted(id_to_input.keys())
    id_to_response = _extract_batched_results_by_id(parsed, expected_ids=expected_order)

    expected_ids = set(expected_order)
    received_ids = set(id_to_response.keys())
    missing = sorted(expected_ids - received_ids)
    extra = sorted(received_ids - expected_ids)

    if missing or extra:
        rich.print(
            "[coerce_batched_novelty_response] ⚠️ Batched novelty response id mismatch; "
            f"missing={missing}, extra={extra}. Missing items will be kept as PARTIALLY_NEW."
        )

    out: List[QualityNoveltyResult] = []
    for rid in expected_order:
        novelty_input = id_to_input[rid]
        payload = id_to_response.get(rid)
        if payload is None:
            out.append(
                _fallback_novelty_result(
                    novelty_input,
                    reason=(
                        "Novelty LLM response did not include a valid result for this item; "
                        "kept for review as PARTIALLY_NEW."
                    ),
                )
            )
            continue

        out.append(coerce_quality_novelty_result(payload, novelty_input=novelty_input))

    return out

# For UI routes, we can reuse the same coercion logic to convert browser-sent novelty results
def coerce_from_browser_dict(d: Dict[str, Any]) -> QualityNoveltyResult:
    """
    Convert a browser-sent novelty result dict back into a fully typed
    QualityNoveltyResult using the shared coercion utility.

    This ensures:
    - decision validation
    - safe defaults
    - confidence clamping
    - EXISTING + novel_spans consistency guard
    - zero duplication of logic

    The browser sends enriched results (quality + max_score included),
    so we reconstruct a minimal QualityNoveltyInput and reuse the
    shared coerce_quality_novelty_result().
    """

    quality = str(d.get("quality") or "").strip()
    if not quality:
        raise ValueError("Missing/empty field: quality")

    max_score = float(d.get("max_score", 0.0))

    novelty_input = QualityNoveltyInput(
        quality=quality,
        max_score=max_score,
        neighbors=[], # since we will be passing this to extract triplets, we don't need the neighbors;
        # We're just passing [] because it is required in QualityNoveltyInput.

        # if your QualityNoveltyInput has additional fields,
        # fill with safe defaults here
    )

    return coerce_quality_novelty_result(
        parsed=d,
        novelty_input=novelty_input,
    )