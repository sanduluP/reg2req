import os
from random import random
import re
from time import time
from typing import Any, Callable, List, Optional, Sequence, TypeVar

from kbdebugger.novelty.types import NoveltyDecision
from kbdebugger.types import ExtractionResult, TripletSubjectObjectPredicate
from kbdebugger.utils.json import write_json
from kbdebugger.utils.time import now_utc_compact
from .types import Qualities
from typing import Any, Dict


def _predicate_set(allowed_predicates: Optional[Sequence[str]]) -> set[str] | None:
    """
    Lowercased allowed-predicate set for case-insensitive membership tests.

    Predicate vocabulary is PascalCase by convention, but the LLM (and a user
    typing a custom free-text predicate) may differ in casing. Matching
    case-insensitively avoids flagging "provides" as non-standard when the
    allowed list has "Provides". The triplet keeps its original casing;
    Neo4j storage snake-cases it later regardless.
    """
    if allowed_predicates is None:
        return None
    return {p.strip().lower() for p in allowed_predicates if isinstance(p, str) and p.strip()}


def coerce_triplets(
    item: Dict[str, Any],
    fallback_sentence: str,
    *,
    allowed_predicates: Optional[Sequence[str]] = None,
    strict_predicates: bool = False,
    derive_modality_from_predicate: bool = False,
) -> ExtractionResult:
    """
    Coerce a single item dict to ExtractionResult.

    Parameters
    ----------
    strict_predicates:
        Edge-label policy. When False (default, "relaxed"), predicates outside
        the allowed list are KEPT and reported via ``non_standard_predicates``
        so the reviewer can decide. When True ("constrained"), off-vocabulary
        predicates are dropped instead.
    derive_modality_from_predicate:
        When True and the item has no explicit (valid) modality, the deontic
        strength is inferred from strength-bearing predicates (Requires ->
        MANDATORY, Prohibits -> PROHIBITED, etc.). Off by default to preserve
        existing behavior; enabled by the standards extraction path so the
        Compare tab always has a modality to compare.
    """
    from kbdebugger.extraction.predicate_options import modality_from_predicates

    sentence = item.get("sentence", fallback_sentence)
    raw_triplets = item.get("triplets", [])
    allowed = _predicate_set(allowed_predicates)
    triplets: list[TripletSubjectObjectPredicate] = []
    non_standard_predicates: list[str] = []
    dropped_predicates: list[str] = []

    if isinstance(raw_triplets, list):
        for t in raw_triplets:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                subj, obj, rel = t
                if all(isinstance(x, str) for x in (subj, obj, rel)):
                    subj_clean = subj.strip()
                    obj_clean = obj.strip()
                    rel_clean = rel.strip()
                    if not (subj_clean and obj_clean and rel_clean):
                        continue
                    if allowed is not None and rel_clean.lower() not in allowed:
                        if strict_predicates:
                            # Constrained edge labels: drop off-vocabulary.
                            dropped_predicates.append(rel_clean)
                            continue
                        non_standard_predicates.append(rel_clean)
                    triplets.append((subj_clean, obj_clean, rel_clean))

    result: ExtractionResult = {"sentence": str(sentence), "triplets": triplets}
    if non_standard_predicates:
        result["non_standard_predicates"] = sorted(set(non_standard_predicates))

    modality = str(item.get("modality") or "").strip().upper()
    if modality in {"MANDATORY", "RECOMMENDED", "OPTIONAL", "PROHIBITED"}:
        result["modality"] = modality
    elif derive_modality_from_predicate:
        derived = modality_from_predicates(rel for _s, _o, rel in triplets)
        if derived:
            result["modality"] = derived

    skipped_reason_raw = item.get("skipped_reason")
    skipped_reason = str(skipped_reason_raw).strip() if skipped_reason_raw else ""

    if skipped_reason:
        result["skipped_reason"] = skipped_reason
    elif not triplets and dropped_predicates:
        result["skipped_reason"] = (
            "Only off-vocabulary predicate(s) were extracted and dropped in "
            "constrained mode: " + ", ".join(sorted(set(dropped_predicates)))
        )
    elif allowed is not None and not triplets:
        result["skipped_reason"] = "No allowed relationship type fit this quality."

    return result


def coerce_triplets_batch(
    obj: Dict[str, Any],
    sentences: List[str],
    *,
    allowed_predicates: Optional[Sequence[str]] = None,
    strict_predicates: bool = False,
    derive_modality_from_predicate: bool = False,
) -> List[ExtractionResult]:
    """
    Coerce the LLM batch output of shape:
    {
      "triplets_batch": [
        {"id": 0, "sentence": "...", "triplets": [...]},
        ...
      ]
    }
    into a list[ExtractionResult], aligned by input index.
    """
    results: List[ExtractionResult] = [{"sentence": "", "triplets": []} for _ in sentences]

    batch = obj.get("triplets_batch", [])
    if not isinstance(batch, list):
        return [{"sentence": s, "triplets": [], "skipped_reason": "The LLM did not return a triplets_batch array."} for s in sentences]

    for item in batch:
        if not isinstance(item, dict):
            continue

        idx = item.get("id")
        if isinstance(idx, int) and 0 <= idx < len(sentences):
            results[idx] = coerce_triplets(
                item,
                sentences[idx],
                allowed_predicates=allowed_predicates,
                strict_predicates=strict_predicates,
                derive_modality_from_predicate=derive_modality_from_predicate,
            )

    for i, res in enumerate(results):
        if res["sentence"] == "":
            results[i] = {
                "sentence": sentences[i],
                "triplets": [],
                "skipped_reason": "The LLM did not return a result for this quality.",
            }

    return results


def coerce_qualities(obj: Dict) -> Qualities:
    if not isinstance(obj, dict):
        return []
    qualities = obj.get("qualities")
    if not isinstance(qualities, list):
        return []
    out: Qualities = []
    for q in qualities:
        if isinstance(q, str):
            s = q.strip()
            if s:
                out.append(s)
    return out


def save_results_json(results: List[ExtractionResult]) -> None:
    """
    Write extraction results to a JSON file.
    """
    created_at = now_utc_compact()
    data = {
        "results": results,
    }
    path = f"logs/05_triplet_extraction_results_{created_at}.json"
    write_json(path, data)
    print(f"\n[INFO] Wrote JSON results to {path}")


# ---------------------------------------------------------------------------
# Helpers for `build_chunk_batch_decomposer`
# ---------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+") # this matches all whitespace sequences i.e. newlines, tabs, multiple spaces, etc.

def sanitize_chunk(text: str) -> str:
    """
    Normalize a chunk into a single-line string.

    We intentionally avoid aggressive cleaning here: the upstream PDF cleaning
    stage already handles boilerplate/DOI stripping etc. Our goal is only to
    prevent formatting artifacts from confusing the LLM.
    """
    # replace all whitespace sequences (newlines, tabs, multiple spaces) with single space " "
    return _WS_RE.sub(" ", text or "").strip()


def coerce_batch_qualities(
    obj: Any,
    *,
    expected_n: int,
) -> Dict[int, Qualities]:
    """
    Parse the JSON object returned by the batch prompt into an id->qualities map.

    Expected schema (strict, by prompt contract):
        {
          "results": [
            {"id": 0, "qualities": ["...", "..."]},
            {"id": 1, "qualities": []}
          ]
        }

    This parser is defensive:
    - Accepts "id" as int or numeric string.
    - Accepts "qualities" as list[str] or other coercible structures.
    - Ignores unknown items; only keeps ids within range.
    - Returns a possibly sparse mapping; caller fills missing ids with [].
    """
    if not isinstance(obj, dict):
        return {}

    results = obj.get("results")
    if not isinstance(results, list):
        return {}

    out: Dict[int, Qualities] = {}

    for item in results:
        if not isinstance(item, dict):
            continue

        raw_id = item.get("id")
        if raw_id is None:
            continue

        # Coerce id -> int if possible
        chunk_id: Optional[int] = None
        if isinstance(raw_id, int):
            chunk_id = raw_id
        elif isinstance(raw_id, str) and raw_id.strip().isdigit():
            chunk_id = int(raw_id.strip())

        if chunk_id is None:
            continue
        if chunk_id < 0 or chunk_id >= expected_n:
            continue

        raw_qualities = item.get("qualities", [])
        # Try to coerce qualities robustly.
        # - If it's already a list, keep string-like entries.
        # - If it's a dict (rare), attempt coerce_qualities on it.
        qualities: Qualities = []

        if isinstance(raw_qualities, list):
            qualities = [str(x).strip() for x in raw_qualities if str(x).strip()]
        else:
            # Some models might accidentally return {"qualities": [...]} per item.
            # coerce_qualities can often salvage this.
            try:
                qualities = coerce_qualities(raw_qualities)  # type: ignore[arg-type]
            except Exception:
                qualities = []

        out[chunk_id] = qualities

    return out

def load_triplet_qualifying_decisions() -> set[NoveltyDecision]:
    """
    Load which novelty decisions qualify a quality for triplet extraction.

    Environment variable:
        KB_TRIPLET_QUALIFY_DECISIONS=PARTIALLY_NEW,NEW

    Defaults to:
        {"PARTIALLY_NEW", "NEW"}
    """
    raw = os.getenv("KB_TRIPLET_QUALIFY_DECISIONS", "").strip()

    fallback = {
        NoveltyDecision.PARTIALLY_NEW,
        NoveltyDecision.NEW,
    }

    if not raw:
        return fallback
    
    decisions: set[NoveltyDecision] = set()
    for token in raw.split(","):
        token = token.strip().upper()
        if not token:
            continue
        try:
            decisions.add(NoveltyDecision(token))
        except ValueError:
            # Ignore unknown tokens silently
            continue

    # Safety fallback
    if not decisions:
        decisions = fallback

    return decisions

# ---------------------------------------------------------------------------
# Parallelism helpers
# ---------------------------------------------------------------------------
T = TypeVar("T")

_RETRY_AFTER_RE = re.compile(r"try again in\s+([0-9]*\.?[0-9]+)s", re.IGNORECASE)


def _extract_retry_after_seconds(error_text: str) -> Optional[float]:
    """
    Extract a retry delay (in seconds) from Groq-style 429 error messages.

    Example message fragment:
        "Please try again in 13.45s."

    Returns
    -------
    Optional[float]
        The parsed delay in seconds if present, otherwise None.
    """
    match = _RETRY_AFTER_RE.search(error_text or "")
    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None
    

def _call_with_rate_limit_retries(
    fn: Callable[[], T],
    *,
    max_retries: int = 8,
    default_backoff_s: float = 2.0,
    max_sleep_s: float = 30.0,
) -> T:
    """
    Call `fn()` with rate-limit-aware retries.

    Strategy
    --------
    - If the exception message contains "try again in Xs", sleep for X seconds
      (+ small jitter) and retry.
    - Otherwise, use a conservative exponential backoff.

    Why this exists
    ---------------
    Groq on-demand has strict TPM (Token-per-Minute) limits. When we batch or parallelize,
    occasional 429s are expected. Dropping a batch silently corrupts results.

    Raises
    ------
    RuntimeError
        If all retry attempts fail.
    """
    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:  # SDKs often raise generic exceptions
            last_err = e
            msg = str(e)

            retry_after = _extract_retry_after_seconds(msg)
            if retry_after is not None:
                # Add a tiny jitter to avoid synchronizing retries across threads.
                sleep_s = retry_after + random.uniform(0.1, 0.4)
            else:
                # Exponential backoff for unknown transient failures
                sleep_s = default_backoff_s * (2 ** (attempt - 1))

            sleep_s = min(sleep_s, max_sleep_s)
            time.sleep(sleep_s)

    raise RuntimeError(f"LLM call failed after {max_retries} retries") from last_err