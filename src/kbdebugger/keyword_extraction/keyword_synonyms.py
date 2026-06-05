from functools import lru_cache

from kbdebugger.prompts import render_prompt
from kbdebugger.llm.model_access import respond
from kbdebugger.utils import ensure_json_object
import rich


_CURATED_SYNONYMS: dict[str, tuple[str, ...]] = {
    "accountability": (
        "responsibility",
        "answerability",
        "auditability",
        "traceability",
        "governance",
        "oversight",
        "liability",
        "accountable",
    ),
    "data governance": (
        "data management",
        "data stewardship",
        "data oversight",
        "data control",
        "data policy",
        "data quality management",
        "metadata management",
    ),
    "diversity": (
        "inclusion",
        "representation",
        "variety",
        "heterogeneity",
        "plurality",
        "demographic diversity",
    ),
    "environmental wellbeing": (
        "environmental sustainability",
        "sustainability",
        "ecological impact",
        "environmental impact",
        "resource efficiency",
        "energy efficiency",
    ),
    "explainability": (
        "interpretability",
        "transparency",
        "understandability",
        "comprehensibility",
        "intelligibility",
        "explicability",
        "explanation",
        "explanatory clarity",
        "traceability",
    ),
    "fairness": (
        "equity",
        "impartiality",
        "non-discrimination",
        "unbiasedness",
        "justice",
        "equal treatment",
        "fair treatment",
    ),
    "human agency": (
        "human control",
        "human autonomy",
        "user control",
        "human oversight",
        "human intervention",
        "human decision-making",
    ),
    "human agency and oversight": (
        "human control",
        "human autonomy",
        "user control",
        "human oversight",
        "human intervention",
        "human decision-making",
    ),
    "non-discrimination": (
        "fairness",
        "equal treatment",
        "equity",
        "impartiality",
        "anti-discrimination",
        "unbiasedness",
    ),
    "oversight": (
        "supervision",
        "monitoring",
        "governance",
        "review",
        "audit",
        "human oversight",
    ),
    "privacy": (
        "confidentiality",
        "data protection",
        "information privacy",
        "personal data protection",
        "anonymity",
        "secrecy",
    ),
    "robustness": (
        "resilience",
        "stability",
        "reliability",
        "fault tolerance",
        "dependability",
        "non-fragility",
    ),
    "safety": (
        "harmlessness",
        "risk prevention",
        "risk mitigation",
        "safe operation",
        "hazard prevention",
        "security",
    ),
    "transparency": (
        "openness",
        "clarity",
        "disclosure",
        "visibility",
        "explainability",
        "interpretability",
        "traceability",
    ),
}


def _normalize_keyword(keyword: str) -> str:
    return " ".join(str(keyword or "").lower().split())


def _dedupe_limit(values: list[str] | tuple[str, ...], *, keyword: str, limit: int = 10) -> tuple[str, ...]:
    seen = {keyword}
    clean: list[str] = []
    for value in values:
        term = " ".join(str(value or "").strip().split())
        key = term.lower()
        if not term or key in seen:
            continue
        seen.add(key)
        clean.append(term)
        if len(clean) >= limit:
            break
    return tuple(clean)


def _fallback_synonyms(keyword: str) -> tuple[str, ...]:
    curated = _CURATED_SYNONYMS.get(keyword)
    if curated:
        return _dedupe_limit(curated, keyword=keyword)

    variants: list[str] = []
    if keyword.endswith("ability"):
        stem = keyword[:-7]
        variants.extend([stem + "able", stem + "ation", stem])
    elif keyword.endswith("ibility"):
        stem = keyword[:-7]
        variants.extend([stem + "ible", stem + "ion", stem])
    elif keyword.endswith("ness"):
        variants.append(keyword[:-4])
    elif keyword.endswith("ity"):
        variants.append(keyword[:-3])
    elif keyword.endswith("tion"):
        variants.append(keyword[:-3] + "e")
    elif keyword.endswith("ment"):
        variants.append(keyword[:-4])

    return _dedupe_limit(tuple(variants), keyword=keyword)


@lru_cache(maxsize=128)
def _generate_synonyms_for_normalized_keyword(keyword: str) -> tuple[str, ...]:
    prompt = render_prompt("keyword_synonyms", keyword=keyword)

    try:
        raw = respond(prompt, json_mode=True, temperature=0.0, max_tokens=1024)
        obj = ensure_json_object(raw)
        synonyms = obj.get("synonyms", [])
        if not isinstance(synonyms, list):
            synonyms = []
        normalized = _dedupe_limit(synonyms, keyword=keyword)
        if not normalized:
            normalized = _fallback_synonyms(keyword)
        rich.print("[green][LLM Synonym Generation][/green] Generated synonyms:", list(normalized))
        return normalized
    except Exception as exc:  # noqa: BLE001 - keyword expansion must not abort the pipeline
        fallback = _fallback_synonyms(keyword)
        rich.print(
            "[yellow][LLM Synonym Generation][/yellow] "
            f"Falling back to local synonyms for {keyword!r}: {list(fallback)} "
            f"(reason: {exc})"
        )
        return fallback


def generate_synonyms_for_keyword(keyword: str) -> list[str]:
    """
    Given a user-provided keyword, query an LLM to generate up to 10 semantically similar synonyms.

    Parameters
    ----------
    keyword:
        A single-word or short-phrase topic (e.g., "fairness", "explainability").

    Returns
    -------
    list[str]
        A list of synonym strings to be used in downstream semantic expansion.
    """
    normalized_keyword = _normalize_keyword(keyword)
    if not normalized_keyword:
        return []

    return list(_generate_synonyms_for_normalized_keyword(normalized_keyword))
