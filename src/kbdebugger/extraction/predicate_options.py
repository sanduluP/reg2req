from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

# Normative predicates capture the deontic language of standards documents
# (ISO "shall" / "should" / "may" / "shall not", definitions). They make
# cross-standard comparison possible: obligation-strength conflicts are
# detected by comparing modality on aligned triples.
NORMATIVE_PREDICATES: tuple[str, ...] = (
    "Requires",
    "Recommends",
    "Permits",
    "Prohibits",
    "Defines",
)

DEFAULT_ALLOWED_PREDICATES: tuple[str, ...] = (
    *NORMATIVE_PREDICATES,
    "IsSubclassOf",
    "Implements",
    "IsEquivalentTo",
    "HasParameter",
    "HasInput",
    "HasOutput",
    "HasAttribute",
    "HasPackage",
    "HasPosition",
    "HasName",
    "AppliesTo",
    "IsAn",
    "IsA",
    "IsOfType",
    "Fallback",
    "WithLongDescription",
    "WithDescription",
    "WithParameter",
    "WithLogScale",
    "WithDefault",
    "WithChoices",
    "WithLow",
    "WithHigh",
    "Ensures",
    "ContributesTo",
    "Calls",
    "MightIntroduce",
    "IsThreatTo",
    "IsDimensionOf",
    "AttributesTo",
    "ShouldEnsure",
    "MightMitigate",
    "SurfacesRisk",
    "SensitiveFamily",
    "BelongsToFamily",
    "SuggestsPreprocessing",
    "SuggestsReplacement",
    "WhenBelow",
    "WhenAbove",
    "AndContainsFamily",
    "AndPerformsTask",
    "ImplementedBy",
    "Performs",
    "HasDecisionFunction",
    "HasThreshold",
    "Has",
    "ChecksFor",
    "Evaluates",
)

_SPLIT_RE = re.compile(r"[\n,]+")


def sanitize_allowed_predicates(raw: Any) -> list[str]:
    """Normalize user-provided predicate options while preserving order."""
    if raw is None:
        items: Iterable[Any] = DEFAULT_ALLOWED_PREDICATES
    elif isinstance(raw, str):
        items = _SPLIT_RE.split(raw)
    elif isinstance(raw, Iterable):
        items = raw
    else:
        return []

    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
