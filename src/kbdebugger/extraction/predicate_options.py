from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

# ---------------------------------------------------------------------------
# Predicate families
# ---------------------------------------------------------------------------
# The vocabulary is organized into four orthogonal families so the UI can
# switch whole groups on/off via a preset, instead of forcing the user to tick
# 50+ individual predicates. The three conceptual layers from the methodology
# map onto these families:
#
#   - NORMATIVE  -> standards/deontic relations (Compare-tab comparison)
#   - DIMENSION  -> trustworthy-AI topic anchoring (both objectives)
#   - DATASCIENCE-> ML/pipeline ontology (refined curation -> DS pipeline)
#   - TAXONOMY   -> shared structural relations (both objectives)
#
# Deontic *strength* lives in the per-statement `modality` field, NOT in the
# relation label (see PREDICATE_MODALITY below), so that "shall provide" and
# "may provide" align on the same edge and differ only by modality — which is
# exactly what makes a cross-standard tension surface as a graph pattern.

NORMATIVE_FAMILY: tuple[str, ...] = (
    # Legacy deontic predicates (kept for back-compat); strength is mirrored
    # into `modality` via PREDICATE_MODALITY.
    "Requires",
    "Recommends",
    "Permits",
    "Prohibits",
    "Defines",
    # Deontic-neutral relations preferred for standards prose.
    "Provides",
    "AppliesTo",
    "Constrains",
    "Measures",
    "DependsOn",
    "Ensures",
)

DIMENSION_FAMILY: tuple[str, ...] = (
    "IsDimensionOf",
    "IsThreatTo",
    "ContributesTo",
    "MightMitigate",
    "MightIntroduce",
    "SurfacesRisk",
    "AttributesTo",
    "ShouldEnsure",
)

DATASCIENCE_FAMILY: tuple[str, ...] = (
    "Implements",
    "HasParameter",
    "HasInput",
    "HasOutput",
    "HasAttribute",
    "HasPackage",
    "HasPosition",
    "HasName",
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
    "Calls",
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

TAXONOMY_FAMILY: tuple[str, ...] = (
    "IsSubclassOf",
    "IsEquivalentTo",
    "IsAn",
    "IsA",
)

# Ordered family registry. Iteration order here defines the precedence used
# when assembling the allowed list (earlier families win on ties), matching the
# prompt rule "prefer the earliest matching Predicate in the allowed list".
PREDICATE_FAMILIES: dict[str, tuple[str, ...]] = {
    "normative": NORMATIVE_FAMILY,
    "dimension": DIMENSION_FAMILY,
    "datascience": DATASCIENCE_FAMILY,
    "taxonomy": TAXONOMY_FAMILY,
}

FAMILY_LABELS: dict[str, str] = {
    "normative": "Normative / standards",
    "dimension": "Trustworthy-AI dimension",
    "datascience": "Data-science / pipeline",
    "taxonomy": "Taxonomy / structure",
}

# Back-compat: the previous flat default list. Kept as the union of every
# family so existing callers (schema snake mapping, default extraction) behave
# exactly as before — plus the new deontic-neutral relations.
DEFAULT_ALLOWED_PREDICATES: tuple[str, ...] = tuple(
    dict.fromkeys(
        predicate
        for family in PREDICATE_FAMILIES.values()
        for predicate in family
    )
)

# Retained for callers that still import the old name.
NORMATIVE_PREDICATES: tuple[str, ...] = (
    "Requires",
    "Recommends",
    "Permits",
    "Prohibits",
    "Defines",
)

# ---------------------------------------------------------------------------
# Deontic strength: predicate -> modality
# ---------------------------------------------------------------------------
# When the LLM chooses one of these strength-bearing predicates we mirror the
# implied obligation strength into the `modality` field so the Compare tab has
# a single, consistent source of truth for strength regardless of which
# relation label was used. "Defines" carries no strength and is intentionally
# excluded.
PREDICATE_MODALITY: dict[str, str] = {
    "Requires": "MANDATORY",
    "Recommends": "RECOMMENDED",
    "Permits": "OPTIONAL",
    "Prohibits": "PROHIBITED",
}

# Priority when a single sentence yields several strength-bearing predicates:
# prohibition is the strongest signal, then obligation, recommendation, option.
_MODALITY_PRIORITY: tuple[str, ...] = ("PROHIBITED", "MANDATORY", "RECOMMENDED", "OPTIONAL")

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
# A preset bundles a sensible default for: which families are active, whether
# off-vocabulary (free-text) predicates are allowed, and whether modality is
# enforced. The user can still override any individual switch.
PRESETS: dict[str, dict[str, Any]] = {
    "standards": {
        "label": "Standards comparison",
        "families": ["normative", "dimension", "taxonomy"],
        "edge_mode": "relaxed",   # keep + flag off-vocabulary predicates
        "modality": True,
    },
    "pipeline": {
        "label": "Pipeline curation",
        "families": ["datascience", "taxonomy", "dimension"],
        "edge_mode": "constrained",  # drop off-vocabulary predicates
        "modality": False,
    },
    "everything": {
        "label": "Everything",
        "families": ["normative", "dimension", "datascience", "taxonomy"],
        "edge_mode": "relaxed",
        "modality": True,
    },
}

DEFAULT_PRESET = "everything"

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


def predicates_for_families(
    families: Iterable[str] | None,
    *,
    custom_predicates: Iterable[str] | None = None,
) -> list[str]:
    """
    Assemble an ordered, de-duplicated predicate list from family names plus
    any user-supplied custom predicates.

    Unknown family names are ignored. Custom predicates are appended last (and
    sanitized) so users can extend any group at run time without being bounded
    by the built-in vocabulary.
    """
    out: list[str] = []
    seen: set[str] = set()

    for family in families or ():
        key = str(family).strip().lower()
        for predicate in PREDICATE_FAMILIES.get(key, ()):
            if predicate not in seen:
                seen.add(predicate)
                out.append(predicate)

    for predicate in sanitize_allowed_predicates(list(custom_predicates or [])):
        if predicate and predicate not in seen:
            seen.add(predicate)
            out.append(predicate)

    return out


def resolve_extraction_vocabulary(
    *,
    preset: str | None = None,
    families: Iterable[str] | None = None,
    custom_predicates: Iterable[str] | None = None,
    excluded_predicates: Iterable[str] | None = None,
    edge_mode: str | None = None,
    modality: bool | None = None,
) -> dict[str, Any]:
    """
    Resolve a full extraction configuration from any combination of preset and
    explicit overrides.

    Resolution order:
      1. Start from the named preset (defaults to "everything").
      2. If explicit ``families`` are given, they replace the preset families.
      3. ``edge_mode`` / ``modality`` overrides replace the preset values.
      4. ``custom_predicates`` are always merged into the allowed list.

    Returns a dict with:
      - allowed_predicates: list[str]
      - families: list[str]
      - edge_mode: "relaxed" | "constrained"
      - modality: bool
      - preset: str
      - custom_predicates: list[str]
    """
    preset_key = (preset or "").strip().lower()
    if preset_key not in PRESETS:
        preset_key = DEFAULT_PRESET
    base = PRESETS[preset_key]

    chosen_families = list(families) if families else list(base["families"])
    chosen_families = [
        f for f in (str(x).strip().lower() for x in chosen_families) if f in PREDICATE_FAMILIES
    ]
    if not chosen_families:
        chosen_families = list(base["families"])

    resolved_edge_mode = (edge_mode or base["edge_mode"]).strip().lower()
    if resolved_edge_mode not in {"relaxed", "constrained"}:
        resolved_edge_mode = base["edge_mode"]

    resolved_modality = bool(base["modality"]) if modality is None else bool(modality)

    custom = sanitize_allowed_predicates(list(custom_predicates or []))
    allowed = predicates_for_families(chosen_families, custom_predicates=custom)

    # Individually excluded predicates are removed from the allowed list
    # (case-insensitive), letting the user drop a single predicate without
    # turning off its whole family.
    excluded = {p.lower() for p in sanitize_allowed_predicates(list(excluded_predicates or []))}
    if excluded:
        allowed = [p for p in allowed if p.lower() not in excluded]

    return {
        "preset": preset_key,
        "families": chosen_families,
        "edge_mode": resolved_edge_mode,
        "modality": resolved_modality,
        "custom_predicates": custom,
        "excluded_predicates": sorted(excluded),
        "allowed_predicates": allowed,
    }


def modality_from_predicates(predicates: Iterable[str]) -> str | None:
    """
    Derive a sentence-level modality from strength-bearing predicates.

    Returns the highest-priority modality among the given predicates, or None
    if none of them carry deontic strength.
    """
    found = {
        PREDICATE_MODALITY[p]
        for p in predicates
        if p in PREDICATE_MODALITY
    }
    for modality in _MODALITY_PRIORITY:
        if modality in found:
            return modality
    return None
