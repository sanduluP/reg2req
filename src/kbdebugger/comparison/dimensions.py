from __future__ import annotations

"""
Trustworthy-AI dimension canonicalization.

Different standards use different words for the same dimension (ISO says
"explainability", others say "interpretability" or "explicability"). Without a
canonical mapping, the Compare tab would undercount real overlap simply because
of vocabulary drift between sources.

This is a *static, curated* alias map applied on top of the reviewer-accepted
SAME_AS clusters. Reviewer decisions always win over these defaults — the map
only fills gaps for the well-known dimension synonyms most frameworks converge
on (ISO/IEC 42001, ISO/IEC TR 24028, EU HLEG, NIST AI RMF).
"""

from typing import Iterable

# alias (lowercased) -> canonical dimension label
DIMENSION_ALIASES: dict[str, str] = {
    # Explainability cluster
    "interpretability": "Explainability",
    "explicability": "Explainability",
    "explainable ai": "Explainability",
    "explainability": "Explainability",
    # Transparency
    "transparency": "Transparency",
    "traceability": "Transparency",
    # Fairness / non-discrimination
    "non-discrimination": "Fairness",
    "non discrimination": "Fairness",
    "nondiscrimination": "Fairness",
    "bias mitigation": "Fairness",
    "fairness": "Fairness",
    # Robustness / safety
    "robustness": "Robustness",
    "reliability": "Robustness",
    "resilience": "Robustness",
    "technical robustness": "Robustness",
    "safety": "Safety",
    # Privacy / data governance
    "privacy": "Privacy",
    "data governance": "Privacy",
    "data protection": "Privacy",
    "privacy and data governance": "Privacy",
    # Accountability
    "accountability": "Accountability",
    "auditability": "Accountability",
    # Human oversight / agency
    "human oversight": "Human oversight",
    "human agency": "Human oversight",
    "human agency and oversight": "Human oversight",
    "human-in-the-loop": "Human oversight",
}


def dimension_canon_for_names(names: Iterable[str]) -> dict[str, str]:
    """
    Build a {node_name -> canonical_dimension} map for the given node names,
    based on the curated DIMENSION_ALIASES (case-insensitive). Only names that
    match a known alias are included; everything else is left untouched.
    """
    out: dict[str, str] = {}
    for name in names:
        key = str(name or "").strip().lower()
        canonical = DIMENSION_ALIASES.get(key)
        if canonical and name != canonical:
            out[name] = canonical
    return out
