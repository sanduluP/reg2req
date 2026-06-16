"""Convert KBExtractor extraction output into KGGen's ``Graph`` JSON format.

KGGen's ``Graph`` (kg_gen/models.py) is::

    {"entities": set[str], "edges": set[str], "relations": set[(subject, predicate, object)]}

⚠️ Order gotcha: KBExtractor stores triplets as ``(subject, object, predicate)``
(S-O-P, see ``kbdebugger.types.base.TripletSubjectObjectPredicate``), whereas
KGGen's ``relations`` are ``(subject, predicate, object)`` (S-P-O). This module
reorders accordingly — getting it wrong silently scrambles every triple.

This file is intentionally dependency-free (pure stdlib) so it can be imported
and unit-tested in either virtualenv.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def extraction_results_to_graph_dict(
    results: Iterable[Mapping[str, Any]],
    *,
    lowercase: bool = True,
) -> dict[str, list]:
    """Flatten KBExtractor ``ExtractionResult`` items into a KGGen ``Graph`` dict.

    Parameters
    ----------
    results:
        Iterable of ``ExtractionResult``-like mappings, each with a ``"triplets"``
        list of ``(subject, object, predicate)`` triples.
    lowercase:
        Lowercase entities/edges to match KGGen's own aggregation preprocessing
        (it normalizes nodes/edges to lowercase). Keep ``True`` for a fair
        comparison.

    Returns
    -------
    dict with JSON-serializable ``entities`` / ``edges`` / ``relations`` lists,
    consumable by ``KGGen.from_dict``.
    """
    entities: set[str] = set()
    edges: set[str] = set()
    relations: set[tuple[str, str, str]] = set()

    for result in results:
        triplets = result.get("triplets") or []
        for triplet in triplets:
            spo = _coerce_triplet(triplet, lowercase=lowercase)
            if spo is None:
                continue
            subject, predicate, obj = spo
            relations.add((subject, predicate, obj))
            entities.add(subject)
            entities.add(obj)
            edges.add(predicate)

    return {
        "entities": sorted(entities),
        "edges": sorted(edges),
        # list-of-lists so json.dump emits arrays; KGGen coerces back to tuples
        "relations": sorted([list(r) for r in relations]),
    }


def _coerce_triplet(
    triplet: Sequence[Any],
    *,
    lowercase: bool,
) -> tuple[str, str, str] | None:
    """Validate + reorder one S-O-P triple into an (S, P, O) tuple, or drop it."""
    if not isinstance(triplet, Sequence) or len(triplet) != 3:
        return None

    # KBExtractor order: (subject, object, predicate)
    subject, obj, predicate = (str(x).strip() for x in triplet)
    if not (subject and obj and predicate):
        return None

    if lowercase:
        subject, obj, predicate = subject.lower(), obj.lower(), predicate.lower()

    # KGGen order: (subject, predicate, object)
    return subject, predicate, obj
