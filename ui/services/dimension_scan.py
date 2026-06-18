from __future__ import annotations

"""
Multi-dimension scan helpers.

A single run can scan one dimension (a keyword) or every trustworthy-AI
dimension at once ("complete scan"). The same paragraph — and therefore the
same quality — can be relevant to several dimensions.

To keep cost down we:
  - run keyword matching per dimension but DECOMPOSE each matched paragraph only
    once (decomposition is the expensive LLM step),
  - deduplicate qualities so similarity / novelty / triplet extraction never
    process the same sentence twice,
  - carry a set of dimension tags on every quality so the reviewer sees which
    dimension(s) each quality belongs to.

These functions are pure (no I/O) so they can be unit-tested without the LLM.
"""

from typing import Any, Iterable, Mapping, Sequence


def assign_paragraph_dimensions(
    all_paragraphs: Sequence[Any],
    matched_docs_by_dimension: Mapping[str, Sequence[Any]],
) -> tuple[list[int], dict[int, set[str]]]:
    """
    Map each matched paragraph (by identity) to the dimensions that matched it.

    Returns
    -------
    (matched_indices, dims_by_index)
        matched_indices: sorted unique indices into ``all_paragraphs`` that
                         matched at least one dimension (the dedup'd pool to
                         decompose).
        dims_by_index:   index-in-all_paragraphs -> set of dimension names.
    """
    index_of = {id(p): i for i, p in enumerate(all_paragraphs)}
    dims_by_index: dict[int, set[str]] = {}

    for dimension, docs in matched_docs_by_dimension.items():
        for doc in docs:
            idx = index_of.get(id(doc))
            if idx is None:
                continue
            dims_by_index.setdefault(idx, set()).add(dimension)

    matched_indices = sorted(dims_by_index.keys())
    return matched_indices, dims_by_index


def build_quality_dimensions(
    quality_sources: Iterable[Mapping[str, Any]],
    dims_by_decompose_index: Mapping[int, set[str]],
) -> dict[str, set[str]]:
    """
    Build a {quality_text -> set(dimensions)} map.

    ``quality_sources`` are the decomposer log entries, each carrying the
    quality text and the ``source_doc_index`` (position in the list of
    paragraphs handed to the decomposer). ``dims_by_decompose_index`` maps that
    same position to the dimension set for the paragraph.
    """
    quality_dims: dict[str, set[str]] = {}
    for entry in quality_sources:
        if not isinstance(entry, Mapping):
            continue
        quality = str(entry.get("quality") or "").strip()
        if not quality:
            continue
        idx = entry.get("source_doc_index")
        dims = dims_by_decompose_index.get(idx) if isinstance(idx, int) else None
        if dims:
            quality_dims.setdefault(quality, set()).update(dims)
        else:
            quality_dims.setdefault(quality, set())
    return quality_dims


def dedup_qualities(qualities: Sequence[str]) -> list[str]:
    """
    Remove duplicate quality strings while preserving first-seen order.

    Dedup is keyed on the trimmed text; the dimension tags for the duplicates
    are already unioned in build_quality_dimensions, so no information is lost.
    """
    seen: set[str] = set()
    out: list[str] = []
    for q in qualities:
        key = str(q or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def attach_dimensions_to_results(
    results: Iterable[dict[str, Any]],
    quality_dims: Mapping[str, set[str]],
) -> None:
    """Mutate each novelty result with a sorted ``dimensions`` list."""
    for item in results:
        if not isinstance(item, dict):
            continue
        quality = str(item.get("quality") or "").strip()
        dims = quality_dims.get(quality)
        item["dimensions"] = sorted(dims) if dims else []
