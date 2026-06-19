from __future__ import annotations

from ui.services.extraction_context import (
    ExtractionContext,
    _ExtractionContextStore,
    newly_included_indices,
)


def _ctx(**overrides):
    base = dict(
        job_id="job1",
        all_paragraphs=[],
        dimensions=["Security"],
        cfg=object(),
        para_relevance={
            0: {"score": 0.80, "literal": False},
            1: {"score": 0.50, "literal": False},
            2: {"score": 0.30, "literal": False},
            3: {"score": None, "literal": True},   # literal -> already decomposed
        },
        dims_by_paragraph_index={},
        chunk_scores_by_dimension={},
        kg_relations=[],
        run_threshold=0.60,
        decomposed_indices={0, 3},  # >= run threshold (0) + literal (3)
        existing_qualities=set(),
    )
    base.update(overrides)
    return ExtractionContext(**base)


def test_newly_included_indices_returns_delta_below_floor() -> None:
    ctx = _ctx()
    # Lowering to 0.40 newly includes paragraph 1 (0.50) but not 2 (0.30).
    assert newly_included_indices(ctx, 0.40) == [1]
    # Lowering further to 0.25 includes both 1 and 2.
    assert newly_included_indices(ctx, 0.25) == [1, 2]


def test_newly_included_indices_excludes_already_decomposed_and_literal() -> None:
    ctx = _ctx(decomposed_indices={0, 1, 3})
    # Paragraph 1 already decomposed; literal 3 never appears.
    assert newly_included_indices(ctx, 0.25) == [2]


def test_newly_included_indices_empty_above_floor() -> None:
    ctx = _ctx()
    # At/above what's already extracted, nothing new.
    assert newly_included_indices(ctx, 0.60) == []


def test_context_store_evicts_oldest_beyond_cap() -> None:
    store = _ExtractionContextStore()
    # _MAX_CONTEXTS is 8; insert 10 and confirm the first two are gone.
    for i in range(10):
        store.save(_ctx(job_id=f"job{i}"))
    assert store.get("job0") is None
    assert store.get("job1") is None
    assert store.get("job9") is not None
    assert store.get("job2") is not None
