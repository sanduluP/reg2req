from __future__ import annotations

from ui.services.dimension_scan import (
    assign_paragraph_dimensions,
    attach_chunk_relevance_to_results,
    attach_dimensions_to_results,
    build_paragraph_relevance,
    build_quality_dimensions,
    dedup_qualities,
)


class _Para:
    """Minimal stand-in for a LangChain Document (identity matters)."""

    def __init__(self, text: str) -> None:
        self.page_content = text


def test_assign_paragraph_dimensions_unions_and_dedups() -> None:
    p0, p1, p2 = _Para("a"), _Para("b"), _Para("c")
    all_paragraphs = [p0, p1, p2]

    # p1 matches two dimensions, p0 one, p2 none.
    matched = {
        "Fairness": [p0, p1],
        "Transparency": [p1],
    }

    indices, dims_by_index = assign_paragraph_dimensions(all_paragraphs, matched)

    assert indices == [0, 1]            # p2 excluded (matched nothing)
    assert dims_by_index[0] == {"Fairness"}
    assert dims_by_index[1] == {"Fairness", "Transparency"}


def test_build_quality_dimensions_maps_via_source_index() -> None:
    # decompose received two paragraphs (positions 0 and 1)
    dims_by_decompose_index = {0: {"Fairness"}, 1: {"Fairness", "Transparency"}}
    quality_sources = [
        {"quality": "Bias must be mitigated.", "source_doc_index": 0},
        {"quality": "The system shall be transparent.", "source_doc_index": 1},
        {"quality": "Bias must be mitigated.", "source_doc_index": 1},  # same text, other para
    ]

    qd = build_quality_dimensions(quality_sources, dims_by_decompose_index)

    # The duplicate quality text unions dimensions from both source paragraphs.
    assert qd["Bias must be mitigated."] == {"Fairness", "Transparency"}
    assert qd["The system shall be transparent."] == {"Fairness", "Transparency"}


def test_dedup_qualities_preserves_order() -> None:
    out = dedup_qualities(["a", "b", "a", " b ", "c"])
    # "a" and "b" dedup on trimmed text; first-seen casing/whitespace kept.
    assert out == ["a", "b", "c"]


def test_attach_dimensions_to_results_sets_sorted_list() -> None:
    results = [
        {"quality": "Bias must be mitigated.", "decision": "NEW"},
        {"quality": "Unknown quality.", "decision": "NEW"},
    ]
    qd = {"Bias must be mitigated.": {"Transparency", "Fairness"}}

    attach_dimensions_to_results(results, qd)

    assert results[0]["dimensions"] == ["Fairness", "Transparency"]  # sorted
    assert results[1]["dimensions"] == []


def test_build_paragraph_relevance_keeps_max_score_and_any_literal() -> None:
    chunk_scores = {
        "Fairness": [
            {"index": 0, "match_type": "near_paragraph_global", "score": 0.40},
            {"index": 1, "match_type": "exact", "score": None},  # literal
        ],
        "Transparency": [
            {"index": 0, "match_type": "near_paragraph_global", "score": 0.62},  # higher
        ],
    }

    rel = build_paragraph_relevance(chunk_scores)

    # Paragraph 0: max score across dims; not literal.
    assert rel[0] == {"score": 0.62, "literal": False}
    # Paragraph 1: literal match -> always kept, score unknown.
    assert rel[1] == {"score": None, "literal": True}


def test_attach_chunk_relevance_maps_decompose_pos_to_paragraph_score() -> None:
    # decompose positions 0,1 map back to original paragraphs 5, 9
    matched_indices = [5, 9]
    chunk_scores = {
        "Fairness": [
            {"index": 5, "match_type": "near_paragraph_global", "score": 0.71},
            {"index": 9, "match_type": "synonym", "score": None},  # literal
        ],
    }
    results = [
        {"quality": "q1", "source_context": {"source_doc_index": 0}},
        {"quality": "q2", "source_context": {"source_doc_index": 1}},
        {"quality": "q3", "source_context": {"source_doc_index": 2}},  # out of range
        {"quality": "q4"},  # no source context
    ]

    attach_chunk_relevance_to_results(
        results,
        matched_indices=matched_indices,
        chunk_scores_by_dimension=chunk_scores,
    )

    assert results[0]["source_chunk_score"] == 0.71
    assert results[0]["source_chunk_literal"] is False
    assert results[1]["source_chunk_score"] is None
    assert results[1]["source_chunk_literal"] is True
    assert "source_chunk_score" not in results[2]
    assert "source_chunk_score" not in results[3]


def test_attach_chunk_relevance_noop_without_scores() -> None:
    results = [{"quality": "q1", "source_context": {"source_doc_index": 0}}]
    attach_chunk_relevance_to_results(
        results, matched_indices=[3], chunk_scores_by_dimension={}
    )
    assert "source_chunk_score" not in results[0]
