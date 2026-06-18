from __future__ import annotations

from ui.services.dimension_scan import (
    assign_paragraph_dimensions,
    attach_dimensions_to_results,
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
