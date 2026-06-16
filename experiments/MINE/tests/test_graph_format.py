"""Dependency-free checks for the KBExtractor → KGGen graph mapping.

Run with plain Python (no venv needed)::

    python experiments/MINE/tests/test_graph_format.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_format import extraction_results_to_graph_dict


def test_reorders_sop_to_spo():
    # KBExtractor triplet order is (subject, OBJECT, predicate).
    results = [{"sentence": "x", "triplets": [("Fairness", "Bias", "MightMitigate")]}]
    graph = extraction_results_to_graph_dict(results)
    # KGGen relation order is (subject, PREDICATE, object), lowercased.
    assert graph["relations"] == [["fairness", "mightmitigate", "bias"]], graph["relations"]
    assert set(graph["entities"]) == {"fairness", "bias"}
    assert graph["edges"] == ["mightmitigate"]


def test_dedup_and_skips_malformed():
    results = [
        {"triplets": [("A", "B", "Rel"), ("A", "B", "Rel")]},          # duplicate
        {"triplets": [("A", "", "Rel")]},                               # empty object
        {"triplets": [("A", "B")]},                                     # wrong arity
        {"triplets": [("  C  ", "D", "  Has  ")]},                       # whitespace
        {"sentence": "no triplets key"},
    ]
    graph = extraction_results_to_graph_dict(results)
    assert ["a", "rel", "b"] in graph["relations"]
    assert ["c", "has", "d"] in graph["relations"]
    assert len(graph["relations"]) == 2  # dup collapsed, malformed dropped


def test_lowercase_toggle_preserves_case():
    results = [{"triplets": [("Fairness", "Bias", "MightMitigate")]}]
    graph = extraction_results_to_graph_dict(results, lowercase=False)
    assert graph["relations"] == [["Fairness", "MightMitigate", "Bias"]]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"✓ {name}")
            except AssertionError as exc:
                failures += 1
                print(f"✗ {name}: {exc}")
    sys.exit(1 if failures else 0)
