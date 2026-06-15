from __future__ import annotations

import pytest

from kbdebugger.extraction.utils import coerce_triplets
from kbdebugger.graph.utils import predicate_to_relationship_type


def test_coerce_triplets_keeps_non_standard_predicates() -> None:
    item = {
        "sentence": "AI providers shall document model limitations.",
        "triplets": [
            ["AI provider", "documentation", "Requires"],
            ["AI provider", "model limitations", "MustDocument"],
        ],
    }

    result = coerce_triplets(item, item["sentence"], allowed_predicates=["Requires"])

    assert ("AI provider", "documentation", "Requires") in result["triplets"]
    assert ("AI provider", "model limitations", "MustDocument") in result["triplets"]
    assert result["non_standard_predicates"] == ["MustDocument"]
    assert "skipped_reason" not in result


def test_coerce_triplets_captures_modality() -> None:
    item = {
        "sentence": "The provider shall document limitations.",
        "triplets": [["provider", "limitations", "Requires"]],
        "modality": "mandatory",
    }

    result = coerce_triplets(item, item["sentence"], allowed_predicates=["Requires"])

    assert result["modality"] == "MANDATORY"


def test_coerce_triplets_ignores_invalid_modality() -> None:
    item = {
        "sentence": "Plain statement.",
        "triplets": [["a", "b", "Requires"]],
        "modality": "SOMETIMES",
    }

    result = coerce_triplets(item, item["sentence"], allowed_predicates=["Requires"])

    assert "modality" not in result


def test_coerce_triplets_skipped_reason_only_when_no_triplets() -> None:
    item = {"sentence": "Nothing extractable here.", "triplets": []}

    result = coerce_triplets(item, item["sentence"], allowed_predicates=["Requires"])

    assert result["triplets"] == []
    assert result["skipped_reason"] == "No allowed relationship type fit this quality."


def test_predicate_to_relationship_type_standard_predicate() -> None:
    assert predicate_to_relationship_type("IsSubclassOf") == "is_subclass_of"
    assert predicate_to_relationship_type("HasParameter") == "has_parameter"


def test_predicate_to_relationship_type_sanitizes_non_standard() -> None:
    assert predicate_to_relationship_type("MustDocument") == "must_document"
    assert predicate_to_relationship_type("Shall Not Use!") == "shall_not_use"


def test_predicate_to_relationship_type_rejects_unsafe() -> None:
    with pytest.raises(ValueError):
        predicate_to_relationship_type("123")
    with pytest.raises(ValueError):
        predicate_to_relationship_type("!!!")
