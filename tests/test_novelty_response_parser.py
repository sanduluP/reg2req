from __future__ import annotations

from kbdebugger.novelty.types import NeighborView, NoveltyDecision, QualityNoveltyInput
from kbdebugger.novelty.utils import coerce_batched_novelty_response


def input_for(idx: int) -> QualityNoveltyInput:
    return QualityNoveltyInput(
        quality=f"quality {idx}",
        neighbors=[NeighborView(score=0.9, sentence=f"neighbor {idx}")],
        max_score=0.9,
    )


def test_batched_novelty_accepts_items_wrapper() -> None:
    results = coerce_batched_novelty_response(
        {
            "items": [
                {
                    "id": 0,
                    "decision": "EXISTING",
                    "rationale": "same",
                    "novel_spans": [],
                    "matched_neighbor_sentence": "neighbor 0",
                    "confidence": 0.8,
                }
            ]
        },
        id_to_input={0: input_for(0)},
    )

    assert results[0].decision == NoveltyDecision.EXISTING
    assert results[0].matched_neighbor_sentence == "neighbor 0"


def test_batched_novelty_accepts_numeric_mapping() -> None:
    results = coerce_batched_novelty_response(
        {
            "results": {
                "0": {
                    "decision": "NEW",
                    "rationale": "new",
                    "novel_spans": ["new"],
                    "matched_neighbor_sentence": None,
                    "confidence": 0.7,
                }
            }
        },
        id_to_input={0: input_for(0)},
    )

    assert results[0].decision == NoveltyDecision.NEW


def test_batched_novelty_maps_positional_results_when_ids_are_missing() -> None:
    results = coerce_batched_novelty_response(
        {
            "results": [
                {"decision": "EXISTING", "rationale": "same", "novel_spans": [], "confidence": 0.8},
                {"decision": "NEW", "rationale": "new", "novel_spans": ["new"], "confidence": 0.9},
            ]
        },
        id_to_input={3: input_for(3), 4: input_for(4)},
    )

    assert [result.quality for result in results] == ["quality 3", "quality 4"]
    assert [result.decision for result in results] == [NoveltyDecision.EXISTING, NoveltyDecision.NEW]


def test_batched_novelty_falls_back_for_missing_results() -> None:
    results = coerce_batched_novelty_response(
        {"summary": "I could not format the response."},
        id_to_input={0: input_for(0), 1: input_for(1)},
    )

    assert [result.decision for result in results] == [
        NoveltyDecision.PARTIALLY_NEW,
        NoveltyDecision.PARTIALLY_NEW,
    ]
    assert [result.confidence for result in results] == [0.0, 0.0]
    assert results[0].matched_neighbor_sentence == "neighbor 0"


def test_batched_novelty_missing_confidence_uses_default() -> None:
    results = coerce_batched_novelty_response(
        {"results": [{"id": 0, "decision": "EXISTING", "rationale": "same", "novel_spans": []}]},
        id_to_input={0: input_for(0)},
    )

    assert results[0].confidence == 0.5
