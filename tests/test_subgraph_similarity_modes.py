from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from kbdebugger.subgraph_similarity import node_entity_filter
from kbdebugger.subgraph_similarity.node_entity_filter import (
    NodeEntitySimilarityFilter,
    extract_candidate_entities,
    extract_candidate_entities_batch,
)
from kbdebugger.subgraph_similarity.similarity_filter import SubgraphSimilarityFilter
from kbdebugger.subgraph_similarity.types import SubgraphSimilarityFilterConfig
from kbdebugger.types import GraphRelation


class StaticEncoder:
    dim = 4

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = [self._vector_for(text) for text in texts]
        return np.asarray(vectors, dtype=np.float32)

    def _vector_for(self, text: str) -> np.ndarray:
        key = " ".join(text.lower().replace("_", " ").split())
        if key in {
            "explainability",
            "model explainability",
            "explainability supports human oversight",
        }:
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if key in {"fairness", "algorithmic fairness", "fairness controls"}:
            return np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        if key in {"latency", "monitoring latency"}:
            return np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def relation(source: str, predicate: str, target: str, sentence: str = "") -> GraphRelation:
    return {
        "source": {"label": source, "id": None, "created_at": None, "last_updated_at": None},
        "target": {"label": target, "id": None, "created_at": None, "last_updated_at": None},
        "edge": {"label": predicate, "properties": {"sentence": sentence}},
    }


def test_pipeline_config_similarity_defaults_and_sentence_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kbdebugger.pipeline.config import PipelineConfig

    for name in (
        "KB_SIMILARITY_MODE",
        "KB_ENTITY_EXTRACTION_MODE",
        "KB_NODE_ENTITY_TOP_K",
        "KB_NODE_ENTITY_MIN_SIMILARITY_THRESHOLD",
        "KB_NODE_ENTITY_MAX_ENTITIES_PER_QUALITY",
        "KB_ENTITY_KEYBERT_MODEL_NAME",
        "KB_ENTITY_KEYBERT_BATCH_SIZE",
        "KB_ENTITY_KEYBERT_NGRAM_MAX",
        "KB_KEYBERT_MODEL_NAME",
        "KB_KEYBERT_BATCH_SIZE",
        "KB_ENCODER_MODEL_NAME",
        "KB_MIN_SIMILARITY_THRESHOLD",
    ):
        monkeypatch.delenv(name, raising=False)

    cfg = PipelineConfig.from_env()

    assert cfg.vector_similarity.similarity_mode == "node_entity"
    assert cfg.vector_similarity.entity_extraction_mode == "keybert"
    assert cfg.vector_similarity.entity_keybert_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.vector_similarity.entity_keybert_batch_size == 32
    assert cfg.vector_similarity.entity_keybert_ngram_max == 3
    assert cfg.vector_similarity.node_entity_top_k == 5
    assert cfg.vector_similarity.node_entity_max_entities_per_quality == 8
    assert cfg.vector_similarity.node_entity_min_similarity_threshold == 0.55

    monkeypatch.setenv("KB_SIMILARITY_MODE", "sentence")
    monkeypatch.setenv("KB_MIN_SIMILARITY_THRESHOLD", "0.61")
    cfg = PipelineConfig.from_env()

    assert cfg.vector_similarity.similarity_mode == "sentence"
    assert cfg.vector_similarity.node_entity_min_similarity_threshold == 0.61


def test_pipeline_config_simple_entity_mode_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from kbdebugger.pipeline.config import PipelineConfig

    monkeypatch.setenv("KB_ENTITY_EXTRACTION_MODE", "simple")

    cfg = PipelineConfig.from_env()

    assert cfg.vector_similarity.entity_extraction_mode == "simple"


def test_pipeline_config_keybert_entity_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    from kbdebugger.pipeline.config import PipelineConfig

    monkeypatch.setenv("KB_ENTITY_KEYBERT_MODEL_NAME", "custom-keybert-model")
    monkeypatch.setenv("KB_ENTITY_KEYBERT_BATCH_SIZE", "9")
    monkeypatch.setenv("KB_ENTITY_KEYBERT_NGRAM_MAX", "4")

    cfg = PipelineConfig.from_env()

    assert cfg.vector_similarity.entity_keybert_model_name == "custom-keybert-model"
    assert cfg.vector_similarity.entity_keybert_batch_size == 9
    assert cfg.vector_similarity.entity_keybert_ngram_max == 4


def test_simple_entity_extraction_is_stable() -> None:
    entities = extract_candidate_entities(
        "The model should provide explainability for high risk decisions.",
        max_entities=8,
    )

    assert not hasattr(node_entity_filter, "_BOUNDARY_WORDS")
    assert "model" in entities
    assert "explainability" in entities
    assert "high risk decisions" in entities
    assert "should" not in entities
    assert len(entities) <= 8


def test_keybert_entity_extraction_batches_and_preserves_phrases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_extract_keyphrases_batch(texts, **kwargs):
        captured["calls"] = captured.get("calls", 0) + 1
        captured["texts"] = list(texts)
        captured["kwargs"] = kwargs
        return [
            ["model explainability", "model explainability", "human oversight"],
            ["risk assessment"],
        ]

    monkeypatch.setattr(node_entity_filter, "extract_keyphrases_batch", fake_extract_keyphrases_batch)

    cfg = SubgraphSimilarityFilterConfig(
        encoder_model_name="static",
        encoder_device=None,
        normalize_embeddings=True,
        quality_to_kg_top_k=3,
        min_similarity_threshold=0.95,
        similarity_mode="node_entity",
        entity_extraction_mode="keybert",
        entity_keybert_model_name="entity-model",
        entity_keybert_batch_size=11,
        entity_keybert_ngram_max=3,
        node_entity_max_entities_per_quality=8,
    )

    groups = extract_candidate_entities_batch(
        ["Quality one.", "Quality two."],
        cfg=cfg,
        max_entities=8,
    )

    assert captured["calls"] == 1
    assert captured["texts"] == ["Quality one.", "Quality two."]
    assert captured["kwargs"] == {
        "embedding_model": "entity-model",
        "batch_size": 11,
        "top_n": 8,
        "ngram_max": 3,
    }
    assert groups == [
        ["model explainability", "human oversight"],
        ["risk assessment"],
    ]


def test_node_entity_index_deduplicates_source_and_target_labels() -> None:
    filt = NodeEntitySimilarityFilter(encoder=StaticEncoder())

    index = filt.build_index(
        [
            relation("Explainability", "AppliesTo", "Fairness"),
            relation("Explainability", "ContributesTo", "Human Oversight"),
        ]
    )

    labels = {payload.label for payload in index.payloads}
    assert labels == {"Explainability", "Fairness", "Human Oversight"}

    explainability_payload = next(payload for payload in index.payloads if payload.label == "Explainability")
    assert len(explainability_payload.relations) == 2


def test_node_entity_keeps_quality_when_any_entity_matches_node() -> None:
    cfg = SubgraphSimilarityFilterConfig(
        encoder_model_name="static",
        encoder_device=None,
        normalize_embeddings=True,
        quality_to_kg_top_k=3,
        min_similarity_threshold=0.95,
        similarity_mode="node_entity",
        entity_extraction_mode="simple",
        node_entity_top_k=3,
        node_entity_min_similarity_threshold=0.95,
        node_entity_max_entities_per_quality=8,
    )
    filt = NodeEntitySimilarityFilter(
        encoder=StaticEncoder(),
        top_k=cfg.node_entity_top_k,
        threshold=float(cfg.node_entity_min_similarity_threshold),
        max_entities_per_quality=cfg.node_entity_max_entities_per_quality,
    )
    index = filt.build_index([relation("Explainability", "Supports", "Human Oversight")])

    kept, dropped = filt.filter_qualities(
        cfg=cfg,
        index=index,
        qualities=["The model should provide explainability while unrelated latency is tracked."],
    )

    assert not dropped
    assert len(kept) == 1
    assert kept[0]["max_score"] >= 0.95
    assert kept[0]["match_mode"] == "node_entity"
    assert "Explainability" in kept[0]["matched_node_labels"]
    assert kept[0]["neighbors"][0]["relation"]["source"]["label"] == "Explainability"
    assert kept[0]["neighbors"][0]["matched_entity"] == "explainability"


def test_node_entity_drops_quality_when_no_entity_matches_node() -> None:
    cfg = SubgraphSimilarityFilterConfig(
        encoder_model_name="static",
        encoder_device=None,
        normalize_embeddings=True,
        quality_to_kg_top_k=3,
        min_similarity_threshold=0.95,
        similarity_mode="node_entity",
        entity_extraction_mode="simple",
        node_entity_top_k=3,
        node_entity_min_similarity_threshold=0.95,
    )
    filt = NodeEntitySimilarityFilter(encoder=StaticEncoder(), top_k=3, threshold=0.95)
    index = filt.build_index([relation("Fairness", "AppliesTo", "Accountability")])

    kept, dropped = filt.filter_qualities(
        cfg=cfg,
        index=index,
        qualities=["Latency monitoring should be documented."],
    )

    assert not kept
    assert len(dropped) == 1
    assert dropped[0]["match_mode"] == "node_entity"
    assert dropped[0]["max_score"] < 0.95


def test_sentence_similarity_mode_still_filters_relation_sentences() -> None:
    cfg = SubgraphSimilarityFilterConfig(
        encoder_model_name="static",
        encoder_device=None,
        normalize_embeddings=True,
        quality_to_kg_top_k=2,
        min_similarity_threshold=0.95,
        similarity_mode="sentence",
    )
    filt = SubgraphSimilarityFilter(encoder=StaticEncoder(), top_k=2, threshold=0.95)
    index = filt.build_index(
        [
            relation(
                "Explainability",
                "Supports",
                "Human Oversight",
                sentence="Explainability supports human oversight",
            )
        ]
    )

    kept, dropped = filt.filter_qualities(
        cfg=cfg,
        index=index,
        qualities=["Explainability supports human oversight"],
    )

    assert not dropped
    assert len(kept) == 1
    assert kept[0]["neighbors"][0]["relation"]["edge"]["label"] == "Supports"
