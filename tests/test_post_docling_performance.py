from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict | None = None


def test_pipeline_config_post_docling_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from kbdebugger.pipeline.config import PipelineConfig

    for name in (
        "KB_KEYBERT_MODEL_NAME",
        "KB_KEYBERT_BATCH_SIZE",
        "KB_KEYBERT_TOP_N",
        "KB_KEYWORD_SYNONYMS_ENABLED",
        "KB_DECOMPOSER_PARALLEL",
        "KB_DECOMPOSER_MAX_WORKERS",
        "KB_DECOMPOSER_BATCH_SIZE",
        "KB_NOVELTY_BATCH_SIZE",
        "KB_NOVELTY_PARALLEL",
        "KB_NOVELTY_MAX_WORKERS",
        "KB_TRIPLET_EXTRACTION_PARALLEL",
        "KB_TRIPLET_EXTRACTION_MAX_WORKERS",
    ):
        monkeypatch.delenv(name, raising=False)

    cfg = PipelineConfig.from_env()

    assert cfg.keybert.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.keybert.batch_size == 32
    assert cfg.keybert.top_n_keywords_per_paragraph == 8
    assert cfg.keyword_synonyms_enabled is True
    assert cfg.decomposer_parallel is True
    assert cfg.decomposer_max_workers == 2
    assert cfg.decomposer_batch_size == 5
    assert cfg.novelty_batch_size == 5
    assert cfg.novelty_parallel is True
    assert cfg.novelty_max_workers == 2
    assert cfg.triplet_extraction_parallel is True
    assert cfg.triplet_extraction_max_workers == 2


def test_synonyms_disabled_skips_llm_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from kbdebugger.keyword_extraction import api
    from kbdebugger.keyword_extraction.types import KeyBERTConfig

    def fail_synonyms(_keyword: str) -> list[str]:
        raise AssertionError("synonym LLM should not be called")

    def fake_run_keybert_matching(**kwargs):
        assert kwargs["synonyms"] == []
        return [], [], {"generated_synonyms": []}

    monkeypatch.setattr(api, "generate_synonyms_for_keyword", fail_synonyms)
    monkeypatch.setattr(api, "run_keybert_matching", fake_run_keybert_matching)

    result, log_payload = api.filter_paragraphs_by_keyword(
        paragraphs=[FakeDocument("privacy controls")],
        search_keyword="privacy",
        config=KeyBERTConfig(),
        synonyms_enabled=False,
    )

    assert result.synonyms == []
    assert log_payload == {"generated_synonyms": []}


def test_synonym_generation_falls_back_when_llm_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from kbdebugger.keyword_extraction import keyword_synonyms

    keyword_synonyms._generate_synonyms_for_normalized_keyword.cache_clear()

    def fail_respond(*_args, **_kwargs):
        raise RuntimeError("LLM response did not contain final assistant content")

    monkeypatch.setattr(keyword_synonyms, "respond", fail_respond)

    synonyms = keyword_synonyms.generate_synonyms_for_keyword("Explainability")

    assert "interpretability" in synonyms
    assert "understandability" in synonyms
    assert len(synonyms) <= 10


def test_literal_keyword_match_does_not_load_keybert(monkeypatch: pytest.MonkeyPatch) -> None:
    from kbdebugger.keyword_extraction import keyBERT
    from kbdebugger.keyword_extraction.types import KeyBERTConfig

    def fail_model_load(_embedding_model: str):
        raise AssertionError("model should not load for literal keyword hits")

    monkeypatch.setattr(keyBERT, "_get_models", fail_model_load)
    monkeypatch.setattr(
        keyBERT,
        "save_keybert_result",
        lambda **kwargs: {"matched": kwargs["matched"], "unmatched": kwargs["unmatched"]},
    )

    matched, unmatched, _log_payload = keyBERT.run_keybert_matching(
        paragraphs=["Fairness requirements are documented."],
        search_keyword="fairness",
        synonyms=[],
        config=KeyBERTConfig(),
    )

    assert len(matched) == 1
    assert not unmatched
    assert matched[0].match_type == "exact"
    assert matched[0].matched_terms == ["fairness"]


def test_batched_semantic_matching_preserves_match_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kbdebugger.keyword_extraction import keyBERT
    from kbdebugger.keyword_extraction.types import KeyBERTConfig

    class FakeSentenceModel:
        def encode(self, texts, **_kwargs):
            return texts

    class FakeKeyBERT:
        def __init__(self) -> None:
            self.calls = []

        def extract_keywords(self, docs, **_kwargs):
            self.calls.append(docs)
            assert isinstance(docs, list)
            return [
                [("controls", 0.9)],
                [("data minimization", 0.8)],
            ]

    fake_kw_model = FakeKeyBERT()

    def fake_cosine_scores(query_embedding, candidate_embeddings):
        score_map = {
            "Controls reduce disclosure risk.": 0.50,
            "The system minimizes collection of personal records.": 0.20,
            "data minimization": 0.70,
        }
        return [score_map.get(candidate, 0.10) for candidate in candidate_embeddings]

    monkeypatch.setattr(keyBERT, "_get_models", lambda _model: (FakeSentenceModel(), fake_kw_model))
    monkeypatch.setattr(keyBERT, "_cosine_scores", fake_cosine_scores)
    monkeypatch.setattr(
        keyBERT,
        "save_keybert_result",
        lambda **kwargs: {"matched": kwargs["matched"], "unmatched": kwargs["unmatched"]},
    )

    matched, unmatched, _log_payload = keyBERT.run_keybert_matching(
        paragraphs=[
            "Controls reduce disclosure risk.",
            "The system minimizes collection of personal records.",
        ],
        search_keyword="privacy",
        synonyms=[],
        config=KeyBERTConfig(batch_size=16),
    )

    assert not unmatched
    assert [record.match_type for record in matched] == [
        "near_paragraph_global",
        "near_paragraph_keywords",
    ]
    assert fake_kw_model.calls == [
        [
            "Controls reduce disclosure risk.",
            "The system minimizes collection of personal records.",
        ]
    ]


def test_decomposer_wrapper_passes_parallel_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kbdebugger.extraction import api

    captured = {}

    def fake_decompose_documents(**kwargs):
        captured.update(kwargs)
        return ["quality"], {"ok": True}

    monkeypatch.setattr(api, "decompose_documents", fake_decompose_documents)

    qualities, log_payload = api.decompose_paragraphs_to_qualities(
        paragraphs=[FakeDocument("source paragraph")],
        batch_size=7,
        parallel=True,
        max_workers=3,
    )

    assert qualities == ["quality"]
    assert log_payload == {"ok": True}
    assert captured["batch_size"] == 7
    assert captured["parallel"] is True
    assert captured["max_workers"] == 3
