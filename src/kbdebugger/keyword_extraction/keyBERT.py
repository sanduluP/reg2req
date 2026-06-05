from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, List, Optional, Sequence, Tuple

from kbdebugger.types.ui import ProgressCallback

from .logging import save_keybert_result
from .types import (
    KeyBERTConfig,
    MatchType,
    ParagraphMatch,
)


def _normalized_terms(terms: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for term in terms:
        value = " ".join(str(term or "").lower().split())
        if value and value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


@lru_cache(maxsize=512)
def _literal_term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")


def _literal_matches(text: str, terms: Sequence[str]) -> list[str]:
    text_lower = text.lower()
    return [term for term in terms if term and _literal_term_pattern(term).search(text_lower)]


@lru_cache(maxsize=4)
def _get_models(embedding_model: str) -> tuple[Any, Any]:
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer

    sentence_model = SentenceTransformer(embedding_model)
    kw_model = KeyBERT(sentence_model)
    return sentence_model, kw_model


def _extract_keywords_batch(
    kw_model: Any,
    paragraphs: Sequence[str],
    cfg: KeyBERTConfig,
) -> list[list[tuple[str, float]]]:
    if not paragraphs:
        return []

    kwargs = {
        "keyphrase_ngram_range": cfg.ngram_range,
        "stop_words": "english",
        "top_n": cfg.top_n_keywords_per_paragraph,
    }

    try:
        raw = kw_model.extract_keywords(list(paragraphs), **kwargs)
        normalized = _normalize_keyword_batch(raw, expected=len(paragraphs))
        if normalized is not None:
            return normalized
    except Exception:
        pass

    return [
        _normalize_keyword_list(kw_model.extract_keywords(paragraph, **kwargs))
        for paragraph in paragraphs
    ]


def extract_keyphrases_batch(
    texts: Sequence[str],
    *,
    embedding_model: str,
    batch_size: int = 32,
    top_n: int = 8,
    ngram_max: int = 3,
) -> list[list[str]]:
    """
    Public batched KeyBERT keyphrase helper for concept/entity extraction.

    This intentionally reuses the same cached KeyBERT/SentenceTransformer model
    path as paragraph keyword filtering, without adding another NLP dependency.
    """
    if not texts:
        return []

    cfg = KeyBERTConfig(
        embedding_model=embedding_model,
        batch_size=max(1, int(batch_size)),
        ngram_range=(1, max(1, int(ngram_max))),
        top_n_keywords_per_paragraph=max(1, int(top_n)),
    )
    _sentence_model, kw_model = _get_models(cfg.embedding_model)
    keyword_groups = _extract_keywords_batch(kw_model, texts, cfg)

    out: list[list[str]] = []
    for group in keyword_groups[: len(texts)]:
        phrases: list[str] = []
        seen: set[str] = set()
        for phrase, _score in group:
            cleaned = " ".join(str(phrase or "").strip().split())
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                phrases.append(cleaned)
        out.append(phrases)

    while len(out) < len(texts):
        out.append([])

    return out


def _normalize_keyword_batch(raw: Any, *, expected: int) -> Optional[list[list[tuple[str, float]]]]:
    if expected == 1 and _looks_like_keyword_list(raw):
        return [_normalize_keyword_list(raw)]

    if not isinstance(raw, list) or len(raw) != expected:
        return None

    groups: list[list[tuple[str, float]]] = []
    for group in raw:
        if not _looks_like_keyword_list(group):
            return None
        groups.append(_normalize_keyword_list(group))
    return groups


def _looks_like_keyword_list(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return all(_looks_like_keyword_tuple(item) for item in value)


def _looks_like_keyword_tuple(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) >= 2
        and isinstance(value[0], str)
    )


def _normalize_keyword_list(value: Any) -> list[tuple[str, float]]:
    if not isinstance(value, list):
        return []

    normalized: list[tuple[str, float]] = []
    for item in value:
        if not _looks_like_keyword_tuple(item):
            continue
        try:
            score = float(item[1])
        except (TypeError, ValueError):
            score = 0.0
        normalized.append((item[0], score))
    return normalized


def _encode_texts(sentence_model: Any, texts: Sequence[str], cfg: KeyBERTConfig) -> Any:
    return sentence_model.encode(
        list(texts),
        convert_to_tensor=True,
        batch_size=max(1, cfg.batch_size),
    )


def _cosine_scores(query_embedding: Any, candidate_embeddings: Any) -> list[float]:
    from sentence_transformers import util as sbert_util

    scores = sbert_util.cos_sim(query_embedding, candidate_embeddings)[0]
    return [float(score.item()) for score in scores]


def _notify(
    progress: Optional[ProgressCallback],
    current: int,
    total: int,
    message: str,
) -> None:
    if progress:
        progress(current, total, message)


def run_keybert_matching(
    paragraphs: List[str],
    search_keyword: str,
    synonyms: Optional[List[str]] = None,
    config: Optional[KeyBERTConfig] = None,
    progress: Optional[ProgressCallback] = None,
) -> Tuple[
    List[ParagraphMatch],
    List[ParagraphMatch],
    dict,
]:
    """
    Extract keywords from paragraphs using KeyBERT and match them to a target keyword.
    Includes exact match, synonym match, and two levels of semantic similarity fallbacks.
    """
    cfg = config or KeyBERTConfig()
    synonyms = synonyms or []
    synonym_terms = _normalized_terms(synonyms)
    search_keyword_lower = " ".join(search_keyword.lower().split())

    matched: List[ParagraphMatch] = []
    unmatched: List[ParagraphMatch] = []
    total = len(paragraphs)
    records_by_index: dict[int, ParagraphMatch] = {}
    remaining: list[tuple[int, str]] = []

    _notify(
        progress,
        0,
        total,
        f"🔎 Scanning paragraphs for literal keyword matches: \"{search_keyword}\"...",
    )

    for i, paragraph in enumerate(paragraphs):
        matched_terms = _literal_matches(paragraph, [search_keyword_lower])
        if matched_terms:
            records_by_index[i] = ParagraphMatch(
                index=i,
                paragraph=paragraph,
                keywords=[],
                match_type="exact",
                matched_terms=matched_terms,
                cosine_sim_score=None,
            )
            continue

        matched_synonyms = _literal_matches(paragraph, synonym_terms)
        if matched_synonyms:
            records_by_index[i] = ParagraphMatch(
                index=i,
                paragraph=paragraph,
                keywords=[],
                match_type="synonym",
                matched_terms=matched_synonyms,
                cosine_sim_score=None,
            )
            continue

        remaining.append((i, paragraph))

    if not remaining:
        _notify(
            progress,
            total,
            total,
            f"🔎 Keyword scan complete: literal matches found for \"{search_keyword}\".",
        )
    else:
        sentence_model, kw_model = _get_models(cfg.embedding_model)
        search_keyword_embedding = sentence_model.encode(search_keyword, convert_to_tensor=True)

        literal_count = total - len(remaining)
        _notify(
            progress,
            literal_count,
            total,
            f"🔎 Extracting KeyBERT keywords for {len(remaining)} remaining paragraphs...",
        )

        remaining_paragraphs = [paragraph for _idx, paragraph in remaining]
        extracted_keyword_groups = _extract_keywords_batch(kw_model, remaining_paragraphs, cfg)

        keyword_by_index: dict[int, list[str]] = {}
        unresolved_for_semantic: list[int] = []

        for (i, paragraph), extracted_keywords in zip(remaining, extracted_keyword_groups):
            paragraph_keywords = [kw for kw, _probs in extracted_keywords]
            keyword_by_index[i] = paragraph_keywords
            paragraph_keywords_lower = _normalized_terms(paragraph_keywords)

            match_type: Optional[MatchType] = None
            matched_terms: List[str] = []

            matched_synonyms = sorted(set(synonym_terms).intersection(paragraph_keywords_lower))
            if search_keyword_lower in paragraph_keywords_lower:
                match_type = "exact"
                matched_terms = [search_keyword_lower]
            elif matched_synonyms:
                match_type = "synonym"
                matched_terms = matched_synonyms

            if match_type:
                records_by_index[i] = ParagraphMatch(
                    index=i,
                    paragraph=paragraph,
                    keywords=paragraph_keywords,
                    match_type=match_type,
                    matched_terms=matched_terms,
                    cosine_sim_score=None,
                )
            else:
                unresolved_for_semantic.append(i)

        after_keyword_count = total - len(unresolved_for_semantic)
        _notify(
            progress,
            after_keyword_count,
            total,
            f"🔎 Running batched semantic keyword matching for \"{search_keyword}\"...",
        )

        unresolved_global: list[int] = []
        paragraphs_by_index = {i: paragraph for i, paragraph in remaining}
        if unresolved_for_semantic:
            semantic_paragraphs = [paragraphs_by_index[i] for i in unresolved_for_semantic]
            paragraph_embeddings = _encode_texts(sentence_model, semantic_paragraphs, cfg)
            paragraph_scores = _cosine_scores(search_keyword_embedding, paragraph_embeddings)

            for i, similarity_score in zip(unresolved_for_semantic, paragraph_scores):
                if similarity_score >= cfg.search_kw_to_paragraph_similarity_threshold:
                    records_by_index[i] = ParagraphMatch(
                        index=i,
                        paragraph=paragraphs_by_index[i],
                        keywords=keyword_by_index.get(i, []),
                        match_type="near_paragraph_global",
                        matched_terms=[],
                        cosine_sim_score=similarity_score,
                    )
                else:
                    unresolved_global.append(i)

        if unresolved_global:
            unique_keywords = list(
                dict.fromkeys(
                    keyword
                    for i in unresolved_global
                    for keyword in keyword_by_index.get(i, [])
                    if keyword.strip()
                )
            )

            keyword_score_by_text: dict[str, float] = {}
            if unique_keywords:
                keyword_embeddings = _encode_texts(sentence_model, unique_keywords, cfg)
                keyword_scores = _cosine_scores(search_keyword_embedding, keyword_embeddings)
                keyword_score_by_text = dict(zip(unique_keywords, keyword_scores))

            for i in unresolved_global:
                paragraph_keywords = keyword_by_index.get(i, [])
                max_score = max(
                    (keyword_score_by_text.get(keyword, float("-inf")) for keyword in paragraph_keywords),
                    default=float("-inf"),
                )
                if max_score >= cfg.search_kw_to_keywords_similarity_threshold:
                    records_by_index[i] = ParagraphMatch(
                        index=i,
                        paragraph=paragraphs_by_index[i],
                        keywords=paragraph_keywords,
                        match_type="near_paragraph_keywords",
                        matched_terms=[],
                        cosine_sim_score=max_score,
                    )
                else:
                    records_by_index[i] = ParagraphMatch(
                        index=i,
                        paragraph=paragraphs_by_index[i],
                        keywords=paragraph_keywords,
                        match_type=None,
                        matched_terms=[],
                        cosine_sim_score=None,
                    )

        _notify(
            progress,
            total,
            total,
            f"🔎 Keyword scan complete for \"{search_keyword}\".",
        )

    for i, paragraph in enumerate(paragraphs):
        record = records_by_index.get(i)
        if record is None:
            record = ParagraphMatch(
                index=i,
                paragraph=paragraph,
                keywords=[],
                match_type=None,
                matched_terms=[],
                cosine_sim_score=None,
            )

        if record.match_type:
            matched.append(record)
        else:
            unmatched.append(record)

    logging_payload = save_keybert_result(
        matched=matched,
        unmatched=unmatched,
        keyword=search_keyword,
        synonyms=synonyms,
        config=cfg,
    )

    return matched, unmatched, logging_payload
