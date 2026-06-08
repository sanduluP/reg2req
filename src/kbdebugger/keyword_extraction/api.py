from __future__ import annotations

from typing import List, Optional, Sequence

from kbdebugger.compat.langchain import Document
from kbdebugger.types.ui import ProgressCallback

from .keyword_synonyms import generate_synonyms_for_keyword
from .keyBERT import run_keybert_matching
from .types import KeyBERTConfig, KeywordDocMatchResult


def filter_paragraphs_by_keyword(
    *,
    paragraphs: Sequence[Document],
    search_keyword: str,
    max_synonyms: int = 10,
    config: Optional[KeyBERTConfig] = None,
    synonyms_enabled: bool = True,
    synonym_cache_enabled: bool | None = None,
    synonym_cache_path: str | None = None,
    synonym_defaults_path: str | None = None,
    synonym_cache_write: bool | None = None,
    progress: Optional[ProgressCallback] = None,
) -> tuple[
        KeywordDocMatchResult,
        dict # logging payload
    ]:
    """
    Public API: Generate synonyms then run KeyBERT keyword extraction + matching to filter paragraphs
    by their relevance to the user-chosen keyword.


    Parameters
    ----------
    paragraphs:
        Input documents from Docling (or any upstream).
    search_keyword:
        The keyword used to find relevant paragraphs.
    max_synonyms:
        Safety cap for synonym list size (if your generator supports it).
    progress:
        Callback function to update progress


    Notes:
    -----
    - KeyBERT operates on strings, but we keep Documents as the canonical objects.
    - We do NOT filter out empty texts (if any) here to keep indices stable:
        i.e., ParagraphMatch.index == docs[index]
    
    Returns
    -------
    KeywordMatchResult
        Matched/unmatched paragraphs plus the synonyms that were used.
    """
    cfg = config or KeyBERTConfig()
    synonyms = (
        generate_synonyms_for_keyword(
            search_keyword,
            cache_enabled=synonym_cache_enabled,
            cache_path=synonym_cache_path,
            defaults_path=synonym_defaults_path,
            cache_write=synonym_cache_write,
        )
        if synonyms_enabled
        else []
    )
    if max_synonyms and len(synonyms) > max_synonyms:
        synonyms = synonyms[:max_synonyms]

    # Extract paragraph strings from Document objects.
    texts = [paragraph_doc.page_content for paragraph_doc in paragraphs]  # guaranteed non-empty

    matched, unmatched, log_payload = run_keybert_matching(
        paragraphs=texts,
        search_keyword=search_keyword,
        synonyms=synonyms,
        config=cfg,
        progress=progress
    )
    # ⚠️ Notice that we ignore the matched/unmatched ParagraphMatch objects here since they contain 
    # text and keyword info that would be redundant with the Document objects.
    # Anyways, they were all logged in a JSON file for inspection/debugging.
    # But for downstream stages, we only care about the Document objects that correspond to matched vs unmatched paragraphs. 
    # We can always add them back later if needed.

    # Map back to Document objects for the final result.
    matched_docs = [paragraphs[m.index] for m in matched]
    unmatched_docs = [paragraphs[u.index] for u in unmatched]

    
    return KeywordDocMatchResult(
        matched_docs=matched_docs,
        unmatched_docs=unmatched_docs,
        synonyms=synonyms,
        # matched=matched,
        # unmatched=unmatched,
    ), log_payload
