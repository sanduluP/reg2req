
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from kbdebugger.compat.langchain import Document


MatchType = Literal[
    "exact",
    "synonym",
    "near_paragraph_global",
    "near_paragraph_keywords"
]

@dataclass(frozen=True)
class ParagraphMatch:
    index: int
    paragraph: str
    keywords: List[str]
    match_type: Optional[MatchType]
    matched_terms: List[str]
    cosine_sim_score: Optional[float] = None


@dataclass(frozen=True)
class KeyBERTConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32

    ngram_range: Tuple[int, int] = (1, 1)
    # To extract keyphrases, simply set keyphrase_ngram_range to (1, 2) or higher 
    # depending on the number of words you would like in the resulting keyphrases.

    top_n_keywords_per_paragraph: int = 8 # top_n keywords to be extracted by paragraph

    search_kw_to_paragraph_similarity_threshold: float = 0.45  # Fallback semantic similarity (paragraph vs keyword)
    search_kw_to_keywords_similarity_threshold: float = 0.65



@dataclass(frozen=True)
class KeywordDocMatchResult:
    matched_docs: List[Document]
    unmatched_docs: List[Document]
    # matched: List[ParagraphMatch]
    # unmatched: List[ParagraphMatch]
    synonyms: List[str]
    # DEBUG/TEST FEATURE (safe to remove): per-paragraph match scores so the UI
    # can show which chunk got what score against the keyword.
    scored_chunks: List[Dict[str, Any]] = field(default_factory=list)
