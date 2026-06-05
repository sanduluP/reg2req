from typing import List, TypedDict, Literal
from typing_extensions import NotRequired

from kbdebugger.types import GraphRelation
from dataclasses import dataclass

# In our codebase, Qualities is typically something like: list[str]
# We keep it explicit here for clarity and strictness.
Quality = str
SimilarityMode = Literal["sentence", "node_entity"]
EntityExtractionMode = Literal["keybert", "simple"]


@dataclass(frozen=True, slots=True)
class SubgraphSimilarityFilterConfig:
    """
    Configuration for the Vector Similarity Filter stage.

    This stage supports two similarity modes:
      - sentence: candidate quality sentence vs KG relation sentence
      - node_entity: quality candidate entities vs KG source/target node labels

    Attributes
    ----------
    encoder_model_name:
        HuggingFace model id for the SentenceTransformer encoder used to embed
        both quality sentences and KG relation sentences.

    encoder_device:
        Device string passed to the encoder (e.g. "cpu", "cuda", "cuda:0").
        If None, the backend chooses automatically.

    normalize_embeddings:
        Whether to L2-normalize embeddings (recommended for cosine similarity).

    quality_to_kg_top_k:
        Number of nearest KG relation sentences to retrieve per quality
        (for context + logging, and to compute max_score).

    min_similarity_threshold:
        Minimum cosine similarity required to keep a quality.

    similarity_mode:
        "sentence" keeps the original sentence-level behavior.
        "node_entity" matches extracted quality entities against KG node labels.

    node_entity_min_similarity_threshold:
        Minimum cosine similarity required in node/entity mode. When omitted,
        it falls back to `min_similarity_threshold`.
    """
    encoder_model_name: str
    encoder_device: str | None # None will let sentence-transformers choose
    normalize_embeddings: bool # Normalizing is recommended for cosine similarity

    quality_to_kg_top_k: int
    min_similarity_threshold: float
    similarity_mode: SimilarityMode = "node_entity"
    entity_extraction_mode: EntityExtractionMode = "keybert"
    node_entity_top_k: int = 5
    node_entity_min_similarity_threshold: float | None = None
    node_entity_max_entities_per_quality: int = 8
    entity_keybert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    entity_keybert_batch_size: int = 32
    entity_keybert_ngram_max: int = 3

    def __post_init__(self) -> None:
        if self.similarity_mode not in {"sentence", "node_entity"}:
            raise ValueError(f"Unsupported similarity_mode={self.similarity_mode!r}")
        if self.entity_extraction_mode not in {"keybert", "simple"}:
            raise ValueError(f"Unsupported entity_extraction_mode={self.entity_extraction_mode!r}")
        object.__setattr__(self, "quality_to_kg_top_k", max(1, int(self.quality_to_kg_top_k)))
        object.__setattr__(self, "node_entity_top_k", max(1, int(self.node_entity_top_k)))
        object.__setattr__(
            self,
            "node_entity_max_entities_per_quality",
            max(1, int(self.node_entity_max_entities_per_quality)),
        )
        object.__setattr__(self, "entity_keybert_batch_size", max(1, int(self.entity_keybert_batch_size)))
        object.__setattr__(self, "entity_keybert_ngram_max", max(1, int(self.entity_keybert_ngram_max)))
        if self.node_entity_min_similarity_threshold is None:
            object.__setattr__(
                self,
                "node_entity_min_similarity_threshold",
                float(self.min_similarity_threshold),
            )

class NeighborHit(TypedDict):
    """
    One nearest-neighbor hit from the KG vector index.

    - relation:
        The KG relation (GraphRelation) whose sentence was similar to the query vector.

    - score:
        Cosine similarity in [0, 1]. Higher means more similar.
    """
    relation: GraphRelation
    score: float
    match_mode: NotRequired[SimilarityMode]
    matched_entity: NotRequired[str]
    matched_node_label: NotRequired[str]


class KeptQuality(TypedDict):
    """
    A quality sentence that passed the similarity threshold.

    - quality:
        The original atomic sentence/quality text.

    - max_score:
        The highest similarity score among the nearest neighbors.
        The score is between the quality (candidate sentence) and its most similar KG relation sentence.

    - neighbors:
        The top-k most similar KG relations. These are kept as context for the
        next stage (e.g., triplet extraction or LLM comparator).
    """
    quality: Quality
    max_score: float
    neighbors: List[NeighborHit]
    match_mode: NotRequired[SimilarityMode]
    matched_entities: NotRequired[List[str]]
    matched_node_labels: NotRequired[List[str]]


class DroppedQuality(TypedDict):
    """
    A quality sentence that failed the similarity threshold.

    - quality:
        The original atomic sentence/quality text.

    - max_score:
        The best similarity score observed. Useful for debugging/tuning threshold.
    """
    quality: Quality
    max_score: float
    match_mode: NotRequired[SimilarityMode]
    extracted_entities: NotRequired[List[str]]
    dropped_reason: NotRequired[str]
