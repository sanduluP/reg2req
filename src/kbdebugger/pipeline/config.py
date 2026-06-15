from __future__ import annotations

import os
from dataclasses import dataclass
from typing import cast

from kbdebugger.extraction.types import SourceKind
from kbdebugger.keyword_extraction.types import KeyBERTConfig
from kbdebugger.subgraph_similarity.types import (
    EntityExtractionMode,
    SimilarityMode,
    SubgraphSimilarityFilterConfig,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """
    Central runtime configuration for the end-to-end KBDebugger pipeline.

    This config is intentionally:
    - **explicit** (field names describe *exactly* what they control),
    - **environment-driven** (easy to run experiments without code edits).

    Stages controlled by this config
    --------------------------------
    1) KG subgraph retrieval (Neo4j):
        - which keyword to retrieve around
        - how many relations to fetch per retrieval pattern

    2) Corpus ingestion + decomposition:
        - which source kind to read (TEXT / PDF_SENTENCES / PDF_CHUNKS)
        - which path is used for that source kind

    3) Vector similarity filter:
        - which SentenceTransformer encoder is used
        - similarity threshold and top-k neighbor retrieval per quality

    4) Novelty comparator (LLM):
        - decoding/length parameters for the novelty decision model

    5) Triplet extraction (LLM):
        - batch size for triplet extraction calls

    Environment variables
    ---------------------
    1️⃣ KG retrieval:
        KB_RETRIEVAL_KEYWORD:
            Keyword used to retrieve a KG subgraph from Neo4j.
            Default: "requirement"

        KB_LIMIT_PER_PATTERN:
            Number of relations retrieved per retriever pattern.
            Default: 50

    2️⃣ Corpus:
        KB_SOURCE_KIND:
            One of: "TEXT", "PDF_SENTENCES", "PDF_CHUNKS"
            Default: "TEXT"

        KB_TEXT_PATH:
            Path to corpus text file (when KB_SOURCE_KIND == "TEXT")
            Default: "data/DSA/DSA_knowledge.txt"

        KB_PDF_PATH:
            Path to corpus PDF file (when KB_SOURCE_KIND starts with "PDF_")
            Default: "data/SDS/InstructCIR.pdf"

        KB_DROP_REFERENCE_SECTION:
            Drop a detected research-paper References/Bibliography section after Docling.
            Default: true

        KB_REFERENCE_SECTION_FILTER_MODE:
            Reference-section filter mode. Supported: "conservative".
            Default: "conservative"

    3️⃣ Vector similarity filtering:
        KB_ENCODER_MODEL_NAME:
            🤗 HuggingFace model id for the SentenceTransformer encoder used to embed
            candidate qualities/entities and KG relation/node texts.
            Default: "sentence-transformers/all-MiniLM-L6-v2"

        KB_ENCODER_DEVICE:
            Optional device string (e.g., "cpu", "cuda", "cuda:0").
            Empty means "let the backend decide".
            Default: "" (auto)

        KB_NORMALIZE_EMBEDDINGS:
            Whether embeddings are L2-normalized (recommended for cosine similarity).
            Default: true

        KB_QUALITY_TO_KG_TOP_K:
            Number of nearest KG relations to retrieve per candidate quality.
            This determines:
              - how many neighbors are kept for context/logging
              - how max_score is computed (best neighbor similarity)
            Default: 5

        KB_MIN_SIMILARITY_THRESHOLD:
            Minimum cosine similarity required to keep a quality.
            Default: 0.55

        KB_SIMILARITY_MODE:
            One of: "node_entity", "sentence".
            Default: "node_entity"

        KB_NODE_ENTITY_MIN_SIMILARITY_THRESHOLD:
            Minimum cosine similarity required for quality entity ↔ KG node label
            matching. Empty means reuse KB_MIN_SIMILARITY_THRESHOLD.

        KB_NODE_ENTITY_TOP_K:
            Number of nearest KG nodes to retrieve per extracted quality entity.
            Default: 5

        KB_NODE_ENTITY_MAX_ENTITIES_PER_QUALITY:
            Maximum lightweight candidate entities extracted per quality.
            Default: 8

        KB_ENTITY_EXTRACTION_MODE:
            Entity extraction strategy. Supported: "keybert", "simple".
            Default: "keybert"

        KB_ENTITY_KEYBERT_MODEL_NAME:
            Model used by KeyBERT for quality entity/keyphrase extraction.
            Empty means use KB_KEYBERT_MODEL_NAME, then KB_ENCODER_MODEL_NAME.

        KB_ENTITY_KEYBERT_BATCH_SIZE:
            Batch size for KeyBERT quality entity/keyphrase extraction.
            Empty means use KB_KEYBERT_BATCH_SIZE.

        KB_ENTITY_KEYBERT_NGRAM_MAX:
            Maximum n-gram length for KeyBERT entity/keyphrase extraction.
            Default: 3

        KB_KEYWORD_SYNONYMS_ENABLED:
            Full off switch for keyword synonym expansion.
            Default: true

        KB_KEYWORD_SYNONYM_CACHE_ENABLED:
            Read persistent runtime/cache synonyms before calling the LLM.
            Default: true

        KB_KEYWORD_SYNONYM_CACHE_PATH:
            Runtime JSON cache for LLM-generated synonyms.
            Default: runtime/keyword_synonyms_cache.json

        KB_KEYWORD_SYNONYM_DEFAULTS_PATH:
            Versioned curated synonym defaults.
            Default: data/keyword_synonyms.json

        KB_KEYWORD_SYNONYM_CACHE_WRITE:
            Write successful LLM synonym generations to the runtime cache.
            Default: true

    4️⃣ Novelty comparator (LLM):
        KB_NOVELTY_LLM_MAX_TOKENS:
            Max tokens for novelty decision response.
            Default: 700

        KB_NOVELTY_LLM_TEMPERATURE:
            Temperature for novelty decision model.
            Default: 0.0

        KB_NOVELTY_BATCH_SIZE:
            Number of kept qualities to classify in one novelty LLM call.
            Default: 5

        KB_NOVELTY_PARALLEL:
            Whether novelty batches may run concurrently.
            Default: true

        KB_NOVELTY_MAX_WORKERS:
            Maximum novelty LLM worker threads when parallel mode is enabled.
            Default: 2

    5️⃣ Triplet extraction:
        KB_TRIPLET_EXTRACTION_BATCH_SIZE:
            How many qualifying quality sentences to send in one triplet extraction call.
            Default: 5

        KB_TRIPLET_EXTRACTION_PARALLEL:
            Whether triplet extraction batches may run concurrently.
            Default: true

        KB_TRIPLET_EXTRACTION_MAX_WORKERS:
            Maximum triplet extraction LLM worker threads when parallel mode is enabled.
            Default: 2

        KB_SCHEMA_GROUNDING_ENABLED:
            Use the current Neo4j graph as standard schema grounding during triplet extraction.
            Default: true
    """
   # ----------------------------
    # KG retrieval
    # ----------------------------
    kg_retrieval_keyword: str
    kg_limit_per_pattern: int

    # ----------------------------
    # Corpus selection
    # ----------------------------
    source_kind: SourceKind
    corpus_path: str

    # 🦆 Docling
    docling_enable_OCR: bool
    docling_enable_table_recognition: bool
    drop_reference_section: bool
    reference_section_filter_mode: str

    # ----------------------------
    # Vector similarity filter
    # ----------------------------
    vector_similarity: SubgraphSimilarityFilterConfig

    # ----------------------------
    # KeyBERT keyword filter
    # ----------------------------
    keybert: KeyBERTConfig
    keyword_synonyms_enabled: bool
    keyword_synonym_cache_enabled: bool
    keyword_synonym_cache_path: str
    keyword_synonym_defaults_path: str
    keyword_synonym_cache_write: bool

    # ----------------------------
    # LLM decomposer
    # ----------------------------
    decomposer_parallel: bool
    decomposer_max_workers: int
    decomposer_batch_size: int

    # ----------------------------
    # Novelty comparator (LLM)
    # ----------------------------
    novelty_llm_max_tokens: int
    novelty_llm_temperature: float
    novelty_batch_size: int
    novelty_parallel: bool
    novelty_max_workers: int

    # ----------------------------
    # Triplet extraction
    # ----------------------------
    triplet_extraction_batch_size: int
    triplet_extraction_parallel: bool
    triplet_extraction_max_workers: int
    schema_grounding_enabled: bool




    @classmethod
    def from_env(cls) -> PipelineConfig:
        """
        Construct a PipelineConfig from environment variables.

        This method is the **single authoritative place** that defines:
        - supported environment variables
        - defaults
        - validation and normalization rules

        Validation / normalization rules
        --------------------------------
        - KB_SOURCE_KIND is validated strictly.
        - kg_limit_per_pattern and triplet_extraction_batch_size are clamped to >= 1.
        - quality_to_kg_top_k is clamped to >= 1 (inside SubgraphSimilarityFilterConfig).
        - Empty KB_ENCODER_DEVICE is treated as None (auto device).

        Returns
        -------
        PipelineConfig
            Fully populated pipeline configuration.

        Raises
        ------
        ValueError
            If KB_SOURCE_KIND is not one of the supported enum values.
        """
        # ---------- KG retrieval ----------
        kg_retrieval_keyword = os.getenv("KB_RETRIEVAL_KEYWORD", "requirement").strip()
        kg_limit_per_pattern = int(os.getenv("KB_LIMIT_PER_PATTERN", "50").strip())
        kg_limit_per_pattern = max(1, kg_limit_per_pattern)

        # ---------- Corpus ----------
        source_raw = os.getenv("KB_SOURCE_KIND", "TEXT").upper().strip()
        if source_raw not in {"TEXT", "PDF_SENTENCES", "PDF_CHUNKS"}:
            raise ValueError(f"Invalid KB_SOURCE_KIND={source_raw!r}")
        source_kind = cast(SourceKind, source_raw)

        text_path = os.getenv("KB_TEXT_PATH", "data/DSA/DSA_knowledge.txt").strip()
        pdf_path = os.getenv("KB_PDF_PATH", "data/SDS/InstructCIR.pdf").strip()

        corpus_path = text_path if source_kind == SourceKind.TEXT else pdf_path

        docling_enable_OCR = _env_bool("DOCLING_ENABLE_OCR", False)
        docling_enable_table_recognition = _env_bool("DOCLING_ENABLE_TABLE_RECOGNITION", False)
        drop_reference_section = _env_bool("KB_DROP_REFERENCE_SECTION", True)
        reference_section_filter_mode = os.getenv(
            "KB_REFERENCE_SECTION_FILTER_MODE",
            "conservative",
        ).strip().lower()
        if reference_section_filter_mode not in {"conservative"}:
            raise ValueError(
                f"Invalid KB_REFERENCE_SECTION_FILTER_MODE={reference_section_filter_mode!r}"
            )

        # ---------- KeyBERT keyword filter ----------
        keybert_model_name = os.getenv(
            "KB_KEYBERT_MODEL_NAME",
            "sentence-transformers/all-MiniLM-L6-v2",
        ).strip()
        keybert_batch_size = int(os.getenv("KB_KEYBERT_BATCH_SIZE", "32").strip())
        keybert_batch_size = max(1, keybert_batch_size)
        keybert_top_n = int(os.getenv("KB_KEYBERT_TOP_N", "8").strip())
        keybert_top_n = max(1, keybert_top_n)
        keyword_synonyms_enabled = _env_bool("KB_KEYWORD_SYNONYMS_ENABLED", True)
        keyword_synonym_cache_enabled = _env_bool("KB_KEYWORD_SYNONYM_CACHE_ENABLED", True)
        keyword_synonym_cache_path = os.getenv(
            "KB_KEYWORD_SYNONYM_CACHE_PATH",
            "runtime/keyword_synonyms_cache.json",
        ).strip()
        keyword_synonym_defaults_path = os.getenv(
            "KB_KEYWORD_SYNONYM_DEFAULTS_PATH",
            "data/keyword_synonyms.json",
        ).strip()
        keyword_synonym_cache_write = _env_bool("KB_KEYWORD_SYNONYM_CACHE_WRITE", True)

        keybert = KeyBERTConfig(
            embedding_model=keybert_model_name,
            batch_size=keybert_batch_size,
            top_n_keywords_per_paragraph=keybert_top_n,
        )

        # ---------- LLM decomposer ----------
        decomposer_parallel = _env_bool("KB_DECOMPOSER_PARALLEL", True)
        decomposer_max_workers = int(os.getenv("KB_DECOMPOSER_MAX_WORKERS", "2").strip())
        decomposer_max_workers = max(1, decomposer_max_workers)
        decomposer_batch_size = int(os.getenv("KB_DECOMPOSER_BATCH_SIZE", "5").strip())
        decomposer_batch_size = max(1, decomposer_batch_size)

        # ---------- Vector similarity ----------
        encoder_model_name = os.getenv(
            "KB_ENCODER_MODEL_NAME",
            "sentence-transformers/all-MiniLM-L6-v2",
        ).strip()

        encoder_device_raw = os.getenv("KB_ENCODER_DEVICE", "").strip()
        encoder_device = encoder_device_raw or None

        normalize_embeddings = os.getenv("KB_NORMALIZE_EMBEDDINGS", "true").strip().lower() in {
            "1",
            "true",
            "yes",
        }

        quality_to_kg_top_k = int(os.getenv("KB_QUALITY_TO_KG_TOP_K", "5").strip())
        quality_to_kg_top_k = max(1, quality_to_kg_top_k)

        min_similarity_threshold = float(os.getenv("KB_MIN_SIMILARITY_THRESHOLD", "0.55").strip())

        similarity_mode_raw = os.getenv("KB_SIMILARITY_MODE", "node_entity").strip().lower()
        if similarity_mode_raw not in {"sentence", "node_entity"}:
            raise ValueError(f"Invalid KB_SIMILARITY_MODE={similarity_mode_raw!r}")
        similarity_mode = cast(SimilarityMode, similarity_mode_raw)

        entity_extraction_mode_raw = os.getenv("KB_ENTITY_EXTRACTION_MODE", "keybert").strip().lower()
        if entity_extraction_mode_raw not in {"keybert", "simple"}:
            raise ValueError(f"Invalid KB_ENTITY_EXTRACTION_MODE={entity_extraction_mode_raw!r}")
        entity_extraction_mode = cast(EntityExtractionMode, entity_extraction_mode_raw)

        entity_keybert_model_name = (
            os.getenv("KB_ENTITY_KEYBERT_MODEL_NAME", "").strip()
            or keybert_model_name
            or encoder_model_name
        )
        entity_keybert_batch_size_raw = os.getenv("KB_ENTITY_KEYBERT_BATCH_SIZE", "").strip()
        entity_keybert_batch_size = (
            int(entity_keybert_batch_size_raw)
            if entity_keybert_batch_size_raw
            else keybert_batch_size
        )
        entity_keybert_batch_size = max(1, entity_keybert_batch_size)
        entity_keybert_ngram_max = int(os.getenv("KB_ENTITY_KEYBERT_NGRAM_MAX", "3").strip())
        entity_keybert_ngram_max = max(1, entity_keybert_ngram_max)

        node_entity_top_k = int(os.getenv("KB_NODE_ENTITY_TOP_K", "5").strip())
        node_entity_top_k = max(1, node_entity_top_k)

        node_entity_threshold_raw = os.getenv("KB_NODE_ENTITY_MIN_SIMILARITY_THRESHOLD", "").strip()
        node_entity_min_similarity_threshold = (
            float(node_entity_threshold_raw)
            if node_entity_threshold_raw
            else min_similarity_threshold
        )

        node_entity_max_entities_per_quality = int(
            os.getenv("KB_NODE_ENTITY_MAX_ENTITIES_PER_QUALITY", "8").strip()
        )
        node_entity_max_entities_per_quality = max(1, node_entity_max_entities_per_quality)

        vector_similarity = SubgraphSimilarityFilterConfig(
            encoder_model_name=encoder_model_name,
            encoder_device=encoder_device,
            normalize_embeddings=normalize_embeddings,
            quality_to_kg_top_k=quality_to_kg_top_k,
            min_similarity_threshold=min_similarity_threshold,
            similarity_mode=similarity_mode,
            entity_extraction_mode=entity_extraction_mode,
            node_entity_top_k=node_entity_top_k,
            node_entity_min_similarity_threshold=node_entity_min_similarity_threshold,
            node_entity_max_entities_per_quality=node_entity_max_entities_per_quality,
            entity_keybert_model_name=entity_keybert_model_name,
            entity_keybert_batch_size=entity_keybert_batch_size,
            entity_keybert_ngram_max=entity_keybert_ngram_max,
        )

        # ---------- Novelty comparator ----------
        novelty_llm_max_tokens = int(os.getenv("KB_NOVELTY_LLM_MAX_TOKENS", "700").strip())
        novelty_llm_temperature = float(os.getenv("KB_NOVELTY_LLM_TEMPERATURE", "0.0").strip())
        novelty_batch_size = int(os.getenv("KB_NOVELTY_BATCH_SIZE", "5").strip())
        novelty_batch_size = max(1, novelty_batch_size)
        novelty_parallel = _env_bool("KB_NOVELTY_PARALLEL", True)
        novelty_max_workers = int(os.getenv("KB_NOVELTY_MAX_WORKERS", "2").strip())
        novelty_max_workers = max(1, novelty_max_workers)


        # ---------- Triplet extraction ----------
        triplet_extraction_batch_size = int(os.getenv("KB_TRIPLET_EXTRACTION_BATCH_SIZE", "5").strip())
        triplet_extraction_batch_size = max(1, triplet_extraction_batch_size)
        triplet_extraction_parallel = _env_bool("KB_TRIPLET_EXTRACTION_PARALLEL", True)
        triplet_extraction_max_workers = int(os.getenv("KB_TRIPLET_EXTRACTION_MAX_WORKERS", "2").strip())
        triplet_extraction_max_workers = max(1, triplet_extraction_max_workers)
        schema_grounding_enabled = _env_bool("KB_SCHEMA_GROUNDING_ENABLED", True)


        return cls(
            kg_retrieval_keyword=kg_retrieval_keyword,
            kg_limit_per_pattern=kg_limit_per_pattern,

            source_kind=source_kind,
            corpus_path=corpus_path,

            vector_similarity=vector_similarity,
            keybert=keybert,
            keyword_synonyms_enabled=keyword_synonyms_enabled,
            keyword_synonym_cache_enabled=keyword_synonym_cache_enabled,
            keyword_synonym_cache_path=keyword_synonym_cache_path,
            keyword_synonym_defaults_path=keyword_synonym_defaults_path,
            keyword_synonym_cache_write=keyword_synonym_cache_write,
            decomposer_parallel=decomposer_parallel,
            decomposer_max_workers=decomposer_max_workers,
            decomposer_batch_size=decomposer_batch_size,
            
            novelty_llm_max_tokens=novelty_llm_max_tokens,
            novelty_llm_temperature=novelty_llm_temperature,
            novelty_batch_size=novelty_batch_size,
            novelty_parallel=novelty_parallel,
            novelty_max_workers=novelty_max_workers,
            
            triplet_extraction_batch_size=triplet_extraction_batch_size,
            triplet_extraction_parallel=triplet_extraction_parallel,
            triplet_extraction_max_workers=triplet_extraction_max_workers,
            schema_grounding_enabled=schema_grounding_enabled,

            docling_enable_OCR=docling_enable_OCR,
            docling_enable_table_recognition=docling_enable_table_recognition,
            drop_reference_section=drop_reference_section,
            reference_section_filter_mode=reference_section_filter_mode,
        )
