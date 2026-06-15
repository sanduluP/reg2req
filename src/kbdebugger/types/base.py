from __future__ import annotations

from enum import Enum
from typing import Optional, TypedDict, Any
from typing_extensions import Literal, Required, NotRequired

Subject = str
Predicate = str
Object = str

TripletSubjectObjectPredicate = tuple[Subject, Object, Predicate]  # (Subject, Object, Predicate)

class ExtractionResult(TypedDict):
    sentence: str
    triplets: list[TripletSubjectObjectPredicate]  # [["Subject", "Object", "Relation"], ...]
    skipped_reason: NotRequired[str]
    original_quality: NotRequired[str]
    source_context: NotRequired[Any]
    decision: NotRequired[str]
    max_score: NotRequired[float]
    matched_neighbor_sentence: NotRequired[str]
    upsert_eligible: NotRequired[bool]
    schema_status: NotRequired[str]
    schema_template: NotRequired[str]
    grounding_confidence: NotRequired[float]
    matched_schema_nodes: NotRequired[list[str]]
    inferred_node_types: NotRequired[list[str]]
    schema_notes: NotRequired[list[str]]
    schema_grounding: NotRequired[Any]
    non_standard_predicates: NotRequired[list[str]]
    modality: NotRequired[str]  # MANDATORY | RECOMMENDED | OPTIONAL | PROHIBITED
    provenance: NotRequired[Any]  # {doc_name, quality, chunk_index, chunk_excerpt, modality}

class GraphEnd(TypedDict):
    label: str
    id: Optional[str]
    created_at: Optional[str]
    last_updated_at: Optional[str]

class GraphEdge(TypedDict):
    label: str
    properties: EdgeProperties

class EdgeProperties(TypedDict, total=False):
    # total=False: all fields are optional, then we enforce some as Required below
    """
    Canonical schema for a relationship's properties stored in Neo4j.
    We mark some keys Required for downstream logic, and allow extra keys.
    """
    # ------- Required (minimum we rely on) -------
    label: NotRequired[str]                        # e.g., "is" when the sentence is: "AI is transformative"
    sentence: NotRequired[str]                  # human-readable sentence (from extractor or synthesized)
    # predicate_text: NotRequired[str]            # original predicate text (before normalization)

    # ------- Strongly recommended provenance -------
    original_sentence: NotRequired[str]      # the raw text from the Document chunk
    source: NotRequired[str]            # e.g., PDF/file name
    page_number: NotRequired[int]
    start_index: NotRequired[int]
    end_index: NotRequired[int]
    doc_id: NotRequired[str]                 # internal ID of the doc/chunk
    chunk_id: NotRequired[str]               # if you chunked documents
    provenance: NotRequired[Any]             # structured {doc_name, quality, chunk_index, chunk_excerpt}; serialized before Neo4j write

    # ------- Quality / versioning -------
    confidence: NotRequired[float]           # model confidence if available
    extractor_version: NotRequired[str]      # version of your extraction pipeline
    created_at: NotRequired[str]             # ISO timestamp
    last_updated_at: NotRequired[str]        # ISO timestamp

    # ------- Open-ended for extra metadata -------
    # Any extra keys from sentence_doc.metadata are allowed
    # because TypedDict (total=False) + extra merge is permitted.

class GraphRelation(TypedDict):
    source: GraphEnd
    target: GraphEnd
    edge: GraphEdge

# TODO: Remove if unused
CoreEdgePropertyKey = Literal[
    "label", "sentence", "predicate_text"
]

ProvenanceEdgePropertyKey = Literal[
    "original_sentence", "source", "page_number",
    "start_index", "end_index", "doc_id", "chunk_id",
]

QualityEdgePropertyKey = Literal[
    "confidence", "extractor_version", "created_at", "last_updated_at"
]

EdgePropertyKey = (
    CoreEdgePropertyKey
    | ProvenanceEdgePropertyKey
    | QualityEdgePropertyKey
)