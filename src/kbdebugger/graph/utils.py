import re
from typing import Any, Iterable, List, Mapping, Optional

from kbdebugger.compat.langchain import Document
from kbdebugger.extraction.predicate_options import DEFAULT_ALLOWED_PREDICATES
from kbdebugger.types import EdgeProperties, ExtractionResult, GraphRelation


_CAMEL_BOUNDARY_RE = re.compile(r"(?<!^)(?=[A-Z])")
_SAFE_REL_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ALLOWED_PREDICATE_TYPES: dict[str, str] = {
    predicate: _CAMEL_BOUNDARY_RE.sub("_", predicate).lower()
    for predicate in DEFAULT_ALLOWED_PREDICATES
}


def normalize_text(text: str) -> str:
    """
    Normalize a free-text label into a search key:
    - lowercase
    - collapse whitespace
    """
    clean = " ".join(text.strip().split()).lower()
    # clean = clean.replace(" ", "_")
    return clean


def normalize_graph_name(text: str) -> str:
    """Preserve graph node names while trimming whitespace noise."""
    return " ".join(str(text).strip().split())


def predicate_to_relationship_type(predicate: str) -> str:
    """
    Convert a predicate to a safe Neo4j relationship type.

    Standard predicates map through the precomputed whitelist. Non-standard
    predicates (kept for reviewer attention and explicitly included on submit)
    are sanitized into snake_case; the result is only safe to inject into
    Cypher after the `_SAFE_REL_TYPE_RE` check passes.
    """
    raw = str(predicate).strip()
    rel_type = _ALLOWED_PREDICATE_TYPES.get(raw)
    if rel_type is None:
        snake = _CAMEL_BOUNDARY_RE.sub("_", raw).lower()
        snake = re.sub(r"[^a-z0-9]+", "_", snake).strip("_")
        rel_type = snake
    if not rel_type or not _SAFE_REL_TYPE_RE.fullmatch(rel_type):
        raise ValueError(f"Unsafe Neo4j relationship type generated for predicate {predicate!r}: {rel_type!r}")
    return rel_type


def map_doc_extracted_triplets_to_graph_relations(
    extraction: ExtractionResult,
    source_doc: Document,
    # *,
    # include_sentence: bool = True,
) -> List[GraphRelation]:
    """
    Map an ExtractionResult to graph-ready relation dicts.
    - extraction: {"sentence": str, "triplets": [(subj,obj,rel), ...]}
    - source_doc: LangChain Document (for provenance: page_content + metadata)
    """
    # defensive: accept partially-typed dicts
    sentence_text = extraction.get("sentence")
    triplets = extraction.get("triplets", [])

    rels: List[GraphRelation] = []
    for subj, obj, rel in triplets:

        props: EdgeProperties = {
            # for provenance
            'sentence': sentence_text,
            'original_sentence': getattr(source_doc, "page_content", ""),
            **getattr(source_doc, "metadata", {})  # type: ignore[arg-type]
        }

        # if include_sentence:
        #     # human-readable extracted sentence (from the extractor)
        #     props["sentence"] = sentence_text

        rels.append({
            "source": { "label": normalize_graph_name(subj) },
            "target": { "label": normalize_graph_name(obj) },
            "edge":   { "label": str(rel).strip(), "properties": props },
        }) # type: ignore

    return rels


def map_extracted_triplets_to_graph_relations(
    extraction: ExtractionResult,
    source: Optional[str] = None,
) -> List[GraphRelation]:
    """
    Map an ExtractionResult to graph-ready relation dicts.
    - extraction: {"sentence": str, "triplets": [(subj,obj,rel), ...],
                   "provenance": {doc_name, quality, chunk_index, chunk_excerpt}?}
    - source: optional fallback provenance string (e.g. uploaded file name)
    """
    # defensive: accept partially-typed dicts
    sentence_text = extraction.get("sentence")
    triplets = extraction.get("triplets", [])
    provenance = extraction.get("provenance")
    if not isinstance(provenance, dict):
        provenance = None

    rels: List[GraphRelation] = []
    for subj, obj, rel in triplets:

        props: EdgeProperties = {
            # for provenance
            'sentence': sentence_text,
            **({'source': source} if source else {}),  # only include if source is provided
            **({'provenance': provenance} if provenance else {}),  # structured per-doc provenance
        }

        rels.append({
            "source": { "label": normalize_graph_name(subj) },
            "target": { "label": normalize_graph_name(obj) },
            "edge":   { "label": str(rel).strip(), "properties": props },
        }) # type: ignore

    return rels


def rows_to_graph_relations(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_key: str = "source",
    target_key: str = "target",
    predicate_key: str = "predicate",
    props_key: str = "props",
    # if we want to enforce required props fields, will do it here
) -> List[GraphRelation]:
    rels: List[GraphRelation] = []

    for row in rows:
        source = row[source_key]
        target = row[target_key]
        predicate = row[predicate_key]
        props_raw = row.get(props_key, {}) or {}

        source_id = str(row.get("source_id", ""))
        target_id = str(row.get("target_id", ""))

        if not isinstance(props_raw, dict):
            raise TypeError(f"Expected '{props_key}' to be a dict, got {type(props_raw)}: {props_raw!r}")

        props: EdgeProperties = {**props_raw}  # type: ignore[misc]

        rels.append(
            {
                "source": {
                    "label": str(source), 
                    "id": source_id,
                },
                "target": {
                    "label": str(target), 
                    "id": target_id,
                },
                "edge": {
                    "label": str(predicate), 
                    "properties": props
                }   
            } # type: ignore
        )

    return rels

