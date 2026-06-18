from __future__ import annotations

from typing import List, Optional, Sequence

from kbdebugger.types import GraphRelation, ExtractionResult
from .retriever import KnowledgeGraphRetriever
from .utils import map_extracted_triplets_to_graph_relations
from .types import BatchUpsertSummary
from . import get_graph

from .cytoscape import graph_relations_to_cytoscape, CytoscapeGraphPayload

def retrieve_keyword_subgraph(
    *,
    keyword: str,
    limit_per_pattern: int,
) -> List[GraphRelation]:
    """
    Retrieve a keyword-guided KG subgraph from Neo4j and return its relations.

    This function is a **public stage API**: it performs the retrieval and
    enforces a clear contract for downstream stages.

    Parameters
    ----------
    keyword:
        Keyword used to drive the subgraph retrieval patterns.

    limit_per_pattern:
        Maximum number of relations returned per retrieval pattern in the
        KnowledgeGraphRetriever.

    Returns
    -------
    List[GraphRelation]
        The retrieved relations (GraphRelation dicts), ready to be used as the
        reference set for vector similarity filtering.

    Raises
    ------
    ValueError
        If no relations were retrieved. This typically indicates that:
        - the keyword does not exist in the KG (or is too specific),
        - the KG is empty,
        - the retriever patterns are too restrictive,
        - or Neo4j connectivity/configuration is wrong.
    """
    retriever = KnowledgeGraphRetriever(limit_per_pattern=limit_per_pattern)
    hits = retriever.retrieve(keyword)
    relations = [h["relation"] for h in hits]

    return relations


def retrieve_keyword_subgraph_cytoscape(
    *,
    keyword: str,
    limit_per_pattern: int,
) -> CytoscapeGraphPayload:
    """
    Retrieve a keyword-guided KG subgraph and return Cytoscape-ready elements.

    Why this exists
    ---------------
    - `retrieve_keyword_subgraph()` is a pipeline stage API and returns `List[GraphRelation]`
      for downstream algorithmic stages (similarity, novelty, etc.).
    - The UI needs Cytoscape.js elements: {"elements": {"nodes": [...], "edges": [...]}}

    This function is a pure adapter:
    - no new graph logic
    - no DB access beyond the underlying retrieval call
    - strictly typed UI contract

    Parameters
    ----------
    keyword:
        Keyword used to drive the subgraph retrieval patterns.

    limit_per_pattern:
        Maximum number of relations returned per retrieval pattern.

    Returns
    -------
    CytoscapeGraphPayload
        Cytoscape.js compatible graph payload.
    """
    relations = retrieve_keyword_subgraph(
        keyword=keyword,
        limit_per_pattern=limit_per_pattern,
    )
    return graph_relations_to_cytoscape(relations)


def retrieve_full_graph_cytoscape(*, limit: int = 5000) -> CytoscapeGraphPayload:
    """
    Retrieve the ENTIRE knowledge graph (all relationships) as Cytoscape-ready
    elements — used by the "Complete scan (all dimensions)" view.

    Parameters
    ----------
    limit:
        Safety cap on the number of relationships returned, to avoid rendering
        a pathologically large graph in the browser.

    Returns
    -------
    CytoscapeGraphPayload
        Cytoscape.js compatible graph payload for the whole graph.
    """
    graph = get_graph()
    relations = graph.query_relations(
        """
        MATCH (n:Node)-[r]->(m:Node)
        RETURN
            n.name AS source,
            m.name AS target,
            type(r) AS predicate,
            properties(r) AS props,
            elementId(n) AS source_id,
            elementId(m) AS target_id,
            elementId(r) AS rel_id
        LIMIT $limit
        """,
        {"limit": int(limit)},
    )
    return graph_relations_to_cytoscape(relations)


def upsert_extracted_triplets(
    *,
    extractions: Sequence[ExtractionResult],
    source: Optional[str] = None,
    pretty_print: bool = True,
) -> BatchUpsertSummary:
    """
    Convert triplet extraction outputs into graph relations, then upsert them.

    Why this exists
    ---------------
    The triplet extractor returns "ExtractionResult" objects:
        {
          "sentence": str,
          "triplets": [(subject, object, predicate), ...]
        }

    Neo4j upsert expects GraphRelation objects:
        {
          "source": {"label": ...},
          "target": {"label": ...},
          "edge": {"label": ..., "properties": {...}}
        }

    This function is the stage boundary that:
    - maps ExtractionResult -> list[GraphRelation]
    - batches all relations together
    - performs a single high-level upsert call

    Parameters
    ----------
    graph:
        Connected GraphStore instance.

    extractions:
        Sequence of ExtractionResult items. Each item corresponds to one input sentence,
        containing zero or more extracted triplets.

    source:
        Optional provenance string (e.g., PDF filename). If provided, it is stored on
        relationship properties under key "source".

    pretty_print:
        If True, print an upsert summary.

    Returns
    -------
    BatchUpsertSummary
        Summary across all relations produced by all extractions.
    """
    graph = get_graph()
    all_relations: List[GraphRelation] = []

    for extraction in extractions:
        # extractions is like a list of list of triplets.
        # Thus, we have to iterate through each extraction (corresponding to one quality sentence) 
        # and map its triplets to graph relations. And we accumulate all triplets in one flat list to do a single upsert at the end.
        rels = map_extracted_triplets_to_graph_relations(extraction, source=source)
        all_relations.extend(rels)

    # Here we do a single batch upsert for all relations extracted from all sentences.
    # This is more efficient than upserting per `extraction`, and allows us to get a comprehensive summary of the batch operation.
    return graph.upsert_relations(all_relations, pretty_print=pretty_print)
