from __future__ import annotations

from typing import Optional, Sequence, Tuple

from kbdebugger.extraction.types import Qualities
from kbdebugger.subgraph_similarity.logging import build_qualities_to_subgraph_similarity_payload
from kbdebugger.types import GraphRelation
from kbdebugger.types.ui import ProgressCallback
from .encoder import SentenceTransformerEncoder
from .node_entity_filter import NodeEntitySimilarityFilter
from .similarity_filter import SubgraphSimilarityFilter
from .types import KeptQuality, DroppedQuality,SubgraphSimilarityFilterConfig



def filter_qualities_by_subgraph_similarity(
    *,
    kg_relations: Sequence[GraphRelation],
    qualities: Qualities,
    cfg: SubgraphSimilarityFilterConfig,
    pretty_print: bool = True,
    progress: Optional[ProgressCallback] = None
) -> Tuple[
        Tuple[
            list[KeptQuality], 
            list[DroppedQuality]
        ], 
        dict
    ]:
    """
    Public API: run the full vector similarity filter stage.

    Hides:
    - encoder initialization
    - filter initialization
    - index building

    Parameters:
        kg_relations:
            The retrieved subgraph relations for a given keyword.
            This subgraph is used to build the vector index.
            i.e. it is our search space here.

        qualities:
            The candidate qualities extracted from a corpus (e.g. decomposer output).
            Each quality will be treated as a query vector and its cosine similarity
            to the subgraph relation vectors will be computed.

        cfg:
            Configuration for the vector similarity filter stage.

        pretty_print:
            Whether to pretty-print the filtering results to console.

    Returns
    -------
    ((kept, dropped), log_payload)
    """
    total = 2
    step = 0
    def tick(msg: str):
        nonlocal step
        step += 1
        if progress:
            progress(step, total, msg)

    tick(f"📚 Building KG vector index ({cfg.similarity_mode})...")
    encoder = SentenceTransformerEncoder(
        model_name=cfg.encoder_model_name,
        device=cfg.encoder_device,
        normalize=cfg.normalize_embeddings,
    )

    if cfg.similarity_mode == "sentence":
        filt = SubgraphSimilarityFilter(
            encoder=encoder,
            top_k=cfg.quality_to_kg_top_k,
            threshold=cfg.min_similarity_threshold,
        ) # the word "filter" in python is overloaded, so I use "filt" for the instance name
        index = filt.build_index(kg_relations)

        tick("📊 Running similarity search (cos_sim<qualities, kg_relations>)...")
        kept, dropped = filt.filter_qualities(
            cfg=cfg,
            index=index,
            qualities=qualities,
            progress=progress
        )

    elif cfg.similarity_mode == "node_entity":
        filt = NodeEntitySimilarityFilter(
            encoder=encoder,
            top_k=cfg.node_entity_top_k,
            threshold=float(cfg.node_entity_min_similarity_threshold),
            max_entities_per_quality=cfg.node_entity_max_entities_per_quality,
        )
        index = filt.build_index(kg_relations)

        tick("📊 Running similarity search (cos_sim<quality_entities, kg_nodes>)...")
        kept, dropped = filt.filter_qualities(
            cfg=cfg,
            index=index,
            qualities=qualities,
            progress=progress
        )

    else:
        raise ValueError(f"Unsupported similarity mode: {cfg.similarity_mode!r}")

    log_payload = build_qualities_to_subgraph_similarity_payload(
        cfg=cfg,
        kept=kept,
        dropped=dropped,
    )

    if pretty_print:
        filt.pretty_print(kept=kept, dropped=dropped)

    return (kept, dropped), log_payload
