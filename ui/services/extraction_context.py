from __future__ import annotations

"""
Per-run extraction context (in-memory).

When a pipeline run finishes we keep the heavy run context around, keyed by
job_id, so the UI can *incrementally* extend extraction when the reviewer lowers
the relevance threshold below the value the run used. Lowering the threshold
reveals paragraphs that were never decomposed; rather than re-running the whole
pipeline (or forcing a manual re-run), the extend endpoint decomposes only the
newly-included paragraphs and classifies them, reusing everything cached here.

Why a separate store (not JOB_STORE)?
- It holds non-serializable objects (Document list, PipelineConfig, KG relations).
- Its lifecycle is different: it can be evicted aggressively (it is a cache, not
  the job result). If it is gone (e.g. server restart), the UI simply falls back
  to the manual "re-run" hint.

Caveats: in-memory, single-process, lost on restart — same constraints as the
job store. We cap the number of retained contexts to bound memory.
"""

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Sequence, Set

from kbdebugger.compat.langchain import Document
from kbdebugger.pipeline.config import PipelineConfig

# Keep only the most recent few runs' contexts (bounds memory).
_MAX_CONTEXTS = 8


@dataclass
class ExtractionContext:
    """Everything needed to decompose+classify additional paragraphs later."""

    job_id: str
    all_paragraphs: List[Document]
    dimensions: List[str]
    cfg: PipelineConfig
    # original-paragraph-index -> {"score": float|None, "literal": bool}
    para_relevance: Dict[int, Dict[str, Any]]
    # original-paragraph-index -> set of dimension names that matched it
    dims_by_paragraph_index: Dict[int, Set[str]]
    # per-dimension scored chunks (for re-tagging new results' relevance)
    chunk_scores_by_dimension: Dict[str, Any]
    # cached combined KG subgraph (avoid re-querying Neo4j on each extend)
    kg_relations: List[Any]
    run_threshold: float
    # original paragraph indices already decomposed into qualities
    decomposed_indices: Set[int] = field(default_factory=set)
    # quality text already produced (for cross-extend dedup)
    existing_qualities: Set[str] = field(default_factory=set)


class _ExtractionContextStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._contexts: "Dict[str, ExtractionContext]" = {}
        self._order: List[str] = []  # insertion order for eviction

    def save(self, context: ExtractionContext) -> None:
        with self._lock:
            if context.job_id not in self._contexts:
                self._order.append(context.job_id)
            self._contexts[context.job_id] = context
            # Evict oldest beyond the cap.
            while len(self._order) > _MAX_CONTEXTS:
                oldest = self._order.pop(0)
                self._contexts.pop(oldest, None)

    def get(self, job_id: str) -> "ExtractionContext | None":
        with self._lock:
            return self._contexts.get(job_id)


EXTRACTION_CONTEXT_STORE = _ExtractionContextStore()


def newly_included_indices(
    context: ExtractionContext,
    threshold: float,
) -> List[int]:
    """
    Original paragraph indices that are included at ``threshold`` but have not
    been decomposed yet (i.e. their score sits between the new threshold and the
    threshold extraction has already reached). Literal matches are always already
    decomposed, so they never appear here.
    """
    out: List[int] = []
    for idx, rel in context.para_relevance.items():
        if idx in context.decomposed_indices:
            continue
        if rel.get("literal"):
            continue
        score = rel.get("score")
        if isinstance(score, (int, float)) and score >= threshold:
            out.append(int(idx))
    return sorted(out)
