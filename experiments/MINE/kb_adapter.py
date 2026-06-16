"""Build a KGGen-format knowledge graph from a single essay using KBExtractor.

This is the "MINE mode" bypass of the full KBDebugger pipeline: we run only the
two stages that turn text into triples — decomposition and triplet extraction —
skipping the keyword gate, similarity/novelty filters, human oversight, and
Neo4j. For MINE (information *retention*) those filters can only remove
information, so the extract-everything core is the upper bound of what
KBExtractor's graph could contain.

Imports ``kbdebugger``; run inside the KBExtraction virtualenv with
``PYTHONPATH=src``.
"""
from __future__ import annotations

import re
from typing import Any

from kbdebugger.compat.langchain import Document
from kbdebugger.extraction.api import decompose_paragraphs_to_qualities
from kbdebugger.extraction.triplet_extraction_batch import extract_triplets_batch

from graph_format import extraction_results_to_graph_dict

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")


def essay_to_paragraph_docs(essay: str) -> list[Any]:
    """Split a plain-text essay into paragraph ``Document``s (no Docling/PDF)."""
    blocks = [b.strip() for b in _PARAGRAPH_SPLIT.split(essay) if b.strip()]
    if not blocks:
        blocks = [essay.strip()] if essay.strip() else []
    return [
        Document(page_content=block, metadata={"source": "mine_essay"})
        for block in blocks
    ]


def build_kg_from_essay(
    essay: str,
    *,
    batch_size: int = 5,
    parallel: bool = True,
    max_workers: int = 2,
    lowercase: bool = True,
) -> dict[str, list]:
    """Essay text → KGGen ``Graph`` dict via decompose + triplet extraction.

    Triplet extraction runs in ``neutral_mode``: free-form natural-language
    predicates, with the standards controlled-vocabulary and schema-grounding
    layers switched off. On general-topic MINE essays a fixed ISO/standards
    ontology would force-fit or discard facts, so neutral mode is what fairly
    reflects KBExtractor's real extraction core for this benchmark.
    """
    docs = essay_to_paragraph_docs(essay)
    if not docs:
        return {"entities": [], "edges": [], "relations": []}

    qualities, _decomposer_log = decompose_paragraphs_to_qualities(
        paragraphs=docs,
        batch_size=batch_size,
        parallel=parallel,
        max_workers=max_workers,
    )
    if not qualities:
        return {"entities": [], "edges": [], "relations": []}

    extraction_results = extract_triplets_batch(
        qualities,
        batch_size=batch_size,
        parallel=parallel,
        max_workers=max_workers,
        neutral_mode=True,  # free-form predicates, no standards vocab / schema grounding
    )

    return extraction_results_to_graph_dict(extraction_results, lowercase=lowercase)
