"""
AI-assisted extraction-quality checks, run on extracted triples before merge:

- faithfulness:  LLM check that each triple's subject/object is actually
                 supported by its source sentence (catches hallucinated S/O).
- entity dedup:  embedding-based detection of near-duplicate entity names
                 (e.g. "Explainability" vs "Explicability") to merge.
- relation dedup: duplicate / equivalent (subject, predicate, object) edges.

All LLM calls go through the env-driven `kbdebugger.llm` backend, so they use
whatever model the local .env selects (DFKI server for the team, DeepSeek for a
bring-your-own-key fallback).
"""

from .checks import run_quality_check, check_faithfulness, find_entity_duplicates, find_relation_duplicates

__all__ = [
    "run_quality_check",
    "check_faithfulness",
    "find_entity_duplicates",
    "find_relation_duplicates",
]
