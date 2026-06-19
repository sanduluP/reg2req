"""
PipelineConfig service for the Flask UI.

Why this exists
---------------
- The core project already defines `PipelineConfig.from_env()` as the *single*
  authoritative place for runtime configuration.
- The UI should not re-parse environment variables inside routes.
- We create the config once (cached) and reuse it across requests.

Notes
-----
- We assume `.env` is loaded once during app startup (in the app factory),
  not inside request handlers.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Any, Mapping

from kbdebugger.pipeline.config import PipelineConfig


@lru_cache(maxsize=1)
def get_pipeline_config() -> PipelineConfig:
    """
    Return a cached PipelineConfig loaded from environment variables.

    Returns
    -------
    PipelineConfig
        Central runtime config for the pipeline / UI.
    """
    return PipelineConfig.from_env()


def current_thresholds(cfg: PipelineConfig | None = None) -> dict[str, float | int]:
    """The tunable thresholds and their current (env-derived) defaults."""
    cfg = cfg or get_pipeline_config()
    vs = cfg.vector_similarity
    return {
        # Paragraph relevance: how close a paragraph must be to the keyword.
        "para_threshold": float(cfg.keybert.search_kw_to_paragraph_similarity_threshold),
        # Quality <-> KG similarity: how close a quality must be to the subgraph.
        "sim_threshold": float(
            vs.node_entity_min_similarity_threshold
            if vs.node_entity_min_similarity_threshold is not None
            else vs.min_similarity_threshold
        ),
        # Neighbors retrieved per quality.
        "top_k": int(vs.node_entity_top_k),
        # KG relations pulled per retrieval pattern.
        "kg_limit": int(cfg.kg_limit_per_pattern),
    }


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def apply_threshold_overrides(
    cfg: PipelineConfig, overrides: Mapping[str, Any] | None
) -> PipelineConfig:
    """
    Return a copy of ``cfg`` with UI-supplied threshold overrides applied.

    Recognized keys (all optional): para_threshold, sim_threshold (both 0..1),
    top_k, kg_limit (>= 1). Missing or invalid values are ignored, so the
    env-derived defaults stand.
    """
    if not overrides:
        return cfg

    def _as_float(key: str) -> float | None:
        raw = overrides.get(key)
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _as_int(key: str) -> int | None:
        raw = overrides.get(key)
        if raw is None or raw == "":
            return None
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return None

    keybert = cfg.keybert
    vector = cfg.vector_similarity
    top_changes: dict[str, Any] = {}

    para = _as_float("para_threshold")
    if para is not None:
        keybert = replace(
            keybert,
            search_kw_to_paragraph_similarity_threshold=_clamp(para, 0.0, 1.0),
        )

    sim = _as_float("sim_threshold")
    top_k = _as_int("top_k")
    vector_changes: dict[str, Any] = {}
    if sim is not None:
        clamped = _clamp(sim, 0.0, 1.0)
        vector_changes["min_similarity_threshold"] = clamped
        vector_changes["node_entity_min_similarity_threshold"] = clamped
    if top_k is not None:
        k = max(1, top_k)
        vector_changes["quality_to_kg_top_k"] = k
        vector_changes["node_entity_top_k"] = k
    if vector_changes:
        vector = replace(vector, **vector_changes)

    kg_limit = _as_int("kg_limit")
    if kg_limit is not None:
        top_changes["kg_limit_per_pattern"] = max(1, kg_limit)

    if keybert is not cfg.keybert:
        top_changes["keybert"] = keybert
    if vector is not cfg.vector_similarity:
        top_changes["vector_similarity"] = vector

    return replace(cfg, **top_changes) if top_changes else cfg
