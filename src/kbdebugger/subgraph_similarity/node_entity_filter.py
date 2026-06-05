from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from rich.console import Console
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from kbdebugger.keyword_extraction.keyBERT import extract_keyphrases_batch
from kbdebugger.subgraph_similarity.logging import build_qualities_to_subgraph_similarity_payload
from kbdebugger.types import GraphRelation
from kbdebugger.types.ui import ProgressCallback
from kbdebugger.utils.json import write_json
from kbdebugger.utils.progress import stage_status
from kbdebugger.utils.time import now_utc_compact

from .encoder import TextEncoder
from .index import VectorIndex
from .types import DroppedQuality, KeptQuality, NeighborHit, Quality, SubgraphSimilarityFilterConfig


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#.-]*")


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("_", " ").strip().lower())


def _add_candidate(candidates: list[str], seen: set[str], phrase: str, *, min_chars: int) -> None:
    cleaned = re.sub(r"\s+", " ", phrase.replace("_", " ").strip(" .,:;()[]{}\"'")).strip()
    if not cleaned:
        return
    key = _normalize_key(cleaned)
    if len(key) < min_chars or key in ENGLISH_STOP_WORDS or key in seen:
        return
    if key.replace(".", "").isdigit():
        return
    seen.add(key)
    candidates.append(cleaned)


def extract_candidate_entities(
    text: str,
    *,
    max_entities: int = 8,
    min_chars: int = 3,
) -> list[str]:
    """
    Lightweight non-LLM entity phrase extractor for similarity gating.

    It intentionally favors recall: noun-like content chunks, adjacent content
    bigrams, and individual content terms are all eligible. The vector threshold
    decides whether any candidate is close enough to a KG node.
    """
    if not text or not text.strip():
        return []

    tokens = [m.group(0).replace("_", " ") for m in _TOKEN_RE.finditer(text)]
    chunks: list[list[str]] = []
    current: list[str] = []

    for token in tokens:
        key = _normalize_key(token)
        if key in ENGLISH_STOP_WORDS:
            if current:
                chunks.append(current)
                current = []
            continue
        current.append(token)

    if current:
        chunks.append(current)

    candidates: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        if len(candidates) >= max_entities:
            break
        if len(chunk) == 1:
            _add_candidate(candidates, seen, chunk[0], min_chars=min_chars)
            continue

        if len(chunk) <= 5:
            _add_candidate(candidates, seen, " ".join(chunk), min_chars=min_chars)

        for n in (3, 2):
            if len(candidates) >= max_entities or len(chunk) < n:
                continue
            for i in range(len(chunk) - n + 1):
                _add_candidate(candidates, seen, " ".join(chunk[i : i + n]), min_chars=min_chars)
                if len(candidates) >= max_entities:
                    break

        for token in chunk:
            if len(candidates) >= max_entities:
                break
            _add_candidate(candidates, seen, token, min_chars=min_chars)

    return candidates[:max_entities]


def _dedupe_candidate_entities(
    phrases: Sequence[str],
    *,
    max_entities: int,
    min_chars: int = 3,
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        if len(candidates) >= max_entities:
            break
        _add_candidate(candidates, seen, phrase, min_chars=min_chars)
    return candidates


def extract_candidate_entities_batch(
    qualities: Sequence[Quality],
    *,
    cfg: SubgraphSimilarityFilterConfig,
    max_entities: int = 8,
) -> list[list[str]]:
    if cfg.entity_extraction_mode == "keybert":
        keyphrase_groups = extract_keyphrases_batch(
            qualities,
            embedding_model=cfg.entity_keybert_model_name,
            batch_size=cfg.entity_keybert_batch_size,
            top_n=max_entities,
            ngram_max=cfg.entity_keybert_ngram_max,
        )
        return [
            _dedupe_candidate_entities(group, max_entities=max_entities)
            for group in keyphrase_groups
        ]

    if cfg.entity_extraction_mode == "simple":
        return [
            extract_candidate_entities(q, max_entities=max_entities)
            for q in qualities
        ]

    raise ValueError(f"Unsupported entity extraction mode: {cfg.entity_extraction_mode!r}")


@dataclass
class KGNodePayload:
    label: str
    relations: list[GraphRelation] = field(default_factory=list)


def relation_key(relation: GraphRelation) -> str:
    props = relation["edge"]["properties"]
    return "|".join(
        [
            relation["source"]["label"],
            relation["edge"]["label"],
            relation["target"]["label"],
            str(props.get("sentence", "")),
        ]
    )


@dataclass
class NodeEntitySimilarityFilter:
    """
    Filter qualities by comparing extracted quality entities against KG node labels.
    """

    encoder: TextEncoder
    top_k: int = 5
    threshold: float = 0.50
    max_entities_per_quality: int = 8
    console: Console = field(default_factory=Console)
    num_nodes_indexed: int = 0
    num_unique_entities: int = 0

    def build_index(self, relations: Sequence[GraphRelation]) -> VectorIndex[KGNodePayload]:
        if not relations:
            raise ValueError("Cannot build node/entity vector index: 'relations' is empty.")

        nodes_by_key: dict[str, KGNodePayload] = {}
        for relation in relations:
            for end_name in ("source", "target"):
                label = str(relation[end_name]["label"]).strip()
                if not label:
                    continue
                key = _normalize_key(label)
                payload = nodes_by_key.setdefault(key, KGNodePayload(label=label))
                payload.relations.append(relation)

        nodes = list(nodes_by_key.values())
        if not nodes:
            raise ValueError("Cannot build node/entity vector index: no KG node labels found.")
        self.num_nodes_indexed = len(nodes)

        vectors = self.encoder.encode([node.label for node in nodes])
        index = VectorIndex[KGNodePayload].create(
            dim=self.encoder.dim,
            max_elements=len(nodes),
        )
        index.add(vectors, nodes)
        return index

    def filter_qualities(
        self,
        *,
        cfg: SubgraphSimilarityFilterConfig,
        index: VectorIndex[KGNodePayload],
        qualities: Sequence[Quality],
        progress: Optional[ProgressCallback] = None,
    ) -> Tuple[List[KeptQuality], List[DroppedQuality]]:
        total = 5
        step = 0

        def tick(msg: str) -> None:
            nonlocal step
            step += 1
            if progress:
                progress(step, total, msg)

        if not qualities:
            return ([], [])

        tick(f"🧩 Extracting quality entities ({cfg.entity_extraction_mode})…")
        entities_by_quality = extract_candidate_entities_batch(
            qualities,
            cfg=cfg,
            max_entities=self.max_entities_per_quality,
        )

        unique_entities: list[str] = []
        entity_to_index: dict[str, int] = {}
        for entities in entities_by_quality:
            for entity in entities:
                key = _normalize_key(entity)
                if key not in entity_to_index:
                    entity_to_index[key] = len(unique_entities)
                    unique_entities.append(entity)

        kept: List[KeptQuality] = []
        dropped: List[DroppedQuality] = []

        if not unique_entities:
            self.num_unique_entities = 0
            for q in qualities:
                dropped.append(
                    {
                        "quality": q,
                        "max_score": 0.0,
                        "match_mode": "node_entity",
                        "extracted_entities": [],
                        "dropped_reason": "No candidate entities extracted.",
                    }
                )
            self.save_similarity_results_json(cfg=cfg, kept=kept, dropped=dropped)
            return kept, dropped

        self.num_unique_entities = len(unique_entities)

        tick("🧬 Embedding quality entities…")
        entity_vectors = self.encoder.encode(unique_entities)
        entity_vectors_np = np.ascontiguousarray(np.asarray(entity_vectors), dtype=np.float32)

        with stage_status("📊 Performing batch node/entity similarity search:"):
            tick("📊 Searching nearest KG nodes…")
            node_hits_by_entity, scores = index.search_batch(entity_vectors_np, k=self.top_k)

        tick("🧮 Thresholding + assembling node/entity results…")
        for q, entities in zip(qualities, entities_by_quality):
            best_score = 0.0
            candidates: list[tuple[float, str, KGNodePayload]] = []

            for entity in entities:
                entity_idx = entity_to_index[_normalize_key(entity)]
                node_hits = node_hits_by_entity[entity_idx]
                row_scores = scores[entity_idx]
                n = min(len(node_hits), int(row_scores.shape[0]))
                for j in range(n):
                    score = float(row_scores[j])
                    best_score = max(best_score, score)
                    candidates.append((score, entity, node_hits[j]))

            if best_score < float(self.threshold):
                dropped.append(
                    {
                        "quality": q,
                        "max_score": best_score,
                        "match_mode": "node_entity",
                        "extracted_entities": entities,
                        "dropped_reason": "No extracted entity matched a KG node above threshold.",
                    }
                )
                continue

            candidates.sort(key=lambda item: item[0], reverse=True)
            neighbors: list[NeighborHit] = []
            seen_relations: set[str] = set()
            matched_entities: list[str] = []
            matched_node_labels: list[str] = []

            for score, entity, node in candidates:
                if score < float(self.threshold) and neighbors:
                    continue
                if entity not in matched_entities:
                    matched_entities.append(entity)
                if node.label not in matched_node_labels:
                    matched_node_labels.append(node.label)

                for relation in node.relations:
                    key = relation_key(relation)
                    if key in seen_relations:
                        continue
                    seen_relations.add(key)
                    neighbors.append(
                        {
                            "relation": relation,
                            "score": score,
                            "match_mode": "node_entity",
                            "matched_entity": entity,
                            "matched_node_label": node.label,
                        }
                    )
                    if len(neighbors) >= self.top_k:
                        break
                if len(neighbors) >= self.top_k:
                    break

            kept.append(
                {
                    "quality": q,
                    "max_score": best_score,
                    "neighbors": neighbors,
                    "match_mode": "node_entity",
                    "matched_entities": matched_entities,
                    "matched_node_labels": matched_node_labels,
                }
            )

        kept.sort(key=lambda x: x["max_score"], reverse=True)
        dropped.sort(key=lambda x: x["max_score"], reverse=True)

        tick("💾 Saving node/entity similarity results to JSON log…")
        self.save_similarity_results_json(cfg=cfg, kept=kept, dropped=dropped)
        return kept, dropped

    def pretty_print(
        self,
        *,
        kept: Sequence[KeptQuality],
        dropped: Sequence[DroppedQuality],
    ) -> None:
        self.console.print(
            f"[cyan]Node/entity similarity: kept={len(kept)}, dropped={len(dropped)}, "
            f"threshold={self.threshold:.3f}, top_k={self.top_k}[/cyan]"
        )

    def save_similarity_results_json(
        self,
        *,
        cfg: SubgraphSimilarityFilterConfig,
        kept: Sequence[KeptQuality],
        dropped: Sequence[DroppedQuality],
    ) -> Mapping[str, Any]:
        log_payload = build_qualities_to_subgraph_similarity_payload(
            cfg=cfg,
            kept=kept,
            dropped=dropped,
        )
        log_payload["node_entity"] = {
            "num_kg_nodes_indexed": self.num_nodes_indexed,
            "num_unique_quality_entities": self.num_unique_entities,
            "threshold": float(self.threshold),
            "top_k": int(self.top_k),
            "max_entities_per_quality": int(self.max_entities_per_quality),
            "entity_extraction_mode": cfg.entity_extraction_mode,
            "entity_keybert_model_name": cfg.entity_keybert_model_name,
            "entity_keybert_batch_size": cfg.entity_keybert_batch_size,
            "entity_keybert_ngram_max": cfg.entity_keybert_ngram_max,
        }
        path = f"logs/03_vector_similarity_filter_results_{now_utc_compact()}.json"
        write_json(path, log_payload)
        print(f"\n[INFO] 📚️📊 Wrote node/entity similarity results to {path}")
        return log_payload
