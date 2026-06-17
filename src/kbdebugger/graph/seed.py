from __future__ import annotations

"""
Seed the knowledge graph with curated Trustworthy-AI ground-truth knowledge.

Why this exists
---------------
A fresh Neo4j instance is empty, so selecting a keyword in the UI returns an
empty subgraph and a pipeline run fails with "No KG relations retrieved". This
module parses a curated set of natural-language statements into (subject,
predicate, object) triples and upserts them through the SAME Neo4j write path
the extraction pipeline uses (`GraphStore.upsert_relations`), so seeded edges
carry identical structure:

- `(:Node {name})` source/target nodes (casing preserved)
- a typed relationship (e.g. `:is_subclass_of`) via `predicate_to_relationship_type`
- the originating `sentence`
- append-only `provenance_records` / `provenance_docs`

Seeded edges additionally carry `knowledge_type = "seed"` and a
`provenance_source` of `seed:trustworthy-ai`, so ground-truth knowledge is
distinguishable from pipeline-extracted knowledge in the graph.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from kbdebugger.graph.utils import predicate_to_relationship_type
from kbdebugger.types import GraphRelation

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEED_PATH = _REPO_ROOT / "data" / "seed" / "trustworthy_ai_seed.txt"
DEFAULT_SEED_JSON_PATH = _REPO_ROOT / "data" / "seed" / "trustworthy_ai_seed.json"

SEED_SOURCE_LABEL = "seed:trustworthy-ai"
SEED_KNOWLEDGE_TYPE = "seed"

# Surface predicate phrase -> standard predicate (PascalCase). Order does not
# matter for correctness (we pick the earliest, then longest match per line),
# but keep multi-word phrases listed before their shorter prefixes for clarity.
_PHRASE_TO_PREDICATE: tuple[tuple[str, str], ...] = (
    ("is subclass of", "IsSubclassOf"),
    ("is dimension of", "IsDimensionOf"),
    ("is threat to", "IsThreatTo"),
    ("contributes to", "ContributesTo"),
    ("applies to", "AppliesTo"),
    ("implements", "Implements"),
    ("is an", "IsAn"),
    ("is a", "IsA"),
)
_PHRASE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (predicate, re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE))
    for phrase, predicate in _PHRASE_TO_PREDICATE
]


@dataclass(frozen=True, slots=True)
class SeedTriple:
    subject: str
    predicate: str
    object: str
    sentence: str


def _clean(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def parse_seed_line(line: str) -> Optional[SeedTriple]:
    """
    Parse one statement line into a SeedTriple, or None for blank/comment/
    unparseable lines. The predicate phrase is detected as a whole-word match;
    on ties the earliest then longest phrase wins.
    """
    text = line.strip()
    if not text or text.startswith("#"):
        return None

    best: Optional[tuple[tuple[int, int], str, re.Match[str]]] = None
    for predicate, pattern in _PHRASE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        # earliest start first, then longest phrase (negative length sorts longer first)
        sort_key = (match.start(), -(match.end() - match.start()))
        if best is None or sort_key < best[0]:
            best = (sort_key, predicate, match)

    if best is None:
        return None

    _key, predicate, match = best
    subject = _clean(text[: match.start()])
    obj = _clean(text[match.end() :])
    if not subject or not obj:
        return None

    return SeedTriple(subject=subject, predicate=predicate, object=obj, sentence=text)


def load_seed_text(path: Optional[str | Path] = None) -> str:
    """Read the seed sentences file (env `KB_SEED_PATH` overrides the default)."""
    resolved = Path(path or os.getenv("KB_SEED_PATH") or DEFAULT_SEED_PATH)
    if not resolved.exists():
        raise FileNotFoundError(f"Seed file not found: {resolved}")
    return resolved.read_text(encoding="utf-8")


def parse_seed_triples(text: str) -> list[SeedTriple]:
    """Parse all statements, dropping exact duplicate (subject, predicate, object)."""
    seen: set[tuple[str, str, str]] = set()
    out: list[SeedTriple] = []
    for line in text.splitlines():
        triple = parse_seed_line(line)
        if triple is None:
            continue
        key = (triple.subject.lower(), triple.predicate, triple.object.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(triple)
    return out


def seed_triples_to_relations(
    triples: Sequence[SeedTriple],
    *,
    source_label: str = SEED_SOURCE_LABEL,
    knowledge_type: str = SEED_KNOWLEDGE_TYPE,
) -> list[GraphRelation]:
    """
    Map SeedTriples to GraphRelation dicts matching the pipeline's write shape.
    Validates predicates up front so a typo fails fast rather than mid-write.
    """
    for predicate in {t.predicate for t in triples}:
        predicate_to_relationship_type(predicate)  # raises on unsupported/unsafe

    relations: list[GraphRelation] = []
    for idx, triple in enumerate(triples):
        relations.append(
            {
                "source": {"label": triple.subject},
                "target": {"label": triple.object},
                "edge": {
                    "label": triple.predicate,
                    "properties": {
                        "sentence": triple.sentence,
                        "source": source_label,
                        "knowledge_type": knowledge_type,
                        "provenance": {
                            "doc_name": source_label,
                            "quality": triple.sentence,
                            "chunk_index": idx,
                            "chunk_excerpt": triple.sentence,
                        },
                    },
                },
            }  # type: ignore[typeddict-item]
        )
    return relations


def purge_seed_knowledge(
    graph: Optional[Any] = None,
    *,
    knowledge_type: str = SEED_KNOWLEDGE_TYPE,
) -> None:
    """
    Delete only seed-tagged relationships, then drop nodes left with no edges.
    Pipeline-extracted edges (no `knowledge_type`) are untouched.
    """
    if graph is None:
        from kbdebugger.graph import get_graph

        graph = get_graph()

    graph.query(
        "MATCH ()-[r]->() WHERE r.knowledge_type = $kt DELETE r",
        {"kt": knowledge_type},
    )
    graph.query("MATCH (n:Node) WHERE NOT (n)--() DELETE n")


def seed_knowledge_graph(
    graph: Optional[Any] = None,
    *,
    path: Optional[str | Path] = None,
    source_label: str = SEED_SOURCE_LABEL,
    knowledge_type: str = SEED_KNOWLEDGE_TYPE,
    reset_all: bool = False,
    clear_existing: bool = False,
    pretty_print: bool = False,
) -> dict[str, Any]:
    """
    Parse the seed file and upsert it into Neo4j (idempotent via MERGE).

    Parameters
    ----------
    reset_all:
        If True, clear the WHOLE graph (every node and relationship) before
        seeding — this is the "Initialize graph" behavior: build the curated
        baseline on a clean database.
    clear_existing:
        If True (and `reset_all` is False), remove only previously seeded edges
        (and orphaned nodes) first; pipeline-extracted knowledge is kept.

    Returns
    -------
    dict
        Summary counts for the UI / CLI.
    """
    if graph is None:
        from kbdebugger.graph import get_graph

        graph = get_graph()

    triples = parse_seed_triples(load_seed_text(path))
    relations = seed_triples_to_relations(
        triples, source_label=source_label, knowledge_type=knowledge_type
    )

    if reset_all:
        graph.reset_graph()
    elif clear_existing:
        purge_seed_knowledge(graph, knowledge_type=knowledge_type)

    summary = graph.upsert_relations(relations, pretty_print=pretty_print)

    nodes = {t.subject for t in triples} | {t.object for t in triples}
    return {
        "sentences": len(triples),
        "relations": len(relations),
        "nodes": len(nodes),
        "attempted": summary.attempted,
        "succeeded": summary.succeeded,
        "failed": summary.failed,
        "errors": summary.errors[:5],
        "source_label": source_label,
        "knowledge_type": knowledge_type,
        "reset_all": reset_all,
        "cleared_existing": clear_existing,
    }


def dump_seed_json(out_path: Optional[str | Path] = None) -> Path:
    """Write the parsed triples to a committed JSON artifact for inspection."""
    triples = parse_seed_triples(load_seed_text())
    data = {
        "source_label": SEED_SOURCE_LABEL,
        "knowledge_type": SEED_KNOWLEDGE_TYPE,
        "count": len(triples),
        "triples": [
            {
                "subject": t.subject,
                "predicate": t.predicate,
                "object": t.object,
                "sentence": t.sentence,
            }
            for t in triples
        ],
    }
    resolved = Path(out_path or DEFAULT_SEED_JSON_PATH)
    resolved.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return resolved
