"""
Extraction-quality checks (faithfulness, entity dedup, relation dedup).

These run on the extracted triples *before* they are written to the graph, so
the reviewer can drop hallucinations and merge duplicates first.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Optional, Sequence

from kbdebugger.graph.utils import normalize_text, predicate_to_relationship_type


# ---------------------------------------------------------------------------
# 3) Relation duplicates — deterministic
# ---------------------------------------------------------------------------
def find_relation_duplicates(triples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Group triples that collapse to the same (subject, relationship-type, object).
    Predicate phrasing differences that map to the same relationship type
    (e.g. "Ensures" vs "ensures") are treated as duplicates.
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for t in triples:
        subj, pred, obj = t.get("subject", ""), t.get("predicate", ""), t.get("object", "")
        if not (subj and pred and obj):
            continue
        try:
            rel = predicate_to_relationship_type(pred)
        except ValueError:
            rel = str(pred).strip().lower()
        groups[(normalize_text(subj), rel, normalize_text(obj))].append(t)

    dups: list[dict[str, Any]] = []
    for members in groups.values():
        if len(members) > 1:
            dups.append({
                "subject": members[0].get("subject", ""),
                "predicate": members[0].get("predicate", ""),
                "object": members[0].get("object", ""),
                "count": len(members),
                "predicates": sorted({str(m.get("predicate", "")) for m in members}),
                "sentences": [str(m.get("sentence", "")) for m in members if m.get("sentence")],
            })
    return dups


# ---------------------------------------------------------------------------
# 1) Entity duplicates — embedding-based near-synonym clustering
# ---------------------------------------------------------------------------
_SAME_CONCEPT_PROMPT = """You are de-duplicating concept names in a Trustworthy-AI knowledge graph.
For each pair of terms, decide whether they refer to the SAME concept (synonyms,
spelling variants, or one being an obvious alias of the other) — NOT merely
related concepts. e.g. "Explainability"/"Explicability" = same; "Fairness"/"Bias"
= different (related, not the same).

Return ONLY JSON of this shape:
{"pairs": [{"id": "<id>", "same": true|false}]}

Pairs:
"""


def _candidate_entity_pairs(forms: list[str], *, embed_threshold: float, lexical_threshold: float) -> list[tuple[int, int]]:
    """Cheap candidate pairs from lexical similarity + (optional) embeddings."""
    import difflib

    n = len(forms)
    norms = [normalize_text(f) for f in forms]
    pairs: set[tuple[int, int]] = set()

    # Lexical: high string similarity or substring containment (catches
    # "Explainability"/"Explicability", "Bias"/"Biases", "X"/"X (XAI)").
    for i in range(n):
        for j in range(i + 1, n):
            a, b = norms[i], norms[j]
            if not a or not b:
                continue
            if a in b or b in a or difflib.SequenceMatcher(None, a, b).ratio() >= lexical_threshold:
                pairs.add((i, j))

    # Semantic: embedding cosine above a permissive threshold (LLM filters later).
    try:
        from kbdebugger.subgraph_similarity.encoder import SentenceTransformerEncoder
        model_name = os.getenv("KB_ENCODER_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2").strip()
        device = os.getenv("KB_ENCODER_DEVICE", "").strip() or None
        enc = SentenceTransformerEncoder(model_name=model_name, device=device, normalize=True)
        vecs = enc.encode(forms)
        sims = vecs @ vecs.T
        for i in range(n):
            for j in range(i + 1, n):
                if float(sims[i][j]) >= embed_threshold:
                    pairs.add((i, j))
    except Exception:  # noqa: BLE001 — embeddings are best-effort; lexical still works
        pass

    return sorted(pairs)


def find_entity_duplicates(
    names: Sequence[str],
    *,
    embed_threshold: float = 0.60,
    lexical_threshold: float = 0.82,
    use_llm: bool = True,
) -> list[dict[str, Any]]:
    """
    Detect near-duplicate entity surface forms ("Explainability" vs
    "Explicability"). Two stages so we are both sensitive and precise:

      1. cheap CANDIDATE pairs from lexical similarity + embeddings,
      2. an LLM judge confirms which pairs are genuinely the SAME concept
         (filters out merely-related pairs like Fairness/Bias).

    Returns clusters: [{canonical, members:[...], confirmed_by}].
    """
    # Distinct by normalized key, keeping a representative surface form.
    by_norm: dict[str, str] = {}
    for raw in names:
        s = str(raw or "").strip()
        if s:
            by_norm.setdefault(normalize_text(s), s)
    forms = list(by_norm.values())
    if len(forms) < 2:
        return []

    candidates = _candidate_entity_pairs(forms, embed_threshold=embed_threshold, lexical_threshold=lexical_threshold)
    if not candidates:
        return []

    # LLM confirmation of candidate pairs.
    confirmed: list[tuple[int, int]] = []
    confirmed_by = "llm"
    if use_llm:
        from kbdebugger.llm.model_access import respond
        from kbdebugger.utils import batched
        from kbdebugger.utils.json import ensure_json_object

        idx_by_id = {str(k): pair for k, pair in enumerate(candidates)}
        verdicts: dict[str, bool] = {}
        for batch_ids in batched(list(idx_by_id), 10):
            payload = [{"id": cid, "term_a": forms[idx_by_id[cid][0]], "term_b": forms[idx_by_id[cid][1]]} for cid in batch_ids]
            try:
                resp = respond(_SAME_CONCEPT_PROMPT + json.dumps(payload, ensure_ascii=False),
                               max_tokens=1024, temperature=0.0, json_mode=True)
                parsed = ensure_json_object(resp)
                for v in parsed.get("pairs", []) or []:
                    if isinstance(v, dict) and v.get("id") is not None:
                        verdicts[str(v["id"])] = bool(v.get("same"))
            except Exception:  # noqa: BLE001
                continue
        confirmed = [idx_by_id[cid] for cid in idx_by_id if verdicts.get(cid)]
        # If the LLM judged nothing (e.g. backend unreachable), fall back to
        # only the very-high-confidence lexical pairs so the feature still works.
        if not verdicts:
            confirmed_by = "lexical-fallback"
            import difflib
            norms = [normalize_text(f) for f in forms]
            confirmed = [(i, j) for (i, j) in candidates
                         if difflib.SequenceMatcher(None, norms[i], norms[j]).ratio() >= 0.9]
    else:
        confirmed = candidates
        confirmed_by = "candidates"

    if not confirmed:
        return []

    # Union-find to turn confirmed pairs into clusters.
    parent = list(range(len(forms)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in confirmed:
        parent[find(i)] = find(j)

    groups: dict[int, list[int]] = defaultdict(list)
    for k in range(len(forms)):
        groups[find(k)].append(k)

    clusters: list[dict[str, Any]] = []
    for members in groups.values():
        if len(members) > 1:
            member_forms = [forms[k] for k in members]
            canonical = sorted(member_forms, key=lambda s: (len(s), s))[0]
            clusters.append({"canonical": canonical, "members": member_forms, "confirmed_by": confirmed_by})
    return clusters


# ---------------------------------------------------------------------------
# 2) Faithfulness — LLM judge (catches hallucinated subject/object)
# ---------------------------------------------------------------------------
_FAITHFULNESS_PROMPT = """You are a strict fact-checker for a knowledge-graph extractor.
For each item you are given a source SENTENCE and a TRIPLE (subject, predicate, object)
that was extracted from it. Decide whether the triple is DIRECTLY SUPPORTED by the
sentence — i.e. both the subject and the object actually appear in (or are unambiguously
referred to by) the sentence, and the relation is genuinely stated. If the subject or
object is not grounded in the sentence (a hallucination), mark it unsupported.

Return ONLY a JSON object of this exact shape:
{"verdicts": [{"id": "<id>", "supported": true|false, "reason": "<short reason>"}]}

Items:
"""


def check_faithfulness(
    items: Sequence[dict[str, Any]],
    *,
    batch_size: int = 5,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    """
    LLM faithfulness check. Each item: {id, subject, predicate, object, sentence}.
    Returns the same items annotated with `supported` (True/False/None) + `reason`.
    On LLM/parse failure a triple degrades to supported=None ("could not verify")
    so we never falsely flag a real triple as a hallucination.
    """
    from kbdebugger.llm.model_access import respond
    from kbdebugger.utils import batched
    from kbdebugger.utils.json import ensure_json_object

    out: list[dict[str, Any]] = []
    for batch in batched(list(items), batch_size):
        payload = [
            {
                "id": str(it.get("id")),
                "sentence": str(it.get("sentence", "")),
                "triple": f'({it.get("subject", "")}, {it.get("predicate", "")}, {it.get("object", "")})',
            }
            for it in batch
        ]
        by_id: dict[str, dict[str, Any]] = {}
        try:
            resp = respond(
                _FAITHFULNESS_PROMPT + json.dumps(payload, ensure_ascii=False),
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=True,
            )
            parsed = ensure_json_object(resp)
            for v in parsed.get("verdicts", []) or []:
                if isinstance(v, dict) and v.get("id") is not None:
                    by_id[str(v["id"])] = v
        except Exception as exc:  # noqa: BLE001 — never lose triples on judge failure
            for it in batch:
                out.append({**it, "supported": None, "reason": f"could not verify: {exc}"})
            continue

        for it in batch:
            v = by_id.get(str(it.get("id")))
            if v is None:
                out.append({**it, "supported": None, "reason": "no verdict returned"})
            else:
                out.append({
                    **it,
                    "supported": bool(v.get("supported", True)),
                    "reason": str(v.get("reason", "")),
                })
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_quality_check(
    triples: Sequence[dict[str, Any]],
    *,
    run_faithfulness: bool = True,
) -> dict[str, Any]:
    """
    Run all three checks over the extracted triples.

    Each triple: {subject, predicate, object, sentence?}.
    """
    items = [
        {
            "id": str(i),
            "subject": str(t.get("subject", "")),
            "predicate": str(t.get("predicate", "")),
            "object": str(t.get("object", "")),
            "sentence": str(t.get("sentence", "")),
        }
        for i, t in enumerate(triples)
        if t.get("subject") and t.get("predicate") and t.get("object")
    ]

    faithfulness = check_faithfulness(items) if (run_faithfulness and items) else []
    hallucinated = [f for f in faithfulness if f.get("supported") is False]

    names: list[str] = []
    for it in items:
        names.append(it["subject"])
        names.append(it["object"])
    entity_duplicates = find_entity_duplicates(names)
    relation_duplicates = find_relation_duplicates(items)

    return {
        "summary": {
            "total": len(items),
            "hallucinated": len(hallucinated),
            "unverified": len([f for f in faithfulness if f.get("supported") is None]),
            "entity_clusters": len(entity_duplicates),
            "relation_duplicates": len(relation_duplicates),
        },
        "faithfulness": faithfulness,
        "hallucinated": hallucinated,
        "entity_duplicates": entity_duplicates,
        "relation_duplicates": relation_duplicates,
    }
