"""
Cross-document comparison over the knowledge graph.

Everything in this package operates on the provenance layer written at upsert
time (`provenance_records` / `provenance_docs` on relationships):

- `provenance`  — fetch + parse the provenance layer
- `overlap`     — document coverage, assertion overlap, concept matrix
- `alignment`   — SAME_AS concept alignment candidates + reviewer decisions
- `conflicts`   — typed conflict candidates + LLM adjudication
- `ambiguity`   — undefined normative terms, vague language, near-synonyms

The graph generates candidates cheaply; the LLM only adjudicates pairs;
the human reviewer confirms verdicts.
"""
