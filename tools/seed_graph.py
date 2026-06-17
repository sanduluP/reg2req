"""
Seed the Trustworthy-AI knowledge graph from the curated seed file.

Connection details are read from `.env` (NEO4J_URI / NEO4J_USERNAME /
NEO4J_PASSWORD) by GraphStore, exactly like the rest of the app.

Usage
-----
    PYTHONPATH=src python tools/seed_graph.py            # idempotent upsert
    PYTHONPATH=src python tools/seed_graph.py --clear     # replace seed edges first
    PYTHONPATH=src python tools/seed_graph.py --dump-json # only regenerate the JSON artifact
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the Trustworthy-AI knowledge graph.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the WHOLE graph first, then build the baseline (Initialize-graph behavior).",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove only previously seeded edges (and orphaned nodes) before seeding.",
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Only regenerate data/seed/trustworthy_ai_seed.json from the .txt and exit.",
    )
    args = parser.parse_args()

    from kbdebugger.graph.seed import dump_seed_json, seed_knowledge_graph

    if args.dump_json:
        path = dump_seed_json()
        print(f"Wrote seed JSON artifact: {path}")
        return 0

    try:
        summary = seed_knowledge_graph(
            reset_all=args.reset, clear_existing=args.clear, pretty_print=True
        )
    except Exception as exc:  # noqa: BLE001 — surface a clear message for the CLI
        print(f"❌ Seeding failed: {exc}", file=sys.stderr)
        print(
            "Check that NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD are set in .env "
            "and the database is reachable.",
            file=sys.stderr,
        )
        return 1

    print("🌱 Seed complete:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
