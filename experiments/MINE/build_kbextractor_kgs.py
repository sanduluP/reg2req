"""Stage 1: build a KBExtractor KG per MINE essay and save it in KGGen format.

Run in the **KBExtraction virtualenv**, on the DFKI network (the LLM backend is
configured via the repo's ``.env`` / ``MODEL_*`` vars)::

    PYTHONPATH=src venv/bin/python experiments/MINE/build_kbextractor_kgs.py \
        --data experiments/MINE/data/mine.json \
        --out-dir experiments/MINE/kgs/kbextractor \
        --limit 1

Writes ``<out-dir>/kg_<id>.json`` per essay (KGGen ``Graph`` dict). Idempotent:
already-built KGs are skipped unless ``--overwrite`` is given.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kb_adapter import build_kg_from_essay  # noqa: E402


def _load_records(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="experiments/MINE/data/mine.json")
    parser.add_argument("--out-dir", default="experiments/MINE/kgs/kbextractor")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N essays (smoke test)")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    records = _load_records(args.data)
    if args.limit is not None:
        records = records[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    total = len(records)
    for idx, record in enumerate(records, start=1):
        essay_id = record.get("id", idx - 1)
        out_path = os.path.join(args.out_dir, f"kg_{essay_id}.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[{idx}/{total}] ↩︎ skip id={essay_id} (exists)")
            continue

        essay = record.get("essay_content") or ""
        if not essay.strip():
            print(f"[{idx}/{total}] ⚠️ skip id={essay_id} (empty essay)")
            continue

        started = time.perf_counter()
        graph_dict = build_kg_from_essay(
            essay,
            batch_size=args.batch_size, # tradeoff: smaller batches use less memory but more API calls (and thus more time)
            parallel=not args.no_parallel,
            max_workers=args.max_workers, # only relevant if parallel=True; tradeoff: more workers can speed up processing but also increase memory usage and API rate limits
        )
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(graph_dict, handle, ensure_ascii=False, indent=2)

        elapsed = time.perf_counter() - started
        print(
            f"[{idx}/{total}] ✓ id={essay_id} "
            f"entities={len(graph_dict['entities'])} edges={len(graph_dict['edges'])} "
            f"relations={len(graph_dict['relations'])} elapsed_s={elapsed:.1f} → {out_path}"
        )

    print(f"Done. KGs in {args.out_dir}")


if __name__ == "__main__":
    main()
