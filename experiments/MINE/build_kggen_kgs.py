"""Stage 1 (KGGen baseline): build a KGGen KG per essay on the SAME backbone.

Run in the **kg-gen virtualenv** (has ``kg_gen`` + ``dspy``). Generates a KGGen
knowledge graph for each essay with ``kg_gen.generate`` pointed at the DFKI
deepseek endpoint — the same backbone KBExtractor uses. This gives a fair,
same-backbone, correctly-aligned KGGen baseline (the dataset's pre-generated
``kggen`` graphs are GPT-4o/Gemini-built AND misaligned with the essays from
row 18 on, so we cannot use them).

KGGen forces ``model_type="responses"`` for any ``openai/``-prefixed model, which
the DFKI chat endpoint does not support — so we route via litellm's
``hosted_vllm/`` prefix, which hits ``/v1/chat/completions`` and keeps
``model_type="chat"``.

Output is KGGen's native ``Graph`` JSON (already S-P-O), loadable by the scorer
via ``KGGen.from_dict``.

    .../kg-gen/.venv/bin/python experiments/MINE/build_kggen_kgs.py \
        --data experiments/MINE/data/mine.json \
        --out-dir experiments/MINE/kgs/kggen_deepseek
"""
from __future__ import annotations

import argparse
import json
import os
import time

from kg_gen import KGGen

DEFAULT_MODEL = "hosted_vllm/deepseek-r1:32b"
DEFAULT_API_BASE = "http://serv-3306.kl.dfki.de:8000/v1"


def _graph_to_dict(graph) -> dict[str, list]:
    return {
        "entities": sorted(graph.entities),
        "edges": sorted(graph.edges),
        "relations": [list(r) for r in graph.relations],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="experiments/MINE/data/mine.json")
    parser.add_argument("--out-dir", default="experiments/MINE/kgs/kggen_deepseek")
    parser.add_argument("--model", default=os.getenv("KGGEN_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-base", default=os.getenv("KGGEN_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--api-key", default=os.getenv("KGGEN_API_KEY", "dummy"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", default=None, help="comma-separated essay ids to build")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature (default 0.0, deterministic). Raise to recover essays "
        "where greedy decoding emits schema-invalid (object-less) relations that KGGen "
        "discards wholesale — at temp>0 the sampling differs and may produce valid triples.",
    )
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if args.ids:
        wanted = {int(x) for x in args.ids.split(",") if x.strip()}
        records = [r for r in records if r["id"] in wanted]
    elif args.limit is not None:
        records = records[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    kg = KGGen(model=args.model, api_base=args.api_base, api_key=args.api_key, temperature=args.temperature)
    print(f"🧬 [KGGen Stage 1] model={args.model} api_base={args.api_base} temp={args.temperature} → {args.out_dir}")

    built = skipped = failed = 0
    for idx, record in enumerate(records, start=1):
        essay_id = record["id"]
        out_path = os.path.join(args.out_dir, f"kg_{essay_id}.json")
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            print(f"[{idx}/{len(records)}] ↩︎ skip id={essay_id} (exists)")
            continue

        essay = record.get("essay_content") or ""
        if not essay.strip():
            print(f"[{idx}/{len(records)}] ⚠️ skip id={essay_id} (empty essay)")
            continue

        graph_dict = None
        for attempt in range(1, args.retries + 1):
            try:
                started = time.perf_counter()
                graph = kg.generate(input_data=essay)
                graph_dict = _graph_to_dict(graph)
                elapsed = time.perf_counter() - started
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[{idx}/{len(records)}] ⚠️ id={essay_id} attempt {attempt}/{args.retries} failed: {exc}")
                time.sleep(min(2.0 * attempt, 10.0))

        if graph_dict is None:
            failed += 1
            print(f"[{idx}/{len(records)}] ❌ id={essay_id} failed after {args.retries} attempts — NOT writing (retry next run)")
            continue

        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(graph_dict, handle, ensure_ascii=False, indent=2)
        built += 1
        print(
            f"[{idx}/{len(records)}] ✓ id={essay_id} "
            f"entities={len(graph_dict['entities'])} relations={len(graph_dict['relations'])} "
            f"elapsed_s={elapsed:.1f}"
        )

    print(f"\n=== KGGen Stage 1 done: built={built} skipped={skipped} failed={failed} ===")


if __name__ == "__main__":
    main()
