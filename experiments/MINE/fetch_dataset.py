"""Stage 0: cache the MINE-1 evaluation dataset to a local JSON file.

Run in the kg-gen virtualenv (which has ``datasets``). Decouples later stages
from the network and from the ``datasets`` dependency.

Each output record holds the essay text, the 15 facts to verify, and the
dataset's pre-generated baseline KGs::

    {"id", "essay_content", "generated_queries",
     "kggen", "graphrag_kg", "openie_kg"}
     
/home/faris/code/DSA_HiWi/kg-gen/.venv/bin/python \
  /home/faris/code/RPTU/DSA/KBExtraction/experiments/MINE/fetch_dataset.py \
  --out /home/faris/code/RPTU/DSA/KBExtraction/experiments/MINE/data/mine.json
"""
from __future__ import annotations

import argparse
import json
import os

from datasets import load_dataset

DATASET = "josancamon/kg-gen-MINE-evaluation-dataset"
_FIELDS = ("essay_content", "generated_queries", "kggen", "graphrag_kg", "openie_kg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="experiments/MINE/data/mine.json")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    dataset = load_dataset(DATASET)[args.split]
    records = []
    for i, item in enumerate(dataset.to_list()):
        records.append({"id": i, **{field: item.get(field) for field in _FIELDS}})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} records → {args.out}")


if __name__ == "__main__":
    main()
