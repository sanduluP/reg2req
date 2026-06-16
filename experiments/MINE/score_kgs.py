"""Stage 2: score KGs against MINE-1 using KGGen's retriever + an LLM judge.

Run in the **kg-gen virtualenv** (has ``kg_gen`` + ``dspy``). The retrieval
(``all-MiniLM-L6-v2``, top-k nearest nodes + 2-hop expansion) and the binary
judge are byte-for-byte the same logic KGGen uses, so every system is scored
identically.

    uv run python .../score_kgs.py --system kbextractor \
        --kgs-dir .../kgs/kbextractor --data .../mine.json --out-dir .../results

``--system kggen|graphrag|openie`` scores the dataset's pre-generated baseline
KGs instead (no ``--kgs-dir`` needed).

Judge/backbone via env: ``MINE_JUDGE_MODEL`` (default ``openai/gpt-5``),
``MINE_JUDGE_API_BASE``, ``MINE_JUDGE_API_KEY_ENV`` (default ``OPENAI_API_KEY``).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import mean
from typing import Any

import dspy
from kg_gen.kg_gen import KGGen
from kg_gen.models import Graph

_BASELINE_FIELD = {"kggen": "kggen", "graphrag": "graphrag_kg", "openie": "openie_kg"}


# --- judge -----------------------------------------------------------------
class EvaluateResponse(dspy.Signature):
    """Determine whether the context contains the information stated in the correct answer. Respond with 1 if yes, 0 if no."""

    context: str = dspy.InputField(desc="The context to evaluate")
    correct_answer: str = dspy.InputField(desc="The correct answer to check for")
    evaluation: int = dspy.OutputField(desc="1 if context contains the correct answer, 0 otherwise")


def _configure_judge() -> None:
    model = os.getenv("MINE_JUDGE_MODEL", "openai/gpt-5")
    api_base = os.getenv("MINE_JUDGE_API_BASE") or None
    api_key = os.getenv(os.getenv("MINE_JUDGE_API_KEY_ENV", "OPENAI_API_KEY"))

    kwargs: dict[str, Any] = {"model": model, "api_key": api_key, "api_base": api_base}
    if "gpt-5" in model:  # gpt-5 family requires these
        kwargs.update(temperature=1.0, max_tokens=16000, reasoning={"effort": "high"})
    else:
        kwargs.update(temperature=0.0, max_tokens=4000)
    dspy.configure(lm=dspy.LM(**kwargs))


_JUDGE_MAX_RETRIES = 4


def _judge(judge: dspy.Module, fact: str, context: str) -> tuple[int, bool]:
    """Judge one (fact, context). Returns ``(score, failed)``.

    ``failed=True`` means the judge *call itself* errored (e.g. the endpoint went
    down) even after retries — which is distinct from a legitimate 0. The caller
    uses this to refuse to cache a corrupted essay, so a transient outage doesn't
    poison the results with spurious zeros that the idempotent cache would then
    skip forever.
    """
    last_err: Exception | None = None
    for attempt in range(1, _JUDGE_MAX_RETRIES + 1):
        try:
            return int(judge(context=context, correct_answer=fact).evaluation), False
        except Exception as e:
            last_err = e
            time.sleep(min(2.0 * attempt, 10.0))
    print(f"[judge] ❌ call failed after {_JUDGE_MAX_RETRIES} attempts: {last_err}")
    return 0, True


# --- graph loading ---------------------------------------------------------
def _load_graph(system: str, record: dict, kgs_dir: str | None) -> Graph | None:
    if system == "kbextractor":
        path = os.path.join(kgs_dir, f"kg_{record['id']}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return KGGen.from_dict(json.load(handle))

    data = record.get(_BASELINE_FIELD[system])
    return KGGen.from_dict(data) if data else None


def _load_cached_accuracy(out_dir: str, essay_id: Any) -> float | None:
    """Recover an essay's accuracy from a prior run's results file, or None.

    Recomputes from the stored per-fact verdicts rather than trusting the
    trailing ``accuracy`` string, so a malformed/partial file is treated as a
    cache miss and gets re-judged.
    """
    path = os.path.join(out_dir, f"results_{essay_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None
    facts = [x for x in data if isinstance(x, dict) and "fact" in x]
    if not facts:
        return None
    correct = sum(int(x.get("evaluation", 0)) for x in facts)
    return correct / len(facts)


# --- main ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=["kbextractor", "kggen", "graphrag", "openie"])
    parser.add_argument("--data", default="experiments/MINE/data/mine.json")
    parser.add_argument("--kgs-dir", default=None, help="required for --system kbextractor")
    parser.add_argument("--out-dir", default="experiments/MINE/results")
    parser.add_argument("--top-k", type=int, default=8) # same as KGGen's default for retrieval
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--ids",
        default=None,
        help="comma-separated essay ids to score (e.g. 4,10,20). Targets specific essays — e.g. after a rebuild — instead of the first --limit records.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-judge essays even if results_<id>.json already exists (default: reuse cached results, no judge calls)",
    )
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=1,
        help="parallel judge calls per essay (default 1 = serial). The 15 facts of an essay are judged concurrently; raise cautiously for remote endpoints.",
    )
    args = parser.parse_args()

    if args.system == "kbextractor" and not args.kgs_dir:
        parser.error("--kgs-dir is required for --system kbextractor")

    with open(args.data, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if args.ids:
        wanted = {int(x) for x in args.ids.split(",") if x.strip()}
        records = [r for r in records if r["id"] in wanted]
    elif args.limit is not None:
        records = records[: args.limit]

    _configure_judge()
    judge = dspy.ChainOfThought(EvaluateResponse) # this means the judge will use CoT prompting, which is important for complex retrievals where the context may not be verbatim but still contains the necessary information. And `EvaluateResponse` is designed to be a simple binary classification, so the CoT can help the model reason through the context before arriving at a 0 or 1.
    # KGGen instance is used only as a retrieval utility (no generation happens).
    kggen = KGGen(model="openai/gpt-4o", retrieval_model="all-MiniLM-L6-v2")

    # Key results by (system, judge) so different judges don't collide and the
    # cache only reuses verdicts from the *same* judge.
    judge_model = os.getenv("MINE_JUDGE_MODEL", "openai/gpt-5")
    judge_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", judge_model).strip("-")
    out_dir = os.path.join(args.out_dir, args.system, judge_slug)
    os.makedirs(out_dir, exist_ok=True)
    print(f"📂 results dir: {out_dir}")

    per_essay_accuracy: list[float] = []
    for idx, record in enumerate(records, start=1):
        essay_id = record["id"]

        if not args.overwrite:
            cached = _load_cached_accuracy(out_dir, essay_id)
            if cached is not None:
                per_essay_accuracy.append(cached)
                print(
                    f"[{idx}/{len(records)}] ⏩ cached id={essay_id} "
                    f"accuracy={cached * 100:.1f}% (no judge calls)"
                )
                continue

        facts = record.get("generated_queries") or []
        graph = _load_graph(args.system, record, args.kgs_dir)
        if graph is None or not facts:
            print(f"[{idx}/{len(records)}] ⚠️ skip id={essay_id} (no graph or no facts)")
            continue

        nx_graph = kggen.to_nx(graph) # What is nx? NetworkX, a Python library for working with graphs. Here, we're converting the KGGen graph into a NetworkX graph format, which is what the retrieval method expects.
        node_embeddings, _ = kggen.generate_embeddings(nx_graph)

        # Retrieve each fact's context first (local, cheap): top-k nearest nodes +
        # their 2-hop neighborhood via cosine similarity in all-MiniLM-L6-v2 space.
        fact_contexts = [
            (fact, kggen.retrieve(fact, node_embeddings, nx_graph, k=args.top_k)[-1])
            for fact in facts
        ]

        # Judge each (fact, context) — the LLM calls, optionally concurrent.
        if args.judge_workers > 1:
            with ThreadPoolExecutor(max_workers=args.judge_workers) as pool:
                verdicts = list(pool.map(lambda fc: _judge(judge, fc[0], fc[1]), fact_contexts))
        else:
            verdicts = [_judge(judge, fact, ctx) for fact, ctx in fact_contexts]

        # If any judge call failed (e.g. endpoint outage), do NOT write/cache this
        # essay — leave it unwritten so a later run re-judges it cleanly, instead
        # of poisoning the cache with spurious 0s the idempotent skip would keep.
        n_failed = sum(failed for _, failed in verdicts)
        if n_failed:
            print(
                f"[{idx}/{len(records)}] ⚠️ id={essay_id}: {n_failed}/{len(facts)} judge "
                f"calls failed — NOT caching (will be retried on the next run)"
            )
            continue

        per_fact = []
        correct = 0
        for (fact, context_text), (score, _failed) in zip(fact_contexts, verdicts):
            correct += score
            per_fact.append({"fact": fact, "retrieved_context": context_text, "evaluation": score})

        accuracy = correct / len(facts) # Final Average Accuracy for this essay: number of facts correctly supported by the retrieved context, divided by total number of facts.
        per_essay_accuracy.append(accuracy)
        with open(os.path.join(out_dir, f"results_{essay_id}.json"), "w", encoding="utf-8") as handle:
            json.dump(per_fact + [{"accuracy": f"{accuracy * 100:.2f}%"}], handle, ensure_ascii=False, indent=2)

        print(
            f"[{idx}/{len(records)}] ✓ id={essay_id} "
            f"nodes={nx_graph.number_of_nodes()} edges={nx_graph.number_of_edges()} "
            f"accuracy={accuracy * 100:.1f}%"
        )

    mine_score = mean(per_essay_accuracy) if per_essay_accuracy else 0.0
    summary = {
        "system": args.system,
        "judge_model": judge_model,
        "essays_scored": len(per_essay_accuracy),
        "mine_1_score": f"{mine_score * 100:.2f}%",
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"\n=== MINE-1 [{args.system}] = {summary['mine_1_score']} over {summary['essays_scored']} essays ===")


if __name__ == "__main__":
    main()
