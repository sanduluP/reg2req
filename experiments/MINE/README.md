# MINE-1 evaluation for KBExtractor

Quantitatively evaluate the KBExtractor knowledge graph against the **MINE-1
(Knowledge Retention)** benchmark from the KGGen paper
([arXiv:2502.09956](https://arxiv.org/abs/2502.09956), NeurIPS 2025), and compare
it head-to-head with KGGen / GraphRAG / OpenIE.

## What MINE-1 measures

For each of 100 short articles the benchmark ships **15 known facts**. The
extractor-under-test builds a KG from the article; then, for each fact, a fixed
retriever (`all-MiniLM-L6-v2`, top-k nearest nodes + 2-hop expansion) pulls a
context subgraph and an LLM judge decides **1/0**: is the fact recoverable from
that context? **MINE-1 = % of the 15 facts scored 1, averaged over the 100
articles.** Retrieval and judge are held constant across all systems, so the
benchmark isolates *graph-content quality*.

## Why two stages / two environments

`kg_gen` and `kbdebugger` pin conflicting versions of `dspy` /
`sentence-transformers`, so we do **not** import both in one process:

| Stage | Script | Environment | Needs |
|-------|--------|-------------|-------|
| 0 | `fetch_dataset.py` | kg-gen venv | `datasets` |
| 1 | `build_kbextractor_kgs.py` | **KBExtraction venv** | `kbdebugger` + DFKI LLM endpoint |
| 2 | `score_kgs.py` | **kg-gen venv** | `kg_gen`, `dspy`, judge LLM |

Stage 1 writes KGs in KGGen's `Graph` JSON format; Stage 2 loads them (ours and
the dataset's pre-generated baselines) and scores them through the identical
retrieve + judge pipeline.

Both stages are **idempotent and resumable**: Stage 1 skips essays whose
`kg_<id>.json` already exists, and Stage 2 skips essays already scored — it
reuses `results_<id>.json` and fires **no judge calls** for them. Scoring output
is keyed by `(system, judge)`:
`results/<system>/<judge-slug>/results_<id>.json`, so a deepseek run and a GPT-5
run coexist without clobbering each other (and the cache only reuses verdicts
from the *same* judge). Pass `--overwrite` to force re-judging.

## Usage

The wrapper scripts in [`scripts/MINE/`](../../scripts/MINE/) pick the right
virtualenv and fill in all paths — run them from anywhere, no `source`, no long
absolute paths. Extra flags pass straight through to the underlying Python.

```bash
# --- Stage 0: cache the dataset locally (run once) ---
scripts/MINE/_0_fetch_dataset.sh

# --- Stage 1: build KBExtractor KGs (DFKI network) ---
scripts/MINE/_1_build_kgs.sh --limit 1     # smoke-test one essay; drop --limit for all 100

# --- Stage 2: score a system ---
scripts/MINE/_2_score.sh kbextractor --limit 1
scripts/MINE/_2_score.sh kggen             # free baseline (dataset's pre-generated KG)
scripts/MINE/_2_score.sh graphrag
scripts/MINE/_2_score.sh openie
```

The kg-gen venv defaults to `/home/faris/code/DSA_HiWi/kg-gen/.venv`; override with
`KGGEN_VENV=/path/to/.venv` (and `KB_VENV=` for the KBExtraction venv) if your
layout differs. The underlying `*.py` scripts can still be called directly if
you prefer.

Every wrapper tees its combined stdout+stderr to a timestamped log under
`logs/MINE/` (git-ignored), e.g. `logs/MINE/build_kgs_20260616_121033.log`,
`logs/MINE/score_kbextractor_20260616_121033.log`. The console output is
unchanged; the file is just for traceability/debugging. The wrapper still exits
with the Python script's real status (not `tee`'s).

## Judge / backbone configuration (env, Stage 2)

| Var | Default | Meaning |
|-----|---------|---------|
| `MINE_JUDGE_MODEL` | `openai/gpt-5` | litellm/DSPy model id for the judge |
| `MINE_JUDGE_API_BASE` | *(none)* | OpenAI-compatible base URL (e.g. the DFKI endpoint for an on-prem judge) |
| `MINE_JUDGE_API_KEY_ENV` | `OPENAI_API_KEY` | env var holding the judge API key |

For a fully on-prem, zero-cost run, point the judge at the DFKI `deepseek-r1:32b`
service via `MINE_JUDGE_MODEL=openai/deepseek-r1:32b` +
`MINE_JUDGE_API_BASE=http://serv-3306.kl.dfki.de:8000/v1`.

## Notes / open items

- Stage 1 runs KBExtractor in **MINE mode**: extract-everything, no keyword gate,
  no similarity/novelty filter, no human, no Neo4j. Triplet extraction uses
  `neutral_mode=True` — a free-form predicate prompt
  (`src/kbdebugger/prompts/triplets_batch_neutral.txt`) with the standards
  controlled-vocabulary, deontic/modality, and schema-grounding layers switched
  off. This is what fairly reflects KBExtractor's extraction core against
  KGGen/OpenIE's free-form predicates: on general-topic essays the standards
  ontology (`DEFAULT_ALLOWED_PREDICATES`) would force-fit or *drop* facts the
  prompt can't map to an allowed predicate, depressing the recall MINE measures.
  The standards prompt (`triplets_batch.txt`) is untouched and remains the
  default for the KBDebugger pipeline.
- The dataset's pre-generated `kggen`/`graphrag`/`openie` KGs were built on the
  paper's backbone (GPT-4o/Gemini). Document them as "published baselines under
  our judge"; re-run KGGen on the DFKI backbone only if a same-backbone row is
  needed.
