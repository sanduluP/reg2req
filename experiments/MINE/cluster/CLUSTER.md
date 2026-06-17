# MINE judging on the DFKI Pegasus cluster

**Why:** add a third, open-source judge to the MINE-1 ablation (alongside the
on-prem `deepseek-r1:32b` and the GPT-5 paper-parity judge) at **zero API cost**,
by serving the judge with vLLM on a cluster GPU and judging locally on the node.

**Judge:** `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` — already downloaded under
`/fscratch/abuali/models` and already served by `kggen-eval/scripts/serve_vllm.sh`.
A different family (Qwen3) and mode (instruct, no `<think>`) from `deepseek-r1`,
so it's a genuine robustness axis. **Partition:** H100.

**No new infra.** The `kggen-eval` cluster venv (`/fscratch/abuali/venvs/kggen-eval`)
already has `kg_gen` + `dspy` + the cached `all-MiniLM-L6-v2` retriever, so
`score_kgs.py` runs there unchanged. The `25.02` NVIDIA container (CUDA 12.8)
already covers Ampere→Hopper→Blackwell — nothing to rebuild.

## Workflow A — detached / overnight (recommended; H100 queue is long)

One self-contained job serves the judge **and** scores, so it survives without an
interactive shell. `BEGIN=` holds it PENDING until a quiet hour. Launch inside
`tmux`/`screen` on the login node so the nohup'd srun client survives logout.

All knobs are hardcoded — you don't pass arguments. `run_experiment.sh` serves +
scores + frees the GPU; `submit_overnight.sh` queues it (partition/time/BEGIN
hardcoded inside).

```bash
# 0. (LOCAL) sync the MINE bundle + the BEGIN= update in kggen-eval.
bash experiments/MINE/cluster/sync_to_cluster.sh                        # → kbextractor-mine
( cd /home/faris/code/DSA_HiWi/kggen-eval && bash scripts/sync_to_cluster.sh )

# 1. (CLUSTER, in tmux) ONE command — queues the whole run for midnight on H100.
tmux new -s mine
bash /home/abuali/projects/kbextractor-mine/cluster/submit_overnight.sh
#   detach: Ctrl-b d   ·   watch: squeue -u abuali  (PD until 00:00 / a free H100)
#                                  tail -f /fscratch/abuali/logs/mine_judge_*.log

# 2. (LOCAL, next morning) pull results back, extend the report to a 3-judge ablation.
scp -r abuali@login1.pegasus.kl.dfki.de:/home/abuali/projects/kbextractor-mine/results/* \
       experiments/MINE/results/
experiments/MINE/report/.venv/bin/python experiments/MINE/report/make_report.py
```

Run it **now** instead of midnight: `BEGIN=now bash cluster/submit_overnight.sh`.
Smoke-test first: temporarily set `SCORE_ARGS="--limit 3"` in `score_with_local_vllm.sh`.

## Workflow B — interactive (if you happen to get a node fast)

```bash
bash experiments/MINE/cluster/sync_to_cluster.sh                  # LOCAL
cd /home/abuali/projects/kggen-eval                               # CLUSTER
bash scripts/slurm_pty.sh H100 mine_judge 1 1 1 8 80G 4           # interactive H100
cd /home/abuali/projects/kbextractor-mine
bash cluster/run_experiment.sh                                   # serve + score + stop, all-in-one
#   → results/{kbextractor,kggen_deepseek}/openai-Qwen-Qwen3-30B-.../
```

## Notes

- **The knobs live in two files.** Which LLM to serve → `run_experiment.sh`
  (`MODEL_DIR`/`SERVED_MODEL_NAME`/`PORT`). Which SLURM slot → `submit_overnight.sh`
  (`PARTITION`/`MEM`/`HOURS`/`BEGIN`). The judge id is derived as
  `openai/<SERVED_MODEL_NAME>`, so the scorer always matches what's served.
- **The `openai/` prefix** is just litellm's OpenAI wire format — it routes to
  `/v1/chat/completions`, which vLLM serves. (Nothing to do with OpenAI the vendor.)
- **Idempotent.** Re-running skips essays already judged by this judge; for a
  forced re-judge or a smoke test, edit `SCORE_ARGS` in `score_with_local_vllm.sh`
  (`--overwrite`, `--ids 4,10`, `--limit 3`).
- **Why `run_experiment.sh` and not `serve_vllm.sh` as the task.** A submitted
  task that just serves would background vLLM and *return*, ending the job (and
  killing vLLM) before scoring. `run_experiment.sh` keeps one job alive across
  serve **and** score, then frees the GPU on exit.
- **`BEGIN=` accepts** `HH:MM`, `now`, `now+4hours`, or `2026-06-18T02:00:00`; the
  job sits `PD` in `squeue` until then. Run from `tmux`/`screen` so the nohup'd
  srun client outlives your SSH session.
- **Adding gpt-oss-20b later** (judge #4): download it to `/fscratch/abuali/models`,
  then just point the two `MODEL_DIR`/`SERVED_MODEL_NAME` lines in `run_experiment.sh`
  at it (and bump `JOB_NAME` in `submit_overnight.sh`). Target H100 (native MXFP4, ~16 GB).
