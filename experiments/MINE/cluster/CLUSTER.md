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

```bash
# 0. (LOCAL) push the scorer + dataset + pre-built KGs to the cluster mirror.
bash experiments/MINE/cluster/sync_to_cluster.sh        # → /home/abuali/projects/kbextractor-mine

# 1. (CLUSTER, in tmux) submit a deferred job: serve Qwen3 + judge both systems,
#    starting at midnight, on H100, max 6h. (8 cpus, 1 gpu, 80G.)
tmux new -s mine
cd /home/abuali/projects/kbextractor-mine
BEGIN=00:00 bash /home/abuali/projects/kggen-eval/scripts/srun_submit.sh \
    H100 mine_judge 8 1 80G 6 cluster/serve_and_score.sh
#   watch:  squeue -u abuali        (PD = waiting for the begin time / a free H100)
#           tail -f /fscratch/abuali/logs/mine_judge_*.log

# 2. (LOCAL, next morning) pull results back, extend the report to a 3-judge ablation.
scp -r abuali@login1.pegasus.kl.dfki.de:/home/abuali/projects/kbextractor-mine/results/* \
       experiments/MINE/results/
experiments/MINE/report/.venv/bin/python experiments/MINE/report/make_report.py
```

## Workflow B — interactive (if you happen to get a node fast)

```bash
bash experiments/MINE/cluster/sync_to_cluster.sh                  # LOCAL
cd /home/abuali/projects/kggen-eval                               # CLUSTER
bash scripts/slurm_pty.sh H100 mine_judge 1 1 1 8 80G 4           # interactive H100
bash scripts/serve_vllm.sh                                        # serve Qwen3, waits READY
cd /home/abuali/projects/kbextractor-mine
bash cluster/score_with_local_vllm.sh                             # judge both systems
#   → results/{kbextractor,kggen_deepseek}/openai-Qwen-Qwen3-30B-.../
```

## Notes

- **Served-model-name must match.** `score_with_local_vllm.sh` defaults its judge
  id to `openai/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`, which equals the
  `--served-model-name` in `serve_vllm.sh`. If you serve a different model, set
  `MINE_JUDGE_MODEL=openai/<served-name>` before step 3.
- **The `openai/` prefix** is just litellm's OpenAI wire format — it routes to
  `/v1/chat/completions`, which vLLM serves. (Nothing to do with OpenAI the vendor.)
- **Idempotent.** Re-running skips essays already judged by this judge; add
  `--overwrite` to force, `--ids 4,10` to target specific essays. Args after the
  task script flow through: `… cluster/serve_and_score.sh --limit 5` for a smoke test.
- **Don't submit `serve_vllm.sh` directly to `srun_submit`.** It backgrounds vLLM
  and returns, so the job would end (and kill vLLM) before scoring. `serve_and_score.sh`
  exists precisely to keep one job alive across serve **and** score.
- **`BEGIN=` accepts** `HH:MM`, `now+4hours`, or `2026-06-18T02:00:00`; the job sits
  `PD` in `squeue` until then. Run it from `tmux`/`screen` so the nohup'd srun client
  outlives your SSH session.
- **Adding gpt-oss-20b later** (judge #4): download it to `/fscratch/abuali/models`,
  then reuse the same job with overrides —
  `MODEL_DIR=/fscratch/abuali/models/gpt-oss-20b SERVED_MODEL_NAME=gpt-oss-20b
  BEGIN=00:00 bash …/kggen-eval/scripts/srun_submit.sh H100 mine_gptoss 8 1 80G 6 cluster/serve_and_score.sh`.
  Target H100 (native MXFP4, ~16 GB).
