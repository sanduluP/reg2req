# MINE judging on the DFKI Pegasus cluster

Add an open-source judge to the MINE-1 ablation (alongside the on-prem
`deepseek-r1:32b` and the GPT-5 paper-parity judge) at **zero API cost**, by
serving the judge with vLLM on a cluster GPU and judging locally on the node.

**Self-contained.** Everything lives in this repo's `experiments/MINE/cluster/`
(synced to `/home/abuali/projects/kbextractor-mine` on the cluster). It does NOT
reference any other repo. Environments live at neutral, **reusable** paths under
`/fscratch/abuali/venvs` — no repo borrows another's venv.

**Judge:** `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (under `/fscratch/abuali/models`),
a different family/mode (Qwen3 instruct, no `<think>`) than `deepseek-r1` — a
genuine robustness axis. **Partition:** H100.

## The two environments (created once by `setup_envs.sh`)

| Path | Purpose |
|---|---|
| `/fscratch/abuali/venvs/vllm` | **reusable** — just vLLM; ANY repo serves a model with it |
| `/fscratch/abuali/venvs/kbextractor-mine` | this repo's scorer (`kg_gen` + `dspy` + `all-MiniLM-L6-v2`) |

Two envs because serving (the vLLM server) and scoring (`score_kgs.py`) are
separate programs that only talk over HTTP — so they don't share an interpreter,
and the vLLM env stays pure for reuse. Both are built inside the shared `25.02`
NVIDIA container (CUDA 12.8 → covers Ampere/Hopper/Blackwell).

## The pieces (all in `cluster/`)

| Script | Role |
|---|---|
| `srun_submit.sh` | GENERAL submitter (`BEGIN=` deferred start). Resources + a task script. |
| `setup_envs.sh` | ONE-TIME: build the two venvs above. |
| `run_experiment.sh` | TASK: serve vLLM → wait ready → score both systems → free the GPU. Model knobs hardcoded at top. |
| `score_with_local_vllm.sh` | the scorer ("client"). Scorer knobs (`JUDGE_WORKERS`, `SYSTEMS`, `SCORE_ARGS`). |
| `sync_to_cluster.sh` | push scorer + dataset + KGs + these scripts to the cluster. |

## Workflow

```bash
# 0. (LOCAL) sync this bundle to the cluster mirror.
bash experiments/MINE/cluster/sync_to_cluster.sh        # → /home/abuali/projects/kbextractor-mine

# 1. (CLUSTER) ONE-TIME: build the two venvs on a CPU node (~15-25 min).
cd /home/abuali/projects/kbextractor-mine
bash cluster/srun_submit.sh batch setup_envs 8 0 32G 2 cluster/setup_envs.sh
tail -f /fscratch/abuali/logs/setup_envs_*.log          # wait for "✅ done."

# 2. (CLUSTER, in screen) submit the judging task for midnight on H100.
screen -S mine        # (tmux new -s mine works too)
BEGIN=00:00 bash cluster/srun_submit.sh H100 mine_judge 8 1 80G 6 cluster/run_experiment.sh
#   Ctrl-a d to detach (screen -r mine to return)
#   watch:  squeue -u abuali  (PD until 00:00 / a free H100)
#           tail -f /fscratch/abuali/logs/mine_judge_*.log

# 3. (LOCAL, next morning) pull results back, extend the report to a 3-judge ablation.
scp -r abuali@login1.pegasus.kl.dfki.de:/home/abuali/projects/kbextractor-mine/results/* \
       experiments/MINE/results/
experiments/MINE/report/.venv/bin/python experiments/MINE/report/make_report.py
```

Run step 2 **now** instead of midnight: drop `BEGIN=00:00` (start ASAP) or `BEGIN=now`.
Smoke-test first: set `SCORE_ARGS="--limit 3"` in `score_with_local_vllm.sh`.

## Notes

- **Where the knobs live.** Which LLM to serve → `run_experiment.sh`
  (`MODEL_DIR`/`SERVED_MODEL_NAME`/`PORT`, plus the env paths). Scorer behaviour →
  `score_with_local_vllm.sh` (`JUDGE_WORKERS`/`SYSTEMS`/`SCORE_ARGS`). SLURM
  resources → the `srun_submit.sh` line. The judge id is derived as
  `openai/<SERVED_MODEL_NAME>`, so the scorer always matches what's served.
- **The `openai/` prefix** is just litellm's OpenAI wire format — it routes to
  `/v1/chat/completions`, which vLLM serves. (Nothing to do with OpenAI the vendor.)
- **Idempotent.** Re-running skips essays already judged by this judge; for a forced
  re-judge or smoke test, set `SCORE_ARGS` (`--overwrite`, `--ids 4,10`, `--limit 3`).
- **One task does serve + score.** A task that only serves would background vLLM and
  return, ending the job (killing vLLM) before scoring. `run_experiment.sh` keeps one
  job alive across both, then frees the GPU on any exit (incl. SLURM time-limit).
- **`BEGIN=` accepts** `HH:MM`, `now`, `now+4hours`, or `2026-06-18T02:00:00`; the job
  sits `PD` in `squeue` until then. Run from `screen`/`tmux` so the nohup'd srun
  client outlives your SSH session.
- **Reuse the vLLM env elsewhere.** Any future repo serves a model with
  `/fscratch/abuali/venvs/vllm/bin/vllm serve <model-dir> …` — no need to reinstall.
- **Adding gpt-oss-20b later** (judge #4): download it to `/fscratch/abuali/models`,
  point the `MODEL_DIR`/`SERVED_MODEL_NAME` lines in `run_experiment.sh` at it, pick a
  fresh job name on the submit line. Same vLLM env. Target H100 (native MXFP4, ~16 GB).
