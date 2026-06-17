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

## Workflow

```bash
# 0. (LOCAL) push the scorer + dataset + pre-built KGs to the cluster mirror.
bash experiments/MINE/cluster/sync_to_cluster.sh        # → /home/abuali/projects/kbextractor-mine

# 1. (CLUSTER) grab an interactive H100 job from the kggen-eval dir (it owns
#    slurm_pty.sh + the container mounts). The venv auto-activates.
cd /home/abuali/projects/kggen-eval
bash scripts/slurm_pty.sh H100 mine_judge 1 1 1 8 80G 4   # 1×H100, 4h

# 2. (in the job) serve the judge on localhost:8000 — self-backgrounds, waits "READY".
bash scripts/serve_vllm.sh

# 3. (same job) judge BOTH systems against that local vLLM. Free + fast.
cd /home/abuali/projects/kbextractor-mine
bash cluster/score_with_local_vllm.sh                     # ~200 essays × 15 facts
#   → results/{kbextractor,kggen_deepseek}/openai-Qwen-Qwen3-30B-.../

# 4. (LOCAL) pull results back, then extend the report to a 3-judge ablation.
scp -r abuali@login1.pegasus.kl.dfki.de:/home/abuali/projects/kbextractor-mine/results/* \
       experiments/MINE/results/
experiments/MINE/report/.venv/bin/python experiments/MINE/report/make_report.py
```

## Notes

- **Served-model-name must match.** `score_with_local_vllm.sh` defaults its judge
  id to `openai/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`, which equals the
  `--served-model-name` in `serve_vllm.sh`. If you serve a different model, set
  `MINE_JUDGE_MODEL=openai/<served-name>` before step 3.
- **The `openai/` prefix** is just litellm's OpenAI wire format — it routes to
  `/v1/chat/completions`, which vLLM serves. (Nothing to do with OpenAI the vendor.)
- **Idempotent.** Re-running step 3 skips essays already judged by this judge; add
  `--overwrite` to force, `--ids 4,10` to target specific essays.
- **Detached alternative.** Instead of one interactive job, serve as a background
  job (`kggen-eval/scripts/srun_submit.sh H100 serve_qwen ...`), note the printed
  node IP, then run step 3 from elsewhere with
  `MINE_JUDGE_API_BASE=http://<NODE_IP>:8000/v1`.
- **Adding gpt-oss-20b later** (judge #4): download it to `/fscratch/abuali/models`,
  serve with the same vLLM (`vllm serve <dir> --served-model-name gpt-oss-20b`),
  then `MINE_JUDGE_MODEL=openai/gpt-oss-20b bash cluster/score_with_local_vllm.sh`.
  Target H100 (native MXFP4, ~16 GB).
