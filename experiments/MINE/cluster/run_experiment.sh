#!/bin/bash
# ---------------------------------------------------------------------------
# run_experiment.sh — UNATTENDED MINE judging: serve vLLM → wait until ready →
# score BOTH systems → stop the server (release the GPU). You pass NO arguments;
# the knobs are hardcoded below (single source of truth). Built for a deferred
# SLURM slot — submit it with cluster/submit_overnight.sh.
#
# Sequence (the server must be UP before the scorer connects):
#   1. start vLLM in the BACKGROUND          (owns the GPU + the model)
#   2. wait until /v1/models reports OUR model (a cold 30B load takes minutes)
#   3. run the scorer against localhost      (cluster/score_with_local_vllm.sh)
#   4. on ANY exit (done / error / SLURM time-limit SIGTERM) stop vLLM, so the
#      job releases the GPU instead of holding it idle for the whole window.
#
# No `set -e`: the EXIT trap (server cleanup) must run even if the scorer errors,
# and we want to surface the scorer's real exit code.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # …/kbextractor-mine
source "$(dirname "${BASH_SOURCE[0]}")/lib/logging.sh"     # log_path / LOG_ROOT

# ─────────────── EDIT THESE — the LLM-to-serve knobs ───────────────
VLLM_ENV="/fscratch/abuali/venvs/vllm"                 # reusable; created by setup_envs.sh
SCORER_ENV="/fscratch/abuali/venvs/kbextractor-mine"   # this repo's scorer env
MODEL_DIR="/fscratch/abuali/models/Qwen3-30B-A3B-Instruct-2507-FP8"
SERVED_MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
PORT=8000
MAX_MODEL_LEN=16384
TP_SIZE=1
# (scorer knobs — workers, which systems, --limit — live in score_with_local_vllm.sh)
# ───────────────────────────────────────────────────────────────────

echo "🌙 ===== MINE judging — started $(date) ====="
echo "   host=$(hostname)  model=$SERVED_MODEL_NAME  port=$PORT  CUDA=${CUDA_VISIBLE_DEVICES:-?}"

[ -x "$VLLM_ENV/bin/vllm" ]   || { echo "❌ vllm not in env: $VLLM_ENV — run cluster/setup_envs.sh"; exit 1; }
[ -x "$SCORER_ENV/bin/python" ] || { echo "❌ scorer env missing: $SCORER_ENV — run cluster/setup_envs.sh"; exit 1; }
[ -d "$MODEL_DIR" ]          || { echo "❌ model dir not found: $MODEL_DIR"; exit 1; }

# Fail fast if there's no GPU (e.g. accidentally run on the login node) — otherwise
# vLLM dies later with a cryptic "Failed to infer device type".
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "❌ no GPU visible on $(hostname). Run this INSIDE a GPU Slurm job, not directly:"
  echo "   BEGIN=now bash cluster/srun_submit.sh all mine_judge 8 1 80G 6 cluster/run_experiment.sh"
  exit 1
fi
echo "🎮 GPU(s): $(nvidia-smi -L | paste -sd';' -)"

# vLLM's own (noisy) stdout goes to a SEPARATE sub-log so the master job log stays
# readable; this script's narrative is captured by the master log (srun redirect).
VLLM_LOG="$(log_path vllm)"
echo "📄 vLLM log → $VLLM_LOG"

# 1) start vLLM in the background (from the reusable vLLM env).
nohup "$VLLM_ENV/bin/vllm" serve "$MODEL_DIR" \
    --port "$PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-chunked-prefill \
    >> "$VLLM_LOG" 2>&1 &
SERVE_PID=$!

# Stop the server (→ free the GPU) whenever THIS script exits.
cleanup() {
    echo "🧹 stopping vLLM (PID $SERVE_PID) …"
    kill "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
    echo "   stopped."
}
trap cleanup EXIT
trap 'exit 143' TERM    # SLURM time-limit → exit → EXIT trap frees the GPU

# 2) wait until OUR model is actually serving. Checking the model id (not just the
#    port) avoids latching onto a neighbour's vLLM if the node is shared.
echo "⏳ waiting for vLLM to load $SERVED_MODEL_NAME …"
READY=""
for _ in $(seq 1 150); do                  # 150 × 10s = 25 min cap (cold 30B + compile)
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "❌ vLLM exited during startup — last 30 log lines:"; tail -n 30 "$VLLM_LOG"; exit 1
    fi
    if curl -sf "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "$SERVED_MODEL_NAME"; then
        READY="yes"; break
    fi
    sleep 10
done
[ -n "$READY" ] || { echo "❌ vLLM not ready after ~25 min — last 30 log lines:"; tail -n 30 "$VLLM_LOG"; exit 1; }
echo "✅ vLLM READY → http://localhost:${PORT}/v1"

# 3) run the scorer against our local server (judge id == served name).
export MINE_JUDGE_MODEL="openai/${SERVED_MODEL_NAME}"
export MINE_JUDGE_API_BASE="http://localhost:${PORT}/v1"
export VLLM_PORT="$PORT"
export SCORER_ENV
bash "$ROOT/cluster/score_with_local_vllm.sh"
RC=$?

echo "🏁 ===== done $(date)  (scorer exit=$RC) ====="
exit "$RC"
