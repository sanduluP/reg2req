#!/usr/bin/env bash
# serve_and_score.sh — ONE self-contained task for srun_submit.sh: bring up the
# vLLM judge on THIS node, judge both MINE systems against it, then tear vLLM
# down. Because a single task script stays alive for the whole serve+score, the
# Slurm job (and the backgrounded vLLM) live exactly as long as needed — ideal
# for a detached / overnight run, optionally deferred with BEGIN= in srun_submit.
#
# NOTE: do NOT submit serve_vllm.sh directly via srun_submit — it backgrounds
# vLLM and returns, so the job would end and kill vLLM before any scoring. This
# script keeps the job alive across both phases.
#
# Submit it (run from the kbextractor-mine dir), e.g. overnight on a busy H100:
#   BEGIN=00:00 bash /home/abuali/projects/kggen-eval/scripts/srun_submit.sh \
#       H100 mine_judge 8 1 80G 6 cluster/serve_and_score.sh
#
# Override the judge via env (defaults = the Qwen3-30B judge on /fscratch):
#   MODEL_DIR / SERVED_MODEL_NAME / VLLM_PORT / KGGEN_VENV / JUDGE_WORKERS /
#   TP_SIZE / MAX_MODEL_LEN. For gpt-oss later: MODEL_DIR=…/gpt-oss-20b
#   SERVED_MODEL_NAME=gpt-oss-20b.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # …/kbextractor-mine
VENV="${KGGEN_VENV:-/fscratch/abuali/venvs/kggen-eval}"
MODEL_DIR="${MODEL_DIR:-/fscratch/abuali/models/Qwen3-30B-A3B-Instruct-2507-FP8}"
SERVED="${SERVED_MODEL_NAME:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
PORT="${VLLM_PORT:-8000}"

[ -x "$VENV/bin/vllm" ] || { echo "❌ vllm not in venv ($VENV) — set KGGEN_VENV"; exit 1; }
[ -d "$MODEL_DIR" ]     || { echo "❌ model dir not found: $MODEL_DIR — set MODEL_DIR"; exit 1; }

LOG_DIR="$ROOT/logs"; mkdir -p "$LOG_DIR"
VLLM_LOG="$LOG_DIR/vllm_$(date +%Y%m%d_%H%M%S).log"
echo "🖥️  node=$(hostname)  model=$SERVED  port=$PORT"
echo "📄 vLLM log → $VLLM_LOG"

# 1) launch vLLM in the background on this node (same flags as kggen-eval's proven
#    serve_vllm.sh: TP=1, dtype auto, 16k ctx, chunked prefill).
nohup "$VENV/bin/vllm" serve "$MODEL_DIR" \
    --port "$PORT" \
    --served-model-name "$SERVED" \
    --tensor-parallel-size "${TP_SIZE:-1}" \
    --dtype auto \
    --max-model-len "${MAX_MODEL_LEN:-16384}" \
    --enable-chunked-prefill \
    >> "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
cleanup() { kill "$VLLM_PID" 2>/dev/null || true; }  # stop vLLM on any exit
trap cleanup EXIT

# 2) wait until it answers, or fail fast if it dies during load (~3-8 min).
echo "⏳ waiting for vLLM to load…"
for _ in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "✅ vLLM READY → http://localhost:${PORT}/v1"; break
    fi
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "❌ vLLM died during startup:"; tail -n 30 "$VLLM_LOG"; exit 1; }
    sleep 10
done
curl -sf "http://localhost:${PORT}/v1/models" >/dev/null || { echo "❌ vLLM not ready after ~20 min"; tail -n 30 "$VLLM_LOG"; exit 1; }

# 3) judge both systems against the local server (judge id == served name).
export MINE_JUDGE_MODEL="openai/${SERVED}"
export MINE_JUDGE_API_BASE="http://localhost:${PORT}/v1"
export VLLM_PORT="$PORT"
bash "$ROOT/cluster/score_with_local_vllm.sh" "$@"

echo "✅ serve_and_score complete — results under $ROOT/results/"
