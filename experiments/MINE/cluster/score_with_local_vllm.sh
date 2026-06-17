#!/usr/bin/env bash
# score_with_local_vllm.sh — judge both MINE systems against a vLLM server
# running on the SAME node (localhost). Run this INSIDE a Pegasus GPU job that is
# already serving the judge model (e.g. via kggen-eval/scripts/serve_vllm.sh).
#
# Reuses the kggen-eval venv (it already has kg_gen + dspy + the cached
# all-MiniLM-L6-v2 retriever), so score_kgs.py runs unchanged. Results land in
# ./results/<system>/<judge-slug>/ — scp that back and feed it to make_report.py
# for a 3rd-judge ablation column.
#
# Defaults target the Qwen3-30B judge that serve_vllm.sh already serves; override
# any of these via env:
#   MINE_JUDGE_MODEL   served-model-name, openai/-prefixed (litellm wire format)
#   VLLM_PORT          vLLM port (default 8000)
#   JUDGE_WORKERS      concurrent judge calls (vLLM batches — 16 is comfortable)
#   KGGEN_VENV         path to the venv with kg_gen + dspy
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # …/kbextractor-mine
PY="${KGGEN_VENV:-/fscratch/abuali/venvs/kggen-eval}/bin/python"
PORT="${VLLM_PORT:-8000}"
WORKERS="${JUDGE_WORKERS:-16}"

export MINE_JUDGE_MODEL="${MINE_JUDGE_MODEL:-openai/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
export MINE_JUDGE_API_BASE="${MINE_JUDGE_API_BASE:-http://localhost:${PORT}/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"   # litellm needs *some* key; vLLM ignores it

[[ -x "$PY" ]] || { echo "❌ venv python not found: $PY (set KGGEN_VENV)"; exit 1; }
# Preflight: is the judge actually serving on this node?
if ! curl -sf "$MINE_JUDGE_API_BASE/models" >/dev/null 2>&1; then
  echo "❌ no vLLM at $MINE_JUDGE_API_BASE — serve the judge first:"
  echo "     bash <kggen-eval>/scripts/serve_vllm.sh   # waits until READY"
  exit 1
fi

LOG_DIR="$ROOT/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/score_$(date +%Y%m%d_%H%M%S).log"
echo "🧪 judge=$MINE_JUDGE_MODEL  api_base=$MINE_JUDGE_API_BASE  workers=$WORKERS"
echo "📄 log → $LOG"

{
  for sysname in kbextractor kggen_deepseek; do
    echo "════════ scoring $sysname ════════"
    "$PY" "$ROOT/score_kgs.py" --system "$sysname" \
        --kgs-dir "$ROOT/kgs/$sysname" \
        --data "$ROOT/data/mine.json" \
        --out-dir "$ROOT/results" \
        --judge-workers "$WORKERS" "$@"
  done
} 2>&1 | tee "$LOG"

echo ""
echo "✅ done. results under $ROOT/results/{kbextractor,kggen_deepseek}/"
echo "   scp back to the local repo's experiments/MINE/results/ then:"
echo "     experiments/MINE/report/.venv/bin/python experiments/MINE/report/make_report.py"
