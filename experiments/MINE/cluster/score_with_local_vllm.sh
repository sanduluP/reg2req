#!/usr/bin/env bash
# score_with_local_vllm.sh — judge both MINE systems against a vLLM server
# running on the SAME node (localhost). Normally run by run_experiment.sh once the
# server is up; can also run standalone against an already-serving judge.
#
# Uses this repo's scorer env ($SCORER_ENV = /fscratch/abuali/venvs/kbextractor-mine,
# built by setup_envs.sh: kg_gen + dspy + the cached all-MiniLM-L6-v2 retriever), so
# score_kgs.py runs unchanged. Results land in ./results/<system>/<judge-slug>/ —
# scp that back and feed it to make_report.py for a 3rd-judge ablation column.
#
# Usually invoked by run_experiment.sh (which exports MINE_JUDGE_MODEL/_API_BASE
# after the server is up), but also runnable standalone against an already-serving
# judge. The SCORER knobs live here; the model-to-serve knobs live in
# run_experiment.sh. Anything overridable via env:
#   MINE_JUDGE_MODEL   served-model-name, openai/-prefixed (litellm wire format)
#   VLLM_PORT          vLLM port (default 8000)
#   JUDGE_WORKERS      concurrent judge calls (vLLM batches — 16 is comfortable)
#   SCORE_ARGS         extra score_kgs.py flags (e.g. "--limit 3", "--overwrite")
#   SCORER_ENV         this repo's venv (kg_gen + dspy + retriever; from setup_envs.sh)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # …/kbextractor-mine
source "$(dirname "${BASH_SOURCE[0]}")/lib/logging.sh"     # log_path / LOG_ROOT
PY="${SCORER_ENV:-/fscratch/abuali/venvs/kbextractor-mine}/bin/python"
PORT="${VLLM_PORT:-8000}"

# ─────────────── 🐴 EDIT THESE — the scorer knobs ───────────────
WORKERS="${JUDGE_WORKERS:-16}"     # concurrent judge calls
SYSTEMS=(kbextractor kggen_deepseek)
SCORE_ARGS="${SCORE_ARGS:-}"       # e.g. "--limit 3" smoke test · "--overwrite" · "--ids 4,10"
# ─────────────────────────────────────────────────────────────

export MINE_JUDGE_MODEL="${MINE_JUDGE_MODEL:-openai/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
export MINE_JUDGE_API_BASE="${MINE_JUDGE_API_BASE:-http://localhost:${PORT}/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"   # litellm needs *some* key; vLLM ignores it

[[ -x "$PY" ]] || { echo "❌ scorer python not found: $PY — run cluster/setup_envs.sh (set SCORER_ENV)"; exit 1; }
# Preflight: is the judge actually serving on this node?
if ! curl -sf "$MINE_JUDGE_API_BASE/models" >/dev/null 2>&1; then
  echo "❌ no vLLM at $MINE_JUDGE_API_BASE — start the server first."
  echo "   (run_experiment.sh does this for you; standalone, serve the model on \$VLLM_PORT.)"
  exit 1
fi

LOG="$(log_path score)"   # focused per-essay judging log (also flows to the master log)
echo "🧪 judge=$MINE_JUDGE_MODEL  api_base=$MINE_JUDGE_API_BASE  workers=$WORKERS"
echo "📄 log → $LOG"

{
  for sysname in "${SYSTEMS[@]}"; do
    echo "════════ scoring $sysname ════════"
    "$PY" "$ROOT/score_kgs.py" --system "$sysname" \
        --kgs-dir "$ROOT/kgs/$sysname" \
        --data "$ROOT/data/mine.json" \
        --out-dir "$ROOT/results" \
        --judge-workers "$WORKERS" $SCORE_ARGS "$@"
  done
} 2>&1 | tee "$LOG"

echo ""
echo "✅ done. results under $ROOT/results/{kbextractor,kggen_deepseek}/"
echo "   scp back to the local repo's experiments/MINE/results/ then:"
echo "     experiments/MINE/report/.venv/bin/python experiments/MINE/report/make_report.py"
