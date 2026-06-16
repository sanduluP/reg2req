#!/usr/bin/env bash
# MINE Stage 2 — score a system against MINE-1 (kg-gen venv).
#   scripts/MINE/_2_score.sh                       # defaults to kbextractor
#   scripts/MINE/_2_score.sh kbextractor --limit 1
#   scripts/MINE/_2_score.sh kggen                 # free baseline (pre-generated KG)
#   scripts/MINE/_2_score.sh graphrag
#   scripts/MINE/_2_score.sh openie
#
# Scoring is idempotent: essays already in results/<system>/<judge>/ are reused
# with no judge calls. Add --overwrite to force re-judging.
#
# Judge model is env-driven and defaults to the on-prem DFKI deepseek (zero
# cost). To use GPT-5 instead, export before calling:
#   export MINE_JUDGE_MODEL=openai/gpt-5
#   export OPENAI_API_KEY=sk-...real-key...
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_py "$KGGEN_PY" "kg-gen" "KGGEN_VENV"

# --- judge config (on-prem deepseek by default; everything overridable) ---
# If no judge model is exported, assume the DFKI deepseek + its endpoint.
# If the caller exported MINE_JUDGE_MODEL (e.g. gpt-5), we touch nothing here.
if [[ -z "${MINE_JUDGE_MODEL:-}" ]]; then
  export MINE_JUDGE_MODEL="openai/deepseek-r1:32b"
  export MINE_JUDGE_API_BASE="${MINE_JUDGE_API_BASE:-http://serv-3306.kl.dfki.de:8000/v1}"
fi
# litellm's OpenAI client requires *some* key even when the endpoint ignores it.
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"

# --- system selection (positional; a leading flag means "use the default") ---
SYSTEM="kbextractor"
if [[ $# -gt 0 && "$1" != -* ]]; then
  SYSTEM="$1"
  shift
fi

echo "🧪 [MINE Stage 2] scoring system=$SYSTEM (judge=$MINE_JUDGE_MODEL)"
args=(--system "$SYSTEM" --data "$DATA_JSON" --out-dir "$RESULTS_DIR")
if [[ "$SYSTEM" == "kbextractor" ]]; then
  args+=(--kgs-dir "$KGS_DIR")
fi
# "$@" forwards any remaining flags to the Python script, e.g. --limit 1
run_logged "score_${SYSTEM}" "$KGGEN_PY" "$MINE_DIR/score_kgs.py" "${args[@]}" "$@"
