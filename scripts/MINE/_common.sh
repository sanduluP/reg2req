#!/usr/bin/env bash
# Shared configuration for the MINE step scripts. Sourced, not executed.
#
# Override any interpreter path by exporting the matching var before calling,
# e.g.  KGGEN_VENV=/some/other/.venv  scripts/MINE/fetch_dataset.sh
set -euo pipefail

# Repo root = two levels up from this file (scripts/MINE/_common.sh)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MINE_DIR="$REPO_ROOT/experiments/MINE"
DATA_JSON="$MINE_DIR/data/mine.json"
KGS_DIR="$MINE_DIR/kgs/kbextractor"
KGGEN_KGS_DIR="$MINE_DIR/kgs/kggen_deepseek"
RESULTS_DIR="$MINE_DIR/results"
LOGS_DIR="$REPO_ROOT/logs/MINE"

# KBExtraction venv (Stage 1) — built by ./setup.sh
KB_VENV="${KB_VENV:-$REPO_ROOT/venv}"
KB_PY="$KB_VENV/bin/python"

# kg-gen venv (Stages 0 & 2)
KGGEN_VENV="${KGGEN_VENV:-/home/faris/code/DSA_HiWi/kg-gen/.venv}"
KGGEN_PY="$KGGEN_VENV/bin/python"

require_py () {
  # require_py <python-path> <label> <override-var-name>
  local py="$1" label="$2" var="$3"
  if [[ ! -x "$py" ]]; then
    echo "❌ $label Python not found at: $py" >&2
    echo "   Fix by exporting $var=/path/to/venv (the dir containing bin/python)." >&2
    exit 1
  fi
}

run_logged () {
  # run_logged <stage-label> <command...>
  # Runs the command with combined stdout+stderr tee'd to a timestamped log
  # under $LOGS_DIR, and exits with the command's (not tee's) status.
  local stage="$1"; shift
  mkdir -p "$LOGS_DIR"
  local log="$LOGS_DIR/${stage}_$(date +%Y%m%d_%H%M%S).log"
  echo "📝 logging to $log"
  "$@" 2>&1 | tee "$log"
  local status="${PIPESTATUS[0]}"
  echo "📝 saved log: $log (exit $status)"
  return "$status"
}
