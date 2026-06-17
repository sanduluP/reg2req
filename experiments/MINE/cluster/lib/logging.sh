#!/bin/bash
# logging.sh — shared, timestamped logging for the cluster scripts. Source it:
#   source "$(dirname "$0")/lib/logging.sh"
#
# Logs default to a PER-PROJECT tree that's easy to open in VSCode:
#   <repo>/logs/<category>/<category>_<YYYYmmdd_HHMMSS>.log
# (e.g. kbextractor-mine/logs/mine_judge/mine_judge_20260617_200256.log)
# Override the root with LOG_ROOT=/some/dir (e.g. LOG_ROOT=/fscratch/abuali/logs).

# Repo root = two levels up from this file (cluster/lib/logging.sh).
_LOGGING_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-${_LOGGING_REPO_ROOT}/logs}"

# log_path <category> — echo a fresh timestamped log path (creating its dir),
# WITHOUT redirecting the whole script. Use to log ONE command/stream, e.g.:
#   LOG="$(log_path vllm)";  mycmd >> "$LOG" 2>&1 &
log_path() {
    local category="$1" dir
    dir="${LOG_ROOT}/${category}"
    mkdir -p "$dir"
    echo "${dir}/${category}_$(date +%Y%m%d_%H%M%S).log"
}

# init_log <category> — tee this WHOLE script's stdout+stderr to a timestamped
# file under LOG_ROOT, while still printing to the terminal when one is attached.
# Re-runs never overwrite (each gets its own timestamp). Sets $LOG_FILE.
init_log() {
    local category="$1"
    export LOG_FILE; LOG_FILE="$(log_path "$category")"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "📄 Log → ${LOG_FILE}"
}
