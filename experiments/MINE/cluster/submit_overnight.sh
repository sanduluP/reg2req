#!/bin/bash
# ---------------------------------------------------------------------------
# submit_overnight.sh — ONE command to queue the whole MINE judging run for a
# quiet slot (default: tonight at 00:00) on H100, via kggen-eval's srun_submit.sh.
# Edit the knobs below, then just:  bash cluster/submit_overnight.sh
#
# Run it inside tmux/screen on the login node — srun --begin keeps the client
# PENDING until the start time, and the nohup'd client must outlive your SSH.
# ---------------------------------------------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # …/kbextractor-mine

# ─────────────── EDIT THESE — the SLURM-slot knobs ───────────────
PARTITION="H100"
JOB_NAME="mine_judge"
CPUS=8
GPUS=1
MEM="80G"
HOURS=6
BEGIN="${BEGIN:-00:00}"      # tonight at midnight. Immediate: BEGIN=now bash cluster/submit_overnight.sh
SRUN_SUBMIT="${SRUN_SUBMIT:-/home/abuali/projects/kggen-eval/scripts/srun_submit.sh}"
# ─────────────────────────────────────────────────────────────────

[ -f "$SRUN_SUBMIT" ] || { echo "❌ srun_submit.sh not found: $SRUN_SUBMIT (set SRUN_SUBMIT=…)"; exit 1; }

cd "$ROOT"   # srun_submit.sh uses $(pwd) as WORKDIR / container-workdir
echo "🌙 queuing MINE judging → partition=$PARTITION begin=$BEGIN  ($CPUS cpu / $GPUS gpu / $MEM / ${HOURS}h)"
BEGIN="$BEGIN" bash "$SRUN_SUBMIT" \
    "$PARTITION" "$JOB_NAME" "$CPUS" "$GPUS" "$MEM" "$HOURS" \
    cluster/run_experiment.sh

echo ""
echo "▶ watch:   squeue -u abuali        (PD until $BEGIN / a free $PARTITION)"
echo "▶ log:     tail -f /fscratch/abuali/logs/${JOB_NAME}_*.log"
