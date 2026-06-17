#!/bin/bash
# srun_submit.sh — GENERAL non-interactive Slurm submitter (the batch sibling of
# an interactive `srun --pty`). Knows nothing about MINE: you give it resources +
# a task script, it runs that script on a compute node inside the NVIDIA
# container, logging to /fscratch/abuali/logs. Self-contained — this repo owns it.
#
# Usage:
#   bash cluster/srun_submit.sh PARTITION JOB_NAME CPUS GPUS MEM HOURS SCRIPT [args...]
# Examples:
#   bash cluster/srun_submit.sh batch setup_envs 8 0 32G 2 cluster/setup_envs.sh
#   bash cluster/srun_submit.sh H100  mine_judge 8 1 80G 6 cluster/run_experiment.sh
#
# ⏰ Deferred start: prefix BEGIN= to hold the job PENDING until a time — grab a
# busy GPU overnight when the cluster is idle:
#   BEGIN=00:00                bash cluster/srun_submit.sh H100 mine_judge 8 1 80G 6 cluster/run_experiment.sh
#   BEGIN=now+4hours           bash cluster/srun_submit.sh ...
#   BEGIN=2026-06-18T02:00:00  bash cluster/srun_submit.sh ...
# Run inside screen/tmux on the login node so the nohup'd srun client survives logout.
set -euo pipefail

if [ "$#" -lt 7 ]; then
  echo "Usage: $0 PARTITION JOB_NAME CPUS GPUS MEM HOURS SCRIPT [args...]"
  echo "Example: $0 H100 mine_judge 8 1 80G 6 cluster/run_experiment.sh"
  exit 1
fi

PARTITION=$1; JOB_NAME=$2; CPUS=$3; GPUS=$4; MEM=$5; HOURS=$6; TASK_SCRIPT=$7
shift 7
TASK_ARGS="$@"

[ -f "$TASK_SCRIPT" ] || { echo "Error: task script not found: $TASK_SCRIPT"; exit 1; }

# Normalize memory ('G' / 'GB' / bare → 'G').
if   [[ $MEM == *GB ]]; then MEM="${MEM%GB}G"
elif [[ $MEM != *G  ]]; then MEM="${MEM}G"; fi

WORKDIR=$(pwd)
LOG_DIR="/fscratch/abuali/logs"; mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${JOB_NAME}_$(date +%Y%m%d_%H%M%S).log"

# Shared NVIDIA PyTorch container (CUDA 12.8 → Ampere/Hopper/Blackwell). This is a
# system image, not any repo's — fine to reference.
NVIDIA_CONTAINER_VERSION="${NVIDIA_CONTAINER_VERSION:-25.02}"
CONTAINER="/netscratch/enroot/nvcr.io_nvidia_pytorch_${NVIDIA_CONTAINER_VERSION}-py3.sqsh"
[ -f "$CONTAINER" ] || { echo "🚨 Container not found: $CONTAINER"; exit 1; }

# Only request GPUs when a non-zero count is asked for (so CPU-only setup jobs work).
GPU_FLAGS=""
if [[ "${GPUS}" != "0" && -n "${GPUS}" ]]; then GPU_FLAGS="--gres=gpu:${GPUS}"; fi

# ⏰ Optional deferred start. Empty = start ASAP (unchanged).
BEGIN="${BEGIN:-}"

echo "🚢 Container: $CONTAINER"
echo "💻 $PARTITION | job=$JOB_NAME cpus=$CPUS gpus=$GPUS mem=$MEM time=${HOURS}h"
echo "📜 Script: $TASK_SCRIPT $TASK_ARGS"
echo "📄 Log:    $LOG_FILE"
[[ -n "$BEGIN" ]] && echo "⏰ Begin:   $BEGIN  (job stays PENDING until then)"

nohup srun -K \
  -p "$PARTITION" \
  --job-name "$JOB_NAME" \
  --nodes 1 \
  --ntasks-per-node 1 \
  --cpus-per-task "$CPUS" \
  ${GPU_FLAGS} \
  --mem "$MEM" \
  ${BEGIN:+--begin="$BEGIN"} \
  --time="${HOURS}:00:00" \
  --chdir "$WORKDIR" \
  --container-image "$CONTAINER" \
  --container-mounts /home:/home,/netscratch:/netscratch,/ds:/ds,/fscratch:/fscratch \
  --container-workdir "$WORKDIR" \
  bash "$TASK_SCRIPT" $TASK_ARGS \
  > "$LOG_FILE" 2>&1 &

echo "Started background srun job. Monitor with:"
echo "  squeue -u abuali        # (PD = pending: waiting for the begin time or resources)"
echo "  tail -f $LOG_FILE"
