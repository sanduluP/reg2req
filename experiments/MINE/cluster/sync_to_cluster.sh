#!/usr/bin/env bash
# sync_to_cluster.sh — push the MINE judging bits to Pegasus WITHOUT committing.
# Convention: the local repo is the source of truth, the
# cluster copy is a read-only execution mirror. We only sync what the on-cluster
# judge needs — the scorer, the dataset, and the two systems' pre-built KGs —
# NOT the gitignored results/ (those are generated on the cluster and scp'd back).
#
#   bash experiments/MINE/cluster/sync_to_cluster.sh            # sync
#   bash experiments/MINE/cluster/sync_to_cluster.sh --dry-run  # preview
#
# Override the destination via env: PEGASUS_USER / PEGASUS_HOST / PEGASUS_DEST.
set -euo pipefail

LOCAL_MINE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # …/experiments/MINE
USER_="${PEGASUS_USER:-abuali}"
HOST_="${PEGASUS_HOST:-login1.pegasus.kl.dfki.de}"
DEST_="${PEGASUS_DEST:-/home/abuali/projects/kbextractor-mine}"

DRY=""
[[ "${1:-}" == "--dry-run" ]] && { DRY="--dry-run"; echo "[dry-run] nothing will transfer"; }

# -R/--relative preserves these paths under the destination root. No --delete,
# so a prior run's results/ on the cluster is never wiped by a re-sync.
cd "$LOCAL_MINE"
rsync --archive --compress --human-readable --progress --relative $DRY \
    score_kgs.py \
    data/mine.json \
    kgs/kbextractor \
    kgs/kggen_deepseek \
    cluster \
    "$USER_@$HOST_:$DEST_/"

echo ""
echo "✅ synced → $USER_@$HOST_:$DEST_"
echo "   next (on the cluster, inside a GPU job that's serving the judge):"
echo "     cd $DEST_ && bash cluster/score_with_local_vllm.sh"
