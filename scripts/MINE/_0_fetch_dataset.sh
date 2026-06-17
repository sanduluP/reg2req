#!/usr/bin/env bash
# MINE Stage 0 — cache the evaluation dataset locally (kg-gen venv).
#   scripts/MINE/_0_fetch_dataset.sh            # default output
#   scripts/MINE/_0_fetch_dataset.sh --split train
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_py "$KGGEN_PY" "kg-gen" "KGGEN_VENV"

echo "📥 [MINE Stage 0] dataset → $DATA_JSON"
run_logged fetch_dataset "$KGGEN_PY" "$MINE_DIR/fetch_dataset.py" --out "$DATA_JSON" "$@"
