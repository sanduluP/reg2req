#!/usr/bin/env bash
# MINE Stage 1b — build the KGGen baseline on the SAME deepseek backbone
# (kg-gen venv, DFKI network). Same-backbone, correctly-aligned KGGen graphs to
# compare head-to-head with KBExtractor (the dataset's pre-generated kggen graphs
# are GPT-4o/Gemini-built AND misaligned, so we regenerate).
#   scripts/MINE/_1b_build_kggen.sh --limit 1     # smoke-test one essay
#   scripts/MINE/_1b_build_kggen.sh                # all essays
#
# Backbone override via env: KGGEN_MODEL / KGGEN_API_BASE / KGGEN_API_KEY.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_py "$KGGEN_PY" "kg-gen" "KGGEN_VENV"

echo "🧬 [MINE Stage 1b] KGGen-on-deepseek KGs → $KGGEN_KGS_DIR"
run_logged build_kggen "$KGGEN_PY" "$MINE_DIR/build_kggen_kgs.py" \
  --data "$DATA_JSON" --out-dir "$KGGEN_KGS_DIR" "$@"
