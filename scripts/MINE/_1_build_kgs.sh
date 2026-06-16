#!/usr/bin/env bash
# MINE Stage 1 — build a KBExtractor KG per essay (KBExtraction venv, DFKI network).
#   scripts/MINE/_1_build_kgs.sh --limit 1        # smoke-test one essay
#   scripts/MINE/_1_build_kgs.sh                   # all essays
#   scripts/MINE/_1_build_kgs.sh --overwrite
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_py "$KB_PY" "KBExtraction" "KB_VENV"

echo "🧬 [MINE Stage 1] KBExtractor KGs → $KGS_DIR"
cd "$REPO_ROOT"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
run_logged build_kgs "$KB_PY" "$MINE_DIR/build_kbextractor_kgs.py" \
  --data "$DATA_JSON" --out-dir "$KGS_DIR" "$@"
