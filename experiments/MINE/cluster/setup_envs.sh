#!/bin/bash
# setup_envs.sh — ONE-TIME: create two ISOLATED venvs on /fscratch so no repo ever
# borrows another repo's environment again:
#
#   /fscratch/abuali/venvs/vllm              ← just vLLM. REUSABLE: any repo that
#                                              needs to serve a model uses this one.
#   /fscratch/abuali/venvs/kbextractor-mine  ← THIS repo's scorer (kg_gen + dspy +
#                                              the all-MiniLM-L6-v2 retriever).
#
# Two envs because serving and scoring are two separate programs: the vLLM server
# (one process) and score_kgs.py (another) only talk over HTTP, so they don't need
# to share an interpreter — and keeping vLLM pure means other repos can reuse it.
#
# Run on a COMPUTE node INSIDE the NVIDIA container (heavy CUDA wheels). Submit it
# as a CPU job from this repo's cluster dir:
#   bash cluster/srun_submit.sh batch setup_envs 8 0 32G 2 cluster/setup_envs.sh
#   tail -f logs/setup_envs/setup_envs_*.log     # ~15-25 min
set -euo pipefail

VENV_ROOT="${VENV_ROOT:-/fscratch/abuali/venvs}"
VLLM_ENV="${VLLM_ENV:-$VENV_ROOT/vllm}"
SCORER_ENV="${SCORER_ENV:-$VENV_ROOT/kbextractor-mine}"
mkdir -p "$VENV_ROOT"

# 1) Reusable vLLM serving env (vLLM pins its own torch/CUDA wheels).
if [ -x "$VLLM_ENV/bin/vllm" ]; then
  echo "↩︎ vLLM env already present: $VLLM_ENV"
else
  echo "📦 creating REUSABLE vLLM env → $VLLM_ENV  (heavy; ~10-15 min)"
  python3 -m venv "$VLLM_ENV"
  "$VLLM_ENV/bin/pip" install -q --upgrade pip
  "$VLLM_ENV/bin/pip" install -q vllm
fi
echo "   vllm: $("$VLLM_ENV/bin/vllm" --version 2>/dev/null || echo '?')"

# 2) This repo's scorer env (kg_gen pulls dspy; sentence-transformers = retriever).
if [ -x "$SCORER_ENV/bin/python" ] && "$SCORER_ENV/bin/python" -c "import kg_gen, dspy, sentence_transformers" 2>/dev/null; then
  echo "↩︎ scorer env already present: $SCORER_ENV"
else
  echo "📦 creating scorer env → $SCORER_ENV  (~5-10 min)"
  python3 -m venv "$SCORER_ENV"
  "$SCORER_ENV/bin/pip" install -q --upgrade pip
  # kg-gen from GitHub main (PyPI lags + lacks steps._3_deduplicate); semhash pinned
  # <0.4 for the 0.3.x API kg-gen expects (newer 0.4.x breaks it, see kg-gen issue #98).
  "$SCORER_ENV/bin/pip" install -q \
      'git+https://github.com/stair-lab/kg-gen.git' 'semhash>=0.3.2,<0.4' sentence-transformers
  echo "🤗 pre-caching the retriever (HF's 10s default times out)"
  HF_HUB_DOWNLOAD_TIMEOUT=120 "$SCORER_ENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("sentence-transformers/all-MiniLM-L6-v2"); print("cached all-MiniLM-L6-v2")
PY
fi

echo ""
echo "✅ done."
echo "   vLLM env  (reusable): $VLLM_ENV"
echo "   scorer env (this repo): $SCORER_ENV"
