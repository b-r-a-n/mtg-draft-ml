#!/usr/bin/env bash
# Bootstrap a fresh RunPod pod to train mtg-draft-ml, using uv. Run on the pod:
#   REPO=git@github.com:<you>/mtg-draft-ml.git bash runpod_bootstrap.sh
# (or clone the repo first and just run `bash scripts/runpod_bootstrap.sh`)
#
# Assumes a RunPod "PyTorch" pod (Ubuntu + CUDA). Picks the CUDA torch wheel automatically
# on Linux. Choose a HIGH-vCPU/RAM instance — this workload is data-loading-bound, not FLOP-bound.
set -euo pipefail

# 1. uv (standalone installer — no system python needed)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. get the code
REPO="${REPO:-}"
if [ -n "$REPO" ] && [ ! -d mtg-draft-ml ]; then
  git clone "$REPO" mtg-draft-ml
fi
cd "$(dirname "$0")/.." 2>/dev/null || cd mtg-draft-ml

# 3. env — installs CUDA torch from the default index on Linux.
#    (Tip: `uv venv --system-site-packages` reuses the pod's preinstalled torch to skip a
#     multi-GB download, at the cost of a less-isolated env. Clean venv is the default here.)
uv venv
uv pip install -e ".[dev,hub,embeddings]"

# 4. sanity check: is the GPU visible?
uv run python -c "import torch; ok=torch.cuda.is_available(); print('CUDA:', ok, '|', torch.cuda.get_device_name(0) if ok else 'NO GPU VISIBLE')"

echo
echo "Bootstrap done."
echo "Next:"
echo "  huggingface-cli login            # only needed to PUSH (data/checkpoints); reads are public"
echo "  uv run python -m mtg_draft_ml.data.pipeline --set <SET> --pull --hf-repo <you>/mtg-draft"
echo "  uv run python -m mtg_draft_ml.eval.generalization ...   # train / sweep"
echo "Remember to DELETE the pod + volume when done (storage bills while stopped)."
