#!/usr/bin/env bash
# Bootstrap a fresh RunPod pod to train mtg-draft-ml, using uv.
#
# Recommended (clone then run — robust):
#   cd /workspace && git clone https://github.com/b-r-a-n/mtg-draft-ml.git \
#     && cd mtg-draft-ml && bash scripts/runpod_bootstrap.sh
#
# Assumes a RunPod "PyTorch" pod (Ubuntu + CUDA). Picks the CUDA torch wheel automatically
# on Linux. Choose a HIGH-vCPU/RAM instance — this workload is data-loading-bound, not FLOP-bound.
set -euo pipefail

# 1. uv (standalone installer — no system python needed)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Ensure uv is on PATH now AND in future (non-interactive) shells / SSH sessions.
export PATH="$HOME/.local/bin:$PATH"
grep -qs 'HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null \
  || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"

# 2. get the code. If REPO is set and we're not already in the repo, clone + cd into it.
#    Otherwise assume cwd is the repo root (the recommended clone-then-run flow).
REPO="${REPO:-}"
if [ ! -f pyproject.toml ]; then
  if [ -n "$REPO" ]; then
    [ -d mtg-draft-ml ] || git clone "$REPO" mtg-draft-ml
    cd mtg-draft-ml
  else
    echo "error: run from the repo root, or set REPO=<git-url>" >&2; exit 1
  fi
fi

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
