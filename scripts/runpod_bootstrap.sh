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
# Ensure uv is on PATH everywhere:
#  - current shell + interactive shells (.bashrc)
#  - non-interactive `ssh host 'cmd'` sessions, which do NOT source .bashrc — symlink into
#    /usr/local/bin (on the default PATH for all shell types) so bare `uv ...` works over SSH.
export PATH="$HOME/.local/bin:$PATH"
grep -qs 'HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null \
  || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
ln -sf "$HOME/.local/bin/uv" /usr/local/bin/uv 2>/dev/null || true
ln -sf "$HOME/.local/bin/uvx" /usr/local/bin/uvx 2>/dev/null || true

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

# 3. env. torch's CUDA build (cu124 on Linux) is pinned in pyproject.toml [tool.uv.sources], so a
# plain sync gets a driver-compatible torch consistently — including under `uv run` auto-sync.
uv venv
uv pip install -e ".[dev,hub,embeddings]"

# 4. sanity check via `uv run` (the same path training uses, so it catches any sync regression).
#    FATAL if the GPU isn't visible — better to abort here than crash silently later.
uv run python -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA:', ok, '|', torch.cuda.get_device_name(0) if ok else 'NO GPU VISIBLE', '| torch', torch.__version__); sys.exit(0 if ok else 1)" \
  || { echo 'FATAL: torch cannot see the GPU (driver/CUDA-wheel mismatch) — aborting bootstrap.'; exit 1; }

echo
echo "Bootstrap done."
echo "Next:"
echo "  huggingface-cli login            # only needed to PUSH (data/checkpoints); reads are public"
echo "  uv run python -m mtg_draft_ml.data.pipeline --set <SET> --pull --hf-repo <you>/mtg-draft"
echo "  uv run python -m mtg_draft_ml.eval.generalization ...   # train / sweep"
echo "Remember to DELETE the pod + volume when done (storage bills while stopped)."
