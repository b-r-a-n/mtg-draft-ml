#!/usr/bin/env bash
# CPU-ONLY bootstrap for pure-CPU pod jobs (de-censoring sweep, playprob export — any sklearn/pandas
# /numpy work). The default scripts/runpod_bootstrap.sh installs the CUDA torch wheel pinned in
# pyproject [tool.uv.sources] (cu124, 2-3 GB) and asserts a GPU — overkill for CPU jobs, and it has
# HUNG on pod start mid-download. This installs CPU torch (~200 MB, needed by game_value.fit_card_values'
# torch L-BFGS) + the runtime deps explicitly, and is idempotent.
#
# IMPORTANT: jobs using this must run the venv python DIRECTLY with PYTHONPATH=src
#   PYTHONPATH=src .venv/bin/python scripts/foo.py
# NOT `uv run` (which auto-syncs the project and would re-pull the CUDA torch pin) and NOT
# `uv pip install -e .` (same pin). The package is reached via PYTHONPATH, not an editable install.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
[ -f pyproject.toml ] || cd "$(dirname "$0")/.."
[ -f pyproject.toml ] || { echo "FATAL: run from the repo root"; exit 1; }

# idempotent: if a working CPU env already exists, skip the installs
if [ -x .venv/bin/python ] && .venv/bin/python -c "import torch,sklearn,pandas,numpy,scipy,requests,huggingface_hub" 2>/dev/null; then
  echo "CPU env already present · $(.venv/bin/python -c 'import torch;print("torch",torch.__version__)')"
  exit 0
fi

echo ">>> creating CPU venv (no CUDA torch)"
uv venv
# CPU torch from the CPU index (bypasses the linux CUDA pin); then the rest of the runtime deps.
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy pandas pyarrow scipy scikit-learn requests pyyaml tqdm huggingface_hub
.venv/bin/python -c "import torch,sklearn,pandas,numpy; print('CPU bootstrap OK · torch', torch.__version__)"
