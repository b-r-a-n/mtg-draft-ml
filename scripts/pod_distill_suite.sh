#!/usr/bin/env bash
# Run the full distillation suite on a fresh RunPod pod: bootstrap -> pull data -> 3 experiments.
# Assumes the repo is cloned at /workspace/mtg-draft-ml (or runs from the repo root). Drives the
# three committed experiments on the GPU; results land in data/*.json on the pod.
#
# Launch detached so it survives an SSH drop (note the `< /dev/null` — without it `nohup … &` over
# ssh keeps the channel open and the local ssh hangs):
#   ssh … root@<pod> 'cd /workspace && nohup bash mtg-draft-ml/scripts/pod_distill_suite.sh \
#       > run.log 2>&1 < /dev/null & echo PID $!'
# then poll: ssh … root@<pod> 'tail -30 /workspace/run.log'
#
# Env: EPOCHS (default 8), HF_REPO (default b-r-a-n/mtg-draft).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1   # stream per-epoch prints to run.log live (else block-buffered to a file)
EPOCHS="${EPOCHS:-8}"
HF_REPO="${HF_REPO:-b-r-a-n/mtg-draft}"

# locate / enter the repo
if [ ! -f pyproject.toml ]; then
  cd /workspace 2>/dev/null || true
  [ -d mtg-draft-ml ] || git clone --depth 1 https://github.com/b-r-a-n/mtg-draft-ml.git
  cd mtg-draft-ml
fi
git pull --ff-only 2>/dev/null || true

echo "=== [$(date +%H:%M:%S)] bootstrap (uv + CUDA torch + deps) ==="
bash scripts/runpod_bootstrap.sh
echo "=== [$(date +%H:%M:%S)] pull dataset $HF_REPO ==="
uv run python -m mtg_draft_ml.data.hf pull --repo "$HF_REPO" --out data/hf

echo "=== [$(date +%H:%M:%S)] 1/3 ensemble-of-seeds ==="
uv run python scripts/pod_distill.py     --epochs "$EPOCHS" --device cuda
echo "=== [$(date +%H:%M:%S)] 2/3 WR-softmax (dense vs scalar) ==="
uv run python scripts/pod_wr_distill.py  --epochs "$EPOCHS" --device cuda
echo "=== [$(date +%H:%M:%S)] 3/3 leaky -> release-day ==="
uv run python scripts/pod_leaky_distill.py --epochs "$EPOCHS" --device cuda

echo "=== [$(date +%H:%M:%S)] ALL DONE — results: ==="
ls -la data/distill_DSK_*.json data/wr_distill_*.json data/leaky_distill_*.json 2>/dev/null
echo "Pull them back with: SCP_FROM root@<pod>:/workspace/mtg-draft-ml/data/'*.json' ./"
echo "‼ Then DELETE the pod:  runpodctl remove pod <id>"
