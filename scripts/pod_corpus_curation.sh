#!/usr/bin/env bash
# Corpus-curation run on a fresh RunPod pod: bootstrap -> pull data -> train the 4 curation arms.
# Separates corpus RELEVANCE from COUNT for WR-agreement (see scripts/pod_corpus_curation.py docstring
# and docs/results/wr-scaling.md). Holdout DSK, multi-seed, big-net recipe.
#
# Launch detached so it survives an SSH drop (note `< /dev/null`):
#   ssh … root@<pod> 'cd /workspace && nohup bash mtg-draft-ml/scripts/pod_corpus_curation.sh \
#       > run.log 2>&1 < /dev/null & echo PID $!'
# then poll: ssh … root@<pod> 'tail -40 /workspace/run.log'
#
# Env: SEEDS (default 0,1,2), EPOCHS (default 10), HF_REPO (data, default b-r-a-n/mtg-draft),
#      HF_MODEL_REPO (save result off-pod, e.g. b-r-a-n/mtg-draft-bot), AUTODELETE (1 = self-terminate).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1   # stream per-epoch prints to run.log live (else block-buffered to a file)
SEEDS="${SEEDS:-0,1,2}"
EPOCHS="${EPOCHS:-10}"
HF_REPO="${HF_REPO:-b-r-a-n/mtg-draft}"
HF_MODEL_REPO="${HF_MODEL_REPO:-}"
AUTODELETE="${AUTODELETE:-0}"
OUT="data/corpus_curation_DSK.json"

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

echo "=== [$(date +%H:%M:%S)] corpus curation (4 arms x seeds $SEEDS, holdout DSK, cuda) ==="
uv run python scripts/pod_corpus_curation.py --seeds "$SEEDS" --epochs "$EPOCHS" --device cuda

echo "=== [$(date +%H:%M:%S)] DONE — result: ==="
ls -la "$OUT" 2>/dev/null

# save result off-pod so it survives deletion
if [ -n "$HF_MODEL_REPO" ] && [ -f "$OUT" ]; then
  echo ">>> uploading $OUT to $HF_MODEL_REPO (needs: hf auth login)"
  uv run hf upload "$HF_MODEL_REPO" "$OUT" corpus_curation_DSK.json --repo-type model \
    || echo "WARN: upload failed (not logged in?); result is still on the pod"
fi

if [ "$AUTODELETE" = "1" ]; then
  if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
    [ -z "$HF_MODEL_REPO" ] && echo "    NOTE: HF_MODEL_REPO unset — $OUT will be LOST on delete."
    echo ">>> AUTODELETE=1 -> terminating pod $RUNPOD_POD_ID in 10s (Ctrl-C to cancel)"
    sleep 10
    runpodctl remove pod "$RUNPOD_POD_ID"
  else
    echo "WARN: AUTODELETE=1 but runpodctl or \$RUNPOD_POD_ID unavailable — delete the pod manually."
  fi
else
  echo ">>> REMEMBER to DELETE the pod when finished (storage bills while stopped)."
fi
