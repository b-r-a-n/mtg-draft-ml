#!/usr/bin/env bash
# First end-to-end run on a fresh RunPod pod: pull data from HF, then run the best-config LOSO
# benchmark on the GPU. Assumes you've already cloned the repo and run scripts/runpod_bootstrap.sh
# (so uv + the env exist). Run from the repo root:  bash scripts/pod_first_run.sh
#
# Reproduces our best recipe (content encoder -> Set Transformer -> in-pack CE + aux-WR) on
# 4 training sets, holding out DSK. Expect held-out top-1 ~0.57, far faster than the M1 laptop.
#
# AUTODELETE=1  -> terminate THIS pod via runpodctl after the run finishes (kills compute+storage
#                 billing automatically). Off by default. The result JSON is pushed to HF first
#                 (if HF_MODEL_REPO is set) so it survives the pod's deletion.
set -euo pipefail
cd "$(dirname "$0")/.."

# uv installs to ~/.local/bin, which non-interactive shells don't add to PATH — ensure it's there.
export PATH="$HOME/.local/bin:$PATH"

HF_REPO="${HF_REPO:-b-r-a-n/mtg-draft}"
HF_MODEL_REPO="${HF_MODEL_REPO:-}"   # e.g. b-r-a-n/mtg-draft-bot — to save the result off-pod
D="${D:-data/hf}"
SIZE="${SIZE:-60000}"
EPOCHS="${EPOCHS:-10}"
AUTODELETE="${AUTODELETE:-0}"

# 1. Pull the dataset (public repo -> anonymous, no token needed; zero egress on RunPod).
echo ">>> pulling $HF_REPO -> $D"
uv run python - "$HF_REPO" "$D" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(
    repo, repo_type="dataset", local_dir=dest, token=False,
    allow_patterns=["draft/*.parquet", "manifests/*.json", "scryfall/*.json", "ratings/*.json"],
)
print("pulled ->", dest)
PY

# 2. Build a per-set spec "parquet,manifest,scryfall[,ratings]" for the CLI.
spec() {  # $1 = SET code
  local s="$1" lc
  lc="$(echo "$s" | tr '[:upper:]' '[:lower:]')"
  echo "$D/draft/$s.PremierDraft.sample$SIZE.parquet,$D/manifests/$s.PremierDraft.sample$SIZE.json,$D/scryfall/$lc.json,$D/ratings/$s.PremierDraft.ratings.json"
}

# 3. LOSO benchmark: train on 4 sets, hold out DSK (best config).
echo ">>> training (set_transformer + ce + aux-WR), holdout DSK"
uv run python -m mtg_draft_ml.eval.generalization \
  --train "$(spec BLB)" \
  --train "$(spec OTJ)" \
  --train "$(spec WOE)" \
  --train "$(spec MKM)" \
  --holdout "$(spec DSK)" \
  --pool set_transformer --loss ce --aux-wr 1.0 \
  --holdout-ratings "$D/ratings/DSK.PremierDraft.ratings.json" \
  --epochs "$EPOCHS" --out-json pod_first_run_result.json

echo
echo ">>> done. Result JSON: pod_first_run_result.json"

# 4. Optionally save the result off-pod (so it survives deletion).
if [ -n "$HF_MODEL_REPO" ]; then
  echo ">>> uploading result to $HF_MODEL_REPO (needs: hf auth login)"
  uv run hf upload "$HF_MODEL_REPO" pod_first_run_result.json pod_first_run_result.json \
    --repo-type model || echo "WARN: upload failed (not logged in?); result is still on the pod"
fi

# 5. Optionally self-terminate to stop all billing.
if [ "$AUTODELETE" = "1" ]; then
  if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo ">>> AUTODELETE=1 -> terminating pod $RUNPOD_POD_ID in 10s (Ctrl-C to cancel)"
    [ -z "$HF_MODEL_REPO" ] && echo "    NOTE: HF_MODEL_REPO unset — pod_first_run_result.json will be LOST on delete."
    sleep 10
    runpodctl remove pod "$RUNPOD_POD_ID"
  else
    echo "WARN: AUTODELETE=1 but runpodctl or \$RUNPOD_POD_ID unavailable — delete the pod manually."
  fi
else
  echo ">>> REMEMBER to DELETE the pod + volume when finished (storage bills while stopped)."
  echo "    (or re-run with AUTODELETE=1 to self-terminate; set HF_MODEL_REPO to save results first)"
fi
