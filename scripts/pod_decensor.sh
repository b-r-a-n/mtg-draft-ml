#!/usr/bin/env bash
# Multi-set de-censoring sweep on a pod (CPU; sklearn — no GPU, the pod keeps the heavy game_data CSV
# reads + K-fold cross-fit logistic OFF the dev machine). For each set: stream a slim game_data sample,
# run scripts/decensor_curve.py (cross-fit power), save its JSON, free the CSV; then aggregate to a
# cross-set curve verdict.
#
#   ssh … root@<pod> 'cd /workspace && nohup bash mtg-draft-ml/scripts/pod_decensor.sh > dc.log 2>&1 < /dev/null & echo PID $!'
#
# Env: SETS (default the 8 webapp sets, all have cards.json deck_value), SAMPLE (game rows/set, 250000),
#      MAXDECKS (cap, 30000), HF_REPO (scryfall+manifests, b-r-a-n/mtg-draft).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"; export PYTHONUNBUFFERED=1
SETS="${SETS:-DSK OTJ WOE BLB LCI MH3 MOM MKM}"
SAMPLE="${SAMPLE:-250000}"
MAXDECKS="${MAXDECKS:-30000}"
HF_REPO="${HF_REPO:-b-r-a-n/mtg-draft}"
OUT=docs/results/decensor

if [ ! -f pyproject.toml ]; then
  cd /workspace 2>/dev/null || true
  [ -d mtg-draft-ml ] || git clone --depth 1 https://github.com/b-r-a-n/mtg-draft-ml.git
  cd mtg-draft-ml
fi
git pull --ff-only 2>/dev/null || true

echo "=== [$(date +%H:%M:%S)] bootstrap ==="; bash scripts/runpod_bootstrap.sh
echo "=== [$(date +%H:%M:%S)] pull scryfall+manifests ($HF_REPO) ==="
uv run python -m mtg_draft_ml.data.hf pull --repo "$HF_REPO" --out data/hf
mkdir -p "$OUT"

for s in $SETS; do
  echo "=== [$(date +%H:%M:%S)] $s: stream game_data (${SAMPLE} rows) + decensor ==="
  uv run python -c "from mtg_draft_ml.data.download import download_17lands_game as d; print(d('$s', sample_rows=$SAMPLE))" \
    || { echo "  $s: download FAILED, skipping"; continue; }
  uv run python scripts/decensor_curve.py --set "$s" --max-decks "$MAXDECKS" --out "$OUT/$s.json" \
    || echo "  $s: decensor FAILED"
  rm -f data/raw/game.$s.*.csv      # free disk between sets (each sample is hundreds of MB)
done

echo "=== [$(date +%H:%M:%S)] AGGREGATE ==="
uv run python scripts/decensor_aggregate.py --dir "$OUT" --out "$OUT/_aggregate.json"
echo "=== [$(date +%H:%M:%S)] DECENSOR_DONE ==="; ls -la "$OUT"
