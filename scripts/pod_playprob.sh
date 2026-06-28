#!/usr/bin/env bash
# Generate the webapp buildability-deckbuilder artifacts (P(played|pool) as JSON trees) on a pod's CPU.
# NOTE: this is sklearn HistGradientBoosting — CPU-only, NO GPU speedup; the pod is purely to keep the
# training (and the big game_data CSV reads) OFF the dev machine. Streams slim game_data samples (NOT the
# multi-GB full files), trains + parity-checks per set, writes webapp/model/<SET>.playprob.json.
#
#   ssh … root@<pod> 'cd /workspace && nohup bash mtg-draft-ml/scripts/pod_playprob.sh > pp.log 2>&1 < /dev/null & echo PID $!'
#
# Env: SETS (space-list, default the 8 webapp sets), SAMPLE (game rows/set, default 150000 to match local),
#      HF_REPO (manifests, default b-r-a-n/mtg-draft).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"; export PYTHONUNBUFFERED=1
SETS="${SETS:-BLB DSK LCI MH3 MKM MOM OTJ WOE}"
SAMPLE="${SAMPLE:-150000}"
HF_REPO="${HF_REPO:-b-r-a-n/mtg-draft}"

if [ ! -f pyproject.toml ]; then
  cd /workspace 2>/dev/null || true
  [ -d mtg-draft-ml ] || git clone --depth 1 https://github.com/b-r-a-n/mtg-draft-ml.git
  cd mtg-draft-ml
fi
git pull --ff-only 2>/dev/null || true

echo "=== [$(date +%H:%M:%S)] bootstrap ==="; bash scripts/runpod_bootstrap.sh
echo "=== [$(date +%H:%M:%S)] pull manifests ($HF_REPO) ==="
uv run python -m mtg_draft_ml.data.hf pull --repo "$HF_REPO" --out data/hf
echo "=== [$(date +%H:%M:%S)] stream slim game_data samples (${SAMPLE} rows/set) ==="
for s in $SETS; do
  uv run python -c "from mtg_draft_ml.data.download import download_17lands_game as d; print(d('$s', sample_rows=$SAMPLE))"
done
echo "=== [$(date +%H:%M:%S)] export playprob (train + parity per set) ==="
uv run python scripts/export_playprob.py --sets "$(echo "$SETS" | tr ' ' ',')"
echo "=== [$(date +%H:%M:%S)] PLAYPROB_DONE ==="
ls -la webapp/model/*.playprob.json
