#!/usr/bin/env bash
# Create a RunPod GPU pod and print a ready-to-use SSH/scp connection. Removes the friction we hit
# the hard way (see docs/runpod-runbook.md):
#   - auth comes from ~/.runpod/config.toml (runpodctl) — no RUNPOD_API_KEY env var needed.
#   - SECURE cloud is the default: community pods frequently expose only an http port on a PRIVATE
#     IP (no SSH), and `runpodctl ssh info` then sits at "pod not ready" forever.
#   - GPU availability is roulette, so we PROBE a list of --gpuType until one is allocatable.
#   - the public SSH host:port comes from the GraphQL runtime.ports (tcp/22, isIpPublic), NOT from
#     `runpodctl ssh info`.
#
# Usage:   bash scripts/runpod_launch.sh [name]
# Env:     GPUS="a;b;c"  COST=<ceiling>  IMAGE=<img>  DISK=<gb>  CLOUD=secure|community
# Output:  prints POD_ID and the exact ssh/scp commands; writes them to /tmp/runpod-<id>.env
set -euo pipefail

NAME="${1:-mtg-distill}"
CLOUD="${CLOUD:-secure}"                       # secure = reliable public-IP SSH (recommended)
COST="${COST:-0.60}"                           # $/hr price ceiling
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
DISK="${DISK:-30}"
KEY_FILE="$HOME/.runpod/ssh/runpodctl-ssh-key"
API_KEY="$(grep apikey "$HOME/.runpod/config.toml" | head -1 | cut -d"'" -f2)"
# Probe order — mid-tier 24/16GB cards are ample for this tiny model (see docs/gpu-rental.md).
GPUS="${GPUS:-NVIDIA RTX A5000;NVIDIA RTX A4000;NVIDIA RTX 2000 Ada Generation;NVIDIA L4;NVIDIA GeForce RTX 4090;NVIDIA RTX A6000}"

[ -f "$KEY_FILE" ] || { echo "FATAL: $KEY_FILE missing (run: runpodctl ssh add-key)"; exit 1; }
cloud_flag="--secureCloud"; [ "$CLOUD" = community ] && cloud_flag="--communityCloud"

echo ">>> probing $CLOUD cloud for an available GPU (ceil \$$COST/hr)…"
POD=""
IFS=';' read -ra LIST <<< "$GPUS"
for gpu in "${LIST[@]}"; do
  out=$(runpodctl create pod --name "$NAME" --gpuType "$gpu" $cloud_flag \
        --imageName "$IMAGE" --cost "$COST" --gpuCount 1 \
        --containerDiskSize "$DISK" --volumeSize 0 --startSSH --ports "22/tcp" 2>&1 | grep -ivE "deprecated" || true)
  pid=$(echo "$out" | sed -nE 's/.*pod "([a-z0-9]+)" created.*/\1/p')
  rate=$(echo "$out" | sed -nE 's/.*created for \$([0-9.]+).*/\1/p')
  if [ -n "$pid" ]; then POD="$pid"; echo ">>> created $POD on '$gpu' at \$$rate/hr"; break; fi
  echo "    '$gpu' unavailable"
done
[ -n "$POD" ] || { echo "FATAL: no GPU available in $CLOUD right now — retry, raise COST, or set GPUS=."; exit 1; }

echo ">>> waiting for public SSH (tcp/22)…"
HOSTPORT=""
for i in $(seq 1 40); do
  rt=$(curl -s "https://api.runpod.io/graphql?api_key=$API_KEY" -H 'Content-Type: application/json' \
       -d "{\"query\":\"query{pod(input:{podId:\\\"$POD\\\"}){runtime{ports{ip isIpPublic privatePort publicPort type}}}}\"}")
  HOSTPORT=$(echo "$rt" | python3 -c "import sys,json
try:
 r=json.load(sys.stdin)['data']['pod']['runtime']
 [print(p['ip'],p['publicPort']) for p in (r['ports'] if r else []) if p['privatePort']==22 and p['isIpPublic']]
except Exception: pass" | head -1)
  [ -n "$HOSTPORT" ] && break
  sleep 6
done
[ -n "$HOSTPORT" ] || { echo "FATAL: no public SSH after ~4min. Pod $POD created — inspect or 'runpodctl remove pod $POD'."; exit 1; }

HOST=$(echo "$HOSTPORT" | cut -d' ' -f1); PORT=$(echo "$HOSTPORT" | cut -d' ' -f2)
SSHOPTS="-i $KEY_FILE -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
ENVF="/tmp/runpod-$POD.env"
cat > "$ENVF" <<EOF
POD_ID=$POD
SSH="ssh $SSHOPTS root@$HOST"
SCP_TO="scp -i $KEY_FILE -P $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
REMOVE="runpodctl remove pod $POD"
EOF
cat <<EOF

=== POD READY ===  (details saved to $ENVF)
  pod id : $POD     ssh host : $HOST:$PORT
  ssh    : ssh $SSHOPTS root@$HOST
  run    : ssh $SSHOPTS root@$HOST 'cd /workspace && nohup bash mtg-draft-ml/scripts/pod_distill_suite.sh > run.log 2>&1 < /dev/null & echo PID \$!'
           (clone first if needed: git clone --depth 1 https://github.com/b-r-a-n/mtg-draft-ml.git)
  ‼ DONE : runpodctl remove pod $POD          # ALWAYS delete — storage bills even while stopped
EOF
