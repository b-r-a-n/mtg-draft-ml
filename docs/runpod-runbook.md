# RunPod runbook — launch a GPU pod and run a job (low-friction)

Operational steps + the non-obvious gotchas (learned the hard way). For *which* GPU to rent and the
cost rationale, see [gpu-rental.md](gpu-rental.md). For the on-pod env build, see
`scripts/runpod_bootstrap.sh`.

## TL;DR — the whole flow in three commands

```bash
# 1. create a pod + get a ready-to-use SSH connection (probes GPUs, prints ssh/scp, waits for SSH)
bash scripts/runpod_launch.sh            # writes /tmp/runpod-<id>.env  (POD_ID, SSH, SCP_TO, REMOVE)
source /tmp/runpod-<id>.env

# 2. run the distillation suite detached (bootstrap -> pull data -> 3 experiments)
$SSH 'cd /workspace && git clone --depth 1 https://github.com/b-r-a-n/mtg-draft-ml.git 2>/dev/null; \
      nohup bash mtg-draft-ml/scripts/pod_distill_suite.sh > run.log 2>&1 < /dev/null & echo PID $!'
$SSH 'tail -40 /workspace/run.log'       # poll until "ALL DONE"

# 3. pull results back, then ALWAYS delete the pod (storage bills even while stopped)
$SCP_TO root@$HOST:/workspace/mtg-draft-ml/data/'*.json' ./data/      # or via HF
eval "$REMOVE"                            # runpodctl remove pod <id>
```

## Auth — no env var needed

`runpodctl` reads the API key from `~/.runpod/config.toml` (set once via `runpodctl doctor`). You do
**not** need `RUNPOD_API_KEY` in the env — a missing env var does not mean unauthenticated. Verify
with `runpodctl pod list`. The SSH keypair is `~/.runpod/ssh/runpodctl-ssh-key{,.pub}`; confirm it's
registered on the account with `runpodctl ssh list-keys`.

## Gotchas (each one cost real time)

1. **Use SECURE cloud for SSH.** Community pods frequently expose **only an `http` port on a private
   IP** (`isIpPublic:false`) — no SSH at all — even with `--startSSH --ports 22/tcp`. Symptom:
   `runpodctl ssh info` sits at `{"error":"pod not ready"}` indefinitely while the pod is `RUNNING`.
   `--secureCloud` gives a public IP + real `tcp/22`. (Community is cheaper if you only need the
   http/Jupyter port, but not for our SSH-driven flow.)

2. **`runpodctl ssh info` is unreliable; get SSH from the GraphQL runtime instead.** The public
   host:port is in `pod.runtime.ports` where `privatePort==22 && isIpPublic`:
   ```bash
   KEY=$(grep apikey ~/.runpod/config.toml | cut -d"'" -f2)
   curl -s "https://api.runpod.io/graphql?api_key=$KEY" -H 'Content-Type: application/json' \
     -d '{"query":"query{pod(input:{podId:\"<POD>\"}){runtime{ports{ip isIpPublic privatePort publicPort type}}}}"}'
   ```
   `scripts/runpod_launch.sh` does this polling for you.

3. **GPU availability is roulette.** `create pod` fails with "no longer any instances available" /
   "machine does not have the resources" per GPU type. Probe a list of `--gpuType` until one is
   allocatable (the launch script does). Any mid-tier 16–24GB card (A5000 / A4000 / RTX 2000 Ada /
   L4 / 4090) is ample for this tiny model.

4. **zsh does not word-split unquoted variables.** `SSHOPTS="-i k -p 22512 …"; ssh $SSHOPTS host`
   sends the whole string as one arg in zsh → "Identity file … not accessible / Host key
   verification failed". **Inline the ssh options**, or use `${=SSHOPTS}` / a bash array.

5. **`nohup … &` over SSH hangs without `< /dev/null`.** The ssh channel stays open waiting on the
   inherited stdin, so your *local* command never returns (the remote job is fine). Always redirect:
   `nohup cmd > log 2>&1 < /dev/null &`.

6. **The bootstrap installs `.[dev,hub,embeddings]`** — enough for the ensemble / WR-softmax / leaky
   experiments (they need the MiniLM encoder, not `anthropic`). Only the **cold-start Anthropic
   teacher** needs the `[distill]` extra + `ANTHROPIC_API_KEY`; add it explicitly if you run that.

7. **Local Apple-Silicon caveat (why we go to RunPod at all).** The dev M1 has **8 GB** unified
   memory: the multi-set Set-Transformer + MiniLM run thrashes/swaps at large batch (batch 1536 hung
   with zero epochs). Locally keep `batch_size ≤ 512`; on a 16–24GB CUDA pod the default 512 is fast
   and fine. This is the "step intended for a rented GPU" (DD-006).

8. **Python stdout is block-buffered to a file** (not a tty), so per-epoch prints don't appear in
   `run.log` until the buffer fills or the process exits — looks stalled but isn't. Confirm progress
   with `nvidia-smi` (GPU util > 0) and the growing log byte count; for live lines export
   `PYTHONUNBUFFERED=1` (the suite script does).

9. **ALWAYS delete the pod when done** — `runpodctl remove pod <id>`. Storage bills even while the
   pod is *stopped*; only removal stops all charges. `pod_first_run.sh` supports `AUTODELETE=1` to
   self-terminate (push results to HF first so they survive). Total cost for the distillation suite
   is **pennies** (minutes of a ~$0.24/hr card).

## Driving the pod

- **Bootstrap:** `bash scripts/runpod_bootstrap.sh` (idempotent — uv, CUDA torch pinned via
  `pyproject.toml [tool.uv.sources]`, deps, and a fatal GPU-visibility check).
- **CPU-only jobs** (de-censoring sweep, playprob export — any sklearn/pandas work): use
  `scripts/runpod_bootstrap_cpu.sh` instead. The default bootstrap pulls the 2-3 GB CUDA torch wheel
  and asserts a GPU; for CPU work that's wasted and **has hung on pod start mid-download** (cost a real
  session). The CPU bootstrap installs CPU torch (~200 MB, needed by `game_value.fit_card_values`' torch
  L-BFGS) + the runtime deps and is idempotent. **Run the venv python directly** —
  `PYTHONPATH=src .venv/bin/python scripts/foo.py` — NOT `uv run` (auto-syncs → re-pulls the CUDA pin)
  and NOT `uv pip install -e .` (same pin). See `pod_decensor.sh` / `pod_playprob.sh` for the pattern.
- **Run the suite:** `scripts/pod_distill_suite.sh` (bootstrap → pull `b-r-a-n/mtg-draft` → the 3
  experiments). For the single best-recipe LOSO benchmark instead, use `scripts/pod_first_run.sh`.
- **Get results off-pod:** `scp` the `data/*.json` back, or `huggingface-cli login` + push to an HF
  model repo (zero egress on RunPod) so they survive deletion.
```
