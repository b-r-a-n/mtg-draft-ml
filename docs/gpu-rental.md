# GPU rental — cheapest path for this project

Buyer's guide for the eventual GPU work (capacity sweeps, optional text-encoder unfreezing). From an
adversarially-verified deep-research pass (2025–2026 pricing). **Prices are spot-market and
fast-changing — treat as a starting prior, re-check live before renting.**

## Our profile (why the recommendation is "cheap single GPU," not a hyperscaler)

- Tiny model (~2–40M params now, ~100–300M if the text encoder is unfrozen) → **one mid-tier 24GB
  GPU** (A5000 / L4 / RTX 4090) is ample. No A100/H100/multi-GPU.
- **CPU-data-loading-bound** → prioritize **high vCPU / RAM / NVMe**, not big VRAM.
- **Bursty** (minutes–hours, ~20–40 hrs/month) → want **per-second billing**, fast spin-up, and
  spot/interruptible with **checkpoint+resume**.
- **Data:** pull ~5–20 GB from Hugging Face → local NVMe; push checkpoints back → **egress fees and
  delete-to-stop storage matter.**

## Verified pricing (single mid-tier GPU, USD/hr, 2025–2026)

| Provider | GPU (24GB unless noted) | On-demand | Spot/interruptible | Billing | Egress |
|---|---|---|---|---|---|
| **RunPod** | A5000 | **$0.27** (community ~$0.16) | — | per-second | **none** |
| RunPod | L4 | $0.39 | — | per-second | none |
| RunPod | RTX 4090 (community) | $0.34 (secure $0.69) | — | per-second | none |
| **Vast.ai** | RTX 4090 | $0.31–0.37 | **often 50%+ off (~$0.10–0.18)** | per-second | **per-byte (up+down)** |
| Salad | RTX 4090 | ~$0.18 (only ~4 vCPU/8GB — bad fit) | — | — | — |
| Modal (serverless) | L4 | ~$0.80 ($0.000222/s) | — | per-second, **no idle fee** | — |
| Modal | T4 | ~$0.59 | — | per-second | — |
| AWS g6.xlarge | L4 (4 vCPU/16GB) | ~$0.805 | ~$0.428 (~47% off) | per-hour-ish | ~$0.09/GB |
| AWS g5.xlarge | A10G | ~$1.01 | — | — | ~$0.09/GB |

Hyperscalers are ~2× the marketplace rate for the same chip **plus** egress — only worth it on free
credits. Free tiers (Colab/Kaggle) are fine for tiny experiments but unreliable for training:
unguaranteed GPU type, ~12h caps, idle timeouts, no SLA.

## Total cost of ownership — the footguns (where cheap $/hr gets expensive)

- **RunPod: zero ingress/egress** ("billed by the second for compute and storage, with no fees for
  data ingress or egress") — the cleanest TCO. *But* persistent/network storage still bills while a
  Pod is **stopped** (~$0.10/GB/mo running, ~$0.20/GB/mo stopped for volume) — delete volumes you
  don't need.
- **Vast.ai: two real footguns** — (1) **bandwidth billed per byte for BOTH upload and download** at
  host-set rates (a 5–20 GB HF pull + checkpoint pushes adds up), and (2) **storage bills until you
  DELETE the instance**, not merely stop it. Lowest headline $/hr, most babysitting.
- **Vast interruptible:** pauses when outbid/preempted, resumes when priority returns — **resume can
  be slow/indefinite**, so checkpoint+resume is mandatory (we already do this).
- **Modal:** no idle fee, no storage babysitting — you pay only for compute seconds. Costs more per
  hour but zero ops overhead; best for fully hands-off sweeps.

## Recommendation (cheapest viable, for ~20–40 hrs/month)

1. **RunPod — best all-round / default.** Cheap 24GB cards, per-second, **zero egress** (matches our
   HF-pull + checkpoint-push flow with no surprise bills), pick a config with **high vCPU/RAM** for
   the data-loading bottleneck. **~$8–12/mo** of compute (A5000 ~$0.27/hr or community 4090 ~$0.34/hr
   × 30 hrs). Start here.
2. **Vast.ai interruptible — cheapest headline $/hr** (~$0.10–0.18/hr 4090) **if** you mind the
   bandwidth charge and remember to *delete* (not stop) instances. **~$5–9/mo** + bandwidth. Use once
   comfortable babysitting spot reclaims.
3. **Modal — hands-off serverless** for bursty sweeps; per-second, no idle/storage management.
   **~$24/mo** (L4 ~$0.80/hr × 30). Pay more for zero ops.

**Net:** for this project, **RunPod on a vCPU-rich 24GB instance** is the cheapest *practical* path
(low $/hr **and** no egress footgun); Vast interruptible is the absolute-cheapest if optimizing hard;
Modal if you'd rather not manage instances at all. Budget is **single-digit-to-low-tens of $/month** —
the GPU work was never going to be expensive; the model is tiny and the bottleneck is data loading.

## Setup sketch (matches docs/data-infra.md; uv-based)

1. Rent a RunPod **PyTorch** pod, 24GB GPU, **high vCPU/RAM**, ~30–50 GB volume.
2. Bootstrap with uv: `REPO=<git-url> bash scripts/runpod_bootstrap.sh` — installs uv, clones,
   `uv venv` + `uv pip install -e ".[dev,hub,embeddings]"`, and checks the GPU is visible.
   (On Linux this pulls the **CUDA** torch wheel automatically.)
3. `huggingface-cli login` (only to push); `uv run python -m mtg_draft_ml.data.pipeline --pull …`
   → local NVMe (zero egress on RunPod); train, checkpointing to an HF model repo.
4. **Delete** the pod/volume when done (storage bills while stopped).

*Pricing verified June 2026 via provider pages + getdeploying.com aggregator; spot rates fluctuate.*
