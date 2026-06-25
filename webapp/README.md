# Draft Pod webapp — in-browser model overlay (static, WASM)

An 8-seat draft pod that runs the content draft model **entirely in the browser** via
onnxruntime-web (WASM). You draft seat 1; the model overlays its read on every pick and auto-drafts
the other 7 seats at per-seat aggressiveness, so passing/signals emerge like a real pod. No server
does inference — it's a static site (a dumb file server is only needed because browsers block
`fetch()` from `file://`).

## 1. Export a set (model → ONNX + card metadata)

```bash
uv run python scripts/export_webapp.py --target DSK --train-sets BLB,OTJ,WOE,MKM \
    --emb-dim 256 --enc-hidden 512 --enc-layers 3 --epochs 6 --teacher value
```

- Trains the content model (good-player filter + composite-WR teacher; `--teacher value|+value|gih`
  picks whether the **game_data `deck_value`** target is used), LOSO over `--train-sets`, then bakes
  the **`--target`** set's content into the model and exports `webapp/model/<SET>.onnx`.
- The target set is **held out of training by default** (honest generalization — the model has never
  seen those cards); pass `--target-in-train` to include it.
- Emits `webapp/data/<SET>.cards.json` (name, rarity, Scryfall image, color identity,
  `deck_value`/GIH/IWD) and `<SET>.meta.json`. Re-run for more sets; the app's picker reads them all.
- Emits `<SET>.packs.json` — a library of **real opened packs** sampled from the 17lands draft data.
  Modern **Play Boosters** (14 cards: 6 commons, 1 common/List, 3 uncommons, 1 rare-or-mythic, 1 land,
  2 any-rarity wildcards) don't reduce to a fixed rare/uncommon/common split — the wildcard and land
  slots make real packs vary (8–10 C / 3–5 U / 1–2 R+M in our DSK data). Rather than model that
  collation, the app **deals real packs**, which reproduces the true distribution (wildcards, land
  slot, mythic rate, per-card frequency) exactly. The synthetic Play-Booster generator in `app.js` is
  only a fallback when no `packs.json` is present.
- Self-checks ONNX↔PyTorch parity and that mask-padding is inert (the app pads pool→45, pack→15).

## 2. Serve it locally

```bash
cd webapp && python -m http.server 8011
# open http://localhost:8011
```

## 3. Deploy to GitHub Pages

Fully static (model runs in-browser via WASM, images from Scryfall's CDN), so Pages just serves the
`webapp/` folder — done by `.github/workflows/pages.yml` on every push that touches `webapp/`.

1. Export with **`--quantize`** so the committed model is ~9 MB, not ~35 MB:
   ```bash
   uv run python scripts/export_webapp.py --target DSK --train-sets BLB,OTJ,WOE,MKM \
       --emb-dim 512 --enc-hidden 1024 --enc-layers 4 --epochs 10 --teacher value --quantize
   ```
   (int8 quantization is verified to keep the same picks as float32.) Commit `webapp/model/*.onnx`
   and `webapp/data/*.json`.
2. In the repo: **Settings → Pages → Build and deployment → Source: "GitHub Actions"** (one-time).
3. Push. The workflow publishes to `https://<you>.github.io/<repo>/`. Relative asset paths mean it
   works under that subpath unchanged.

## Using it

- **New draft** deals 3 packs around the pod. Your pack shows each card with the model's pick
  probability (bar + %), its top pick (★, gold border), and the card's `deck_value`.
- **Your overlay aggressiveness** dials how much the model's recommendation weights `deck_value`
  (the de-confounded win signal) over raw pick policy — the Step-2 lever, live.
- **Bot aggressiveness spread** sets the max aggressiveness; each bot gets its own value in `[0, max]`.
- Click a card to pick; bots pick simultaneously and the packs pass. **Open (passed to you)** hints at
  which colors are flowing. The end screen scores your pool by avg `deck_value`, how often you matched
  the model, and how much value you left in packs.

## Files

- `index.html` / `app.css` / `app.js` — the static app (app.js holds the pod engine + WASM inference).
- `model/<SET>.onnx` — the exported draft model (content baked for that set).
- `data/<SET>.{cards,meta}.json`, `data/sets.json` — card metadata + set index.
