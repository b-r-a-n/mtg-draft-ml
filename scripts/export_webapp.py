"""Export a deployable draft model + card metadata for the static WASM draft-pod webapp.

Trains the content draft model (good-player filter + composite-WR teacher, optionally with the
game_data `deck_value` target — the Step-2 recipe), LOSO over the train sets, then bakes the *target*
set's content table into the model and exports it to ONNX so it runs in-browser via onnxruntime-web.
Also emits `cards.json` (per target card: name, rarity, Scryfall image, color identity, deck_value /
GIH / IWD) so the app can generate boosters and drive the aggressiveness dial.

The target set can be held out of training (the honest generalization demo — the model has never seen
those cards) or included. CPU-or-GPU; the small default config trains in minutes for local dev, the
big config (emb512/h1024/L4) matches the Step-2 deployable and is meant for a pod.

    uv run python scripts/export_webapp.py --target DSK --train-sets BLB,OTJ,WOE,MKM \
        --emb-dim 256 --enc-hidden 512 --enc-layers 3 --epochs 6 --teacher value
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from mtg_draft_ml.cards.content_table import build_content_matrix, build_multiset_content
from mtg_draft_ml.cards.text_embed import get_embedder
from mtg_draft_ml.data.dataset import DraftPickDataset, RemappedDataset
from mtg_draft_ml.distill.ensemble import _build_model, _loaders
from mtg_draft_ml.distill.wr import WRSoftmaxTeacher
from mtg_draft_ml.eval.game_value import merged_ratings_with_value
from mtg_draft_ml.eval.winrate import align_winrates, composite_card_quality
from mtg_draft_ml.training.train import pick_device
from mtg_draft_ml.training.train_content import train_loop

D = "data/hf"
SIZE = "60000"
# Fixed export shapes: nn.MultiheadAttention's ONNX export bakes the sequence length, so we export at
# a fixed MAX_POOL/MAX_PACK and the app pads every seat's pool/pack to these (the mask makes padded
# positions inert, so a padded run is numerically identical to the unpadded one). 45 = 3x15 picks.
MAX_POOL = 45
MAX_PACK = 15
TEACHER_FIELDS = {
    "gih": ["ever_drawn_win_rate", "drawn_improvement_win_rate", "avg_pick"],
    "+value": ["ever_drawn_win_rate", "drawn_improvement_win_rate", "avg_pick", "deck_value"],
    "value": ["deck_value"],
    "none": None,
}


def _spec(s, merged_dir):
    base = {
        "parquet": f"{D}/draft/{s}.PremierDraft.sample{SIZE}.parquet",
        "manifest": f"{D}/manifests/{s}.PremierDraft.sample{SIZE}.json",
        "scryfall": f"{D}/scryfall/{s.lower()}.json",
        "ratings_orig": f"{D}/ratings/{s}.PremierDraft.ratings.json",
        "gamevalue": f"{D}/gamevalue/{s}.PremierDraft.gamevalue.json",
    }
    gv = pathlib.Path(base["gamevalue"])
    base["ratings"] = (merged_ratings_with_value(base["ratings_orig"], base["gamevalue"],
                                                 f"{merged_dir}/{s}.merged.json")
                       if gv.exists() else base["ratings_orig"])
    return base


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="DSK", help="set to draft in the app")
    ap.add_argument("--train-sets", default="BLB,OTJ,WOE,MKM")
    ap.add_argument("--target-in-train", action="store_true",
                    help="include the target set in training (default: held out = generalization)")
    ap.add_argument("--teacher", default="value", choices=list(TEACHER_FIELDS),
                    help="composite-WR teacher target fields")
    ap.add_argument("--quality-field", default="deck_value",
                    help="per-card field used for the in-app aggressiveness dial")
    ap.add_argument("--min-winrate", type=float, default=0.55)
    ap.add_argument("--min-games", type=float, default=50)
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument("--enc-hidden", type=int, default=512)
    ap.add_argument("--enc-layers", type=int, default=3)
    ap.add_argument("--n-sab", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quantize", action="store_true",
                    help="int8-quantize the ONNX (~4x smaller, faster WASM) — use for GitHub Pages")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="webapp")
    a = ap.parse_args(argv)

    merged_dir = "data/ratings_merged"
    pathlib.Path(merged_dir).mkdir(parents=True, exist_ok=True)
    train_sets = [s.strip() for s in a.train_sets.split(",") if s.strip() != a.target
                  or a.target_in_train]
    skill = {"min_winrate": a.min_winrate, "min_games": a.min_games, "ranks": None}
    print(f"target={a.target}  train={train_sets}  net=emb{a.emb_dim}/h{a.enc_hidden}/L{a.enc_layers}"
          f"  teacher={a.teacher}  quality={a.quality_field}")

    emb = get_embedder("all-MiniLM-L6-v2")
    train_specs = [_spec(s, merged_dir) for s in train_sets]
    target = _spec(a.target, merged_dir)
    gmat, ginfo, key_to_idx, l2gs = build_multiset_content(train_specs, embedder=emb, text=True)
    tmat, tinfo = build_content_matrix(target["manifest"], target["scryfall"], embedder=emb, text=True)
    dev = pick_device(a.device)

    # train (good players + composite teacher) ------------------------------------------------
    bases = [(RemappedDataset(DraftPickDataset(s["parquet"]), l2g), s["parquet"])
             for s, l2g in zip(train_specs, l2gs)]
    teacher = None
    if TEACHER_FIELDS[a.teacher] is not None:
        rating_specs = [{"manifest": s["manifest"], "ratings": s["ratings"]} for s in train_specs]
        cs, cm = composite_card_quality(rating_specs, l2gs, ginfo["n_cards"],
                                        fields=TEACHER_FIELDS[a.teacher])
        teacher = WRSoftmaxTeacher(cs, cm, tau=1.0, device=dev)
    torch.manual_seed(a.seed)
    tdl, vdl = _loaders(bases, 0.05, split_seed=a.seed, batch_size=512, skill=skill)
    model = _build_model(gmat, dev, emb_dim=a.emb_dim, enc_hidden=a.enc_hidden,
                         enc_layers=a.enc_layers, dropout=0.1, pool="set_transformer",
                         n_heads=4, n_sab=a.n_sab)
    extra = dict(teacher=teacher, distill_lambda=0.5, distill_temp=1.0) if teacher is not None else {}
    train_loop(model, tdl, vdl, dev, tag="webapp", ckpt_prefix="webapp", checkpoint_dir="data/checkpoints",
               checkpoint_every=0, n_cards=ginfo["n_cards"], epochs=a.epochs, lr=1e-3,
               warmup_frac=0.1, grad_clip=1.0, **extra)

    # bake the TARGET set's content + export to ONNX ------------------------------------------
    model.set_content(torch.from_numpy(tmat).to(dev))
    model.eval().to("cpu")
    out = pathlib.Path(a.out)
    (out / "model").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    onnx_path = out / "model" / f"{a.target}.onnx"
    _export_onnx(model, onnx_path)
    if a.quantize:
        _quantize(model, onnx_path)

    # card metadata for booster generation + overlay -----------------------------------------
    cards = _card_metadata(target, a.quality_field)
    (out / "data" / f"{a.target}.cards.json").write_text(json.dumps(cards))

    # real opened packs sampled from the 17lands draft data — the empirically-correct booster
    # distribution (captures Play Booster wildcard/land slots, mythic rate, per-card frequency
    # exactly; no collation modeling). The app deals these and the pod passes them down.
    packs = _real_packs(target["parquet"], n=600)
    (out / "data" / f"{a.target}.packs.json").write_text(json.dumps(packs))
    meta = {"set": a.target, "n_cards": len(cards["cards"]), "n_real_packs": len(packs),
            "onnx": f"model/{a.target}.onnx",
            "quality_field": a.quality_field, "teacher": a.teacher, "train_sets": train_sets,
            "target_in_train": a.target_in_train, "max_pool": MAX_POOL, "max_pack": MAX_PACK,
            "emb_dim": a.emb_dim, "enc_hidden": a.enc_hidden, "enc_layers": a.enc_layers}
    (out / "data" / f"{a.target}.meta.json").write_text(json.dumps(meta, indent=2))
    # index of available sets (the app reads this to populate the set picker)
    sets = sorted({p.name.split(".")[0] for p in (out / "data").glob("*.meta.json")})
    (out / "data" / "sets.json").write_text(json.dumps(sets))
    print(f"\nexported: {onnx_path}  ({onnx_path.stat().st_size/1e6:.1f} MB)")
    print(f"          {out}/data/{a.target}.cards.json  ({len(cards['cards'])} cards)")
    print(f"          sets available: {sets}")


def _export_onnx(model, path):
    # fixed-shape export (see MAX_POOL/MAX_PACK note); only the batch axis stays dynamic so the app
    # can score several seats at once. The app pads pool->MAX_POOL, pack->MAX_PACK with mask=False.
    pool = torch.zeros((1, MAX_POOL), dtype=torch.long)
    pool_mask = torch.zeros((1, MAX_POOL), dtype=torch.bool)
    pack = torch.zeros((1, MAX_PACK), dtype=torch.long)
    pack_mask = torch.ones((1, MAX_PACK), dtype=torch.bool)
    torch.onnx.export(
        model, (pool, pool_mask, pack, pack_mask), str(path),
        input_names=["pool", "pool_mask", "pack", "pack_mask"], output_names=["logits"],
        dynamic_axes={"pool": {0: "B"}, "pool_mask": {0: "B"}, "pack": {0: "B"},
                      "pack_mask": {0: "B"}, "logits": {0: "B"}},
        opset_version=17, dynamo=False)
    _assert_padding_inert(model, path)


def _assert_padding_inert(model, path):
    """The app pads pool/pack with mask=False; verify that's numerically identical to the unpadded
    forward (i.e. masking truly makes padding inert) so bot picks aren't subtly wrong."""
    import numpy as np
    import onnxruntime as ort

    rp, rk = [3, 10, 55, 7, 40], [20, 21, 22, 23, 24]  # a real masked mid-draft pick
    with torch.no_grad():
        ref = model(torch.tensor([rp]), torch.ones(1, len(rp), dtype=torch.bool),
                    torch.tensor([rk]), torch.ones(1, len(rk), dtype=torch.bool)).numpy()[0]
    pool = np.zeros((1, MAX_POOL), np.int64); pool[0, :len(rp)] = rp
    pm = np.zeros((1, MAX_POOL), bool); pm[0, :len(rp)] = True
    pk = np.zeros((1, MAX_PACK), np.int64); pk[0, :len(rk)] = rk
    km = np.zeros((1, MAX_PACK), bool); km[0, :len(rk)] = True
    got = ort.InferenceSession(str(path)).run(
        None, {"pool": pool, "pool_mask": pm, "pack": pk, "pack_mask": km})[0][0, :len(rk)]
    diff = float(np.abs(ref - got).max())
    assert diff < 1e-4, f"padded ONNX != unpadded torch (max diff {diff}) — masking not inert"
    print(f"  padding-inert parity OK (max diff {diff:.2e})")


def _quantize(model, path):
    """int8 dynamic-quantize the ONNX in place (~4x smaller, faster WASM) and verify the quantized
    model still makes the SAME picks (argmax over the pack) as the float32 model on random samples."""
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    f32 = path.with_suffix(".f32.onnx")
    path.rename(f32)
    quantize_dynamic(str(f32), str(path), weight_type=QuantType.QInt8)
    sess = ort.InferenceSession(str(path))
    n_card = model.content.shape[0]
    rng = np.random.default_rng(0)
    agree = 0
    for _ in range(40):
        L, P = int(rng.integers(0, 30)), int(rng.integers(2, MAX_PACK))
        pool = np.zeros((1, MAX_POOL), np.int64); pm = np.zeros((1, MAX_POOL), bool)
        idx = rng.integers(0, n_card, L); pool[0, :L] = idx; pm[0, :L] = True
        pk = np.zeros((1, MAX_PACK), np.int64); km = np.zeros((1, MAX_PACK), bool)
        pkidx = rng.integers(0, n_card, P); pk[0, :P] = pkidx; km[0, :P] = True
        feed = {"pool": pool, "pool_mask": pm, "pack": pk, "pack_mask": km}
        with torch.no_grad():
            ref = model(torch.tensor([list(idx)]) if L else torch.zeros(1, 1, dtype=torch.long),
                        torch.ones(1, max(L, 1), dtype=torch.bool) if L else torch.zeros(1, 1, dtype=torch.bool),
                        torch.tensor([list(pkidx)]), torch.ones(1, P, dtype=torch.bool)).numpy()[0]
        q = sess.run(None, feed)[0][0, :P]
        agree += int(np.argmax(ref) == np.argmax(q))
    size = path.stat().st_size / 1e6
    f32.unlink()
    print(f"  quantized int8 → {size:.1f} MB · pick agreement vs float32: {agree}/40")


def _real_packs(parquet, n=600):
    """Sample real *opened* packs (full 14/15-card boosters) from a set's 17lands draft data.

    A booster = `pack_indices` at the first pick of a pack (pick_number==0). These already encode the
    true Play Booster distribution — wildcard/land slots, mythic rate, per-card frequency — so dealing
    them is the empirically-correct alternative to modeling collation. Returns a list of index lists.
    """
    import pyarrow.parquet as pq

    t = pq.read_table(parquet, columns=["pick_number", "pack_indices"])
    pk = t.column("pick_number").to_numpy()
    packs = t.column("pack_indices")
    out = []
    for r in np.flatnonzero(pk == 0):
        p = packs[r].as_py()
        if len(p) >= 10:                                  # a full opened pack (skip any oddities)
            out.append([int(x) for x in p])
        if len(out) >= n:
            break
    return out


def _card_metadata(target, quality_field):
    """Per-card metadata in manifest-index order: name, rarity, image, colors, deck_value/GIH/IWD."""
    cards = json.load(open(target["manifest"]))["cards"]
    scry = {c.get("name"): c for c in json.load(open(target["scryfall"]))}
    man, rat = target["manifest"], target["ratings"]
    gih = align_winrates(man, rat, field="ever_drawn_win_rate")
    iwd = align_winrates(man, rat, field="drawn_improvement_win_rate")
    try:
        dv = align_winrates(man, rat, field="deck_value")
    except Exception:
        dv = np.full(len(cards), np.nan, dtype=np.float32)
    q = align_winrates(man, rat, field=quality_field)
    out = []
    for c in cards:
        i, name = c["index"], c["name"]
        s = scry.get(name) or scry.get(name.split(" // ", 1)[0], {})
        img = _image_url(s)
        out.append({
            "i": i, "name": name, "rarity": s.get("rarity", "common"),
            "ci": "".join(s.get("color_identity", [])) or "C",
            "cmc": s.get("cmc", 0), "t": _type_cat(s.get("type_line", "")), "img": img,
            "deck_value": _f(dv[i]), "gih": _f(gih[i]), "iwd": _f(iwd[i]), "q": _f(q[i]),
        })
    return {"set": target["manifest"], "quality_field": quality_field, "cards": out}


def _type_cat(type_line):
    """Coarse category for the deck breakdown: land / creature / spell (noncreature nonland)."""
    tl = (type_line or "").split("//")[0]
    if "Land" in tl:
        return "land"
    if "Creature" in tl:
        return "creature"
    return "spell"


def _image_url(s):
    if not s:
        return None
    if s.get("image_uris"):
        return s["image_uris"].get("normal") or s["image_uris"].get("large")
    faces = s.get("card_faces") or []
    for f in faces:
        if f.get("image_uris"):
            return f["image_uris"].get("normal")
    return None


def _f(v):
    v = float(v)
    return None if not np.isfinite(v) else round(v, 5)


if __name__ == "__main__":
    main()
