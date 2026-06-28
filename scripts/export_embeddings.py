"""Dump / query the learned card-embedding space (eval/card_tools.py).

Dev-net (emb64) embeddings, CPU, no pod. Three modes:

    # dump per-set embeddings to data/embeddings/<SET>.npz (E, names, valid) for reuse by other tools
    uv run python scripts/export_embeddings.py --sets DSK,OTJ,BLB

    # "plays like X" — cosine nearest neighbors within a set
    uv run python scripts/export_embeddings.py --similar "Overlord of the Balemurk" --set DSK

    # cross-set analogue — the BLB card most like a DSK card
    uv run python scripts/export_embeddings.py --analogue "Overlord of the Balemurk" --set DSK --to BLB

    # unsupervised archetype map — clusters + signpost cards
    uv run python scripts/export_embeddings.py --archetypes --set DSK --n-clusters 8
"""
from __future__ import annotations

import argparse
import glob
import pathlib

import numpy as np

from mtg_draft_ml.eval import card_tools as ct


def _all_sets(data_dir):
    return sorted({pathlib.Path(p).name.split(".")[0]
                   for p in glob.glob(f"{data_dir}/manifests/*.PremierDraft.sample*.json")})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default=None, help="comma list to DUMP (default: all local sets)")
    ap.add_argument("--similar", default=None, metavar="CARD", help="nearest neighbors of this card")
    ap.add_argument("--analogue", default=None, metavar="CARD", help="cross-set analogue of this card")
    ap.add_argument("--set", dest="set_code", default="DSK", help="set for --similar/--analogue/--archetypes")
    ap.add_argument("--to", dest="to_set", default=None, help="target set for --analogue")
    ap.add_argument("--archetypes", action="store_true", help="cluster the set into archetypes")
    ap.add_argument("--n-clusters", type=int, default=8)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--data-dir", default="data/hf")
    ap.add_argument("--out-dir", default="data/embeddings")
    a = ap.parse_args(argv)

    model = ct.load_dev_model()
    from mtg_draft_ml.cards.text_embed import get_embedder
    emb = get_embedder("all-MiniLM-L6-v2")            # build once, reuse across sets

    if a.similar:
        E, names, valid = ct.embeddings_for_set(model, a.set_code, emb, a.data_dir)
        q, nbrs = ct.neighbors(E, names, a.similar, k=a.k, valid=valid)
        print(f"cards that play like '{q}' ({a.set_code}, dev-net emb):")
        for n, s in nbrs:
            print(f"  {s:.3f}  {n}")
        return

    if a.analogue:
        if not a.to_set:
            raise SystemExit("--analogue needs --to <SET>")
        Ea, na, _ = ct.embeddings_for_set(model, a.set_code, emb, a.data_dir)
        Eb, nb, vb = ct.embeddings_for_set(model, a.to_set, emb, a.data_dir)
        q, nbrs = ct.cross_set_analogues(Ea, na, a.analogue, Eb, nb, k=a.k, valid_b=vb)
        print(f"the {a.to_set} analogues of '{q}' ({a.set_code}):")
        for n, s in nbrs:
            print(f"  {s:.3f}  {n}")
        return

    if a.archetypes:
        E, names, valid = ct.embeddings_for_set(model, a.set_code, emb, a.data_dir)
        clusters = ct.archetype_clusters(E, names, n_clusters=a.n_clusters, valid=valid)
        print(f"{a.set_code} draft archetypes ({a.n_clusters} clusters, signpost cards first):")
        for i, cards in enumerate(clusters):
            print(f"  [{i}] ({len(cards)} cards) " + ", ".join(cards[:6]))
        return

    # default: dump
    sets = ([s.strip() for s in a.sets.split(",")] if a.sets else _all_sets(a.data_dir))
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"dumping dev-net embeddings for {len(sets)} sets -> {out}/")
    for s in sets:
        try:
            E, names, valid = ct.embeddings_for_set(model, s, emb, a.data_dir)
        except FileNotFoundError as e:
            print(f"  {s}: SKIP ({e})"); continue
        np.savez(out / f"{s}.npz", E=E, names=names, valid=valid)
        print(f"  {s}: {E.shape[0]} cards ({int(valid.sum())} valid), emb_dim={E.shape[1]}")


if __name__ == "__main__":
    main()
