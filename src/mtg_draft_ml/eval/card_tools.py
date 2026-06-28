"""Tools over the learned card-embedding space.

The content encoder produces a per-card latent (`ContentDraftModel.card_embeddings()`) trained on draft
value, so cosine-near in this space ≈ similar *draft role / archetype* (not just similar text). This
module exposes that space for analysis tools: nearest-neighbor "plays like X", cross-set analogues, and
unsupervised archetype clustering.

Standalone & CPU (no pod): it uses the local dev-net checkpoint `data/checkpoints/webapp_best.pt`
(emb64) and retargets it per set through the shared, content-based encoder (no standardization, so two
single-set retargets are in a comparable basis). NOTE these are the SMALL dev-net embeddings, NOT the
deployed emb512 model — great for similarity/exploration, but an exact match to the deployed webapp's
picks would need a big-net checkpoint (a one-off pod retrain).
"""
from __future__ import annotations

import glob
import json

import numpy as np

DEV_CKPT = "data/checkpoints/webapp_best.pt"


def load_dev_model(ckpt: str = DEV_CKPT):
    """Load the dev-net ContentDraftModel, inferring its arch from the state dict (robust to config drift)."""
    import torch

    from mtg_draft_ml.models.draft_model import ContentDraftModel

    sd = torch.load(ckpt, map_location="cpu")["model"]
    enc_hidden = sd["card_encoder.body.0.weight"].shape[0]
    emb_dim = sd["card_encoder.proj.weight"].shape[0]
    n_layers = sum(1 for k, v in sd.items()
                   if k.startswith("card_encoder.body") and k.endswith(".weight") and v.dim() == 2)
    aux = any(k.startswith("wr_head") for k in sd)
    m = ContentDraftModel(sd["content"], emb_dim=emb_dim, enc_hidden=enc_hidden, enc_layers=n_layers,
                          pool="set_transformer", n_heads=4, n_sab=1, aux_wr=aux)
    m.load_state_dict(sd, strict=False)   # only card_encoder + content drive embeddings
    m.eval()
    return m


def embeddings_for_set(model, set_code: str, embedder=None, data_dir: str = "data/hf"):
    """Per-card embeddings for one set, retargeted through the shared encoder.

    Returns (E [n, emb], names [n], valid [n] bool). `valid` is False for cards with no Scryfall match
    (all-zero structured features → a single degenerate shared embedding that would otherwise dominate
    neighbors / pollute clusters); callers should skip them.
    """
    import torch

    from mtg_draft_ml.cards.content_table import FEATURE_DIM, build_content_matrix
    from mtg_draft_ml.cards.text_embed import get_embedder

    man = sorted(glob.glob(f"{data_dir}/manifests/{set_code}.PremierDraft.sample*.json"))
    if not man:
        raise FileNotFoundError(f"no manifest for {set_code} under {data_dir}/manifests")
    scry = f"{data_dir}/scryfall/{set_code.lower()}.json"
    embedder = embedder or get_embedder("all-MiniLM-L6-v2")
    mat, _ = build_content_matrix(man[0], scry, embedder=embedder, text=True)
    model.set_content(torch.from_numpy(mat))
    with torch.no_grad():
        E = model.card_embeddings().cpu().numpy()
    names = [c["name"] for c in sorted(json.load(open(man[0]))["cards"], key=lambda c: c["index"])]
    valid = np.linalg.norm(mat[:, :FEATURE_DIM], axis=1) > 1e-9     # structured block all-zero => missing
    return E, np.array(names), valid


def _unit(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def _resolve(names, query):
    idx = {n: i for i, n in enumerate(names)}
    if query in idx:
        return query
    ql = query.lower()
    cand = ([n for n in names if n.lower() == ql]
            or [n for n in names if n.lower().startswith(ql)]
            or [n for n in names if ql in n.lower()])
    if not cand:
        raise KeyError(f"card not found: {query!r}")
    return cand[0]


def neighbors(E, names, query, k: int = 10, valid=None):
    """Cosine nearest neighbors of `query` (card name). Returns (resolved_name, [(name, cos), ...])."""
    names = np.asarray(names)
    q = _resolve(names, query)
    qi = int(np.where(names == q)[0][0])
    En = _unit(E)
    sims = En @ En[qi]
    out = []
    for j in np.argsort(-sims):
        if j == qi or (valid is not None and not valid[j]):
            continue
        out.append((str(names[j]), float(sims[j])))
        if len(out) >= k:
            break
    return q, out


def cross_set_analogues(Ea, names_a, query, Eb, names_b, k: int = 10, valid_b=None):
    """Nearest cards in set B to a card from set A (a comparable-basis cross-set 'analogue')."""
    names_a, names_b = np.asarray(names_a), np.asarray(names_b)
    q = _resolve(names_a, query)
    qv = _unit(Ea)[int(np.where(names_a == q)[0][0])]
    sims = _unit(Eb) @ qv
    out = []
    for j in np.argsort(-sims):
        if valid_b is not None and not valid_b[j]:
            continue
        out.append((str(names_b[j]), float(sims[j])))
        if len(out) >= k:
            break
    return q, out


def archetype_clusters(E, names, n_clusters: int = 10, valid=None, seed: int = 0):
    """KMeans over (cosine-normalized) embeddings → format archetypes; returns [{cards-by-centrality}]."""
    from sklearn.cluster import KMeans

    names = np.asarray(names)
    keep = np.ones(len(names), bool) if valid is None else np.asarray(valid)
    En = _unit(E)[keep]
    nm = names[keep]
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(En)
    out = []
    for c in range(n_clusters):
        members = np.where(km.labels_ == c)[0]
        # signpost cards = closest to the centroid (most central)
        cen = km.cluster_centers_[c]
        order = members[np.argsort(-(En[members] @ cen))]
        out.append([str(nm[i]) for i in order])
    return out
