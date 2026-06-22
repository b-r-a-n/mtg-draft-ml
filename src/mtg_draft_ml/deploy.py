"""Deployable drafter: load a trained model and pick cards, with an aggressiveness dial.

See docs/architecture.md, docs/results/README.md.

`Drafter.pick(pool, pack, aggressiveness)` ranks the cards in `pack` given the cards already in
`pool`. aggressiveness is the win-rate dial (DD-004 / IWD blend):
  - 0.0  -> human-like contextual policy (imitation)
  - >0   -> bias toward higher win-rate cards (logit += aggressiveness * quality[card])
`quality` is a per-card score (e.g. the aux head's predicted IWD, which generalizes to unseen cards,
or external 17lands IWD for sets with data).

A bundle on disk is: model.pt (state_dict + config), content.npy (the card-feature matrix the model
encodes), cards.json (index<->name), quality.npy (per-card dial score). To draft a different set,
rebuild content+cards+quality for it (build_content_matrix) and `retarget()` — the weights are
content-based so they transfer to unseen cards.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from .models.draft_model import ContentDraftModel


class Drafter:
    def __init__(self, model: ContentDraftModel, names: list[str], quality, config: dict,
                 device: str = "cpu"):
        self.model = model.to(device).eval()
        self.names = list(names)
        self.name_to_idx = {n: i for i, n in enumerate(names)}
        # split / DFC: also index the front face
        for i, n in enumerate(names):
            if " // " in n:
                self.name_to_idx.setdefault(n.split(" // ", 1)[0], i)
        # Standardize the dial signal (z-score over rated cards) so `aggressiveness` means the same
        # thing regardless of the quality source's raw scale (raw IWD ~±0.03 vs aux-head outputs).
        q = np.asarray(quality, dtype="float32")
        finite = np.isfinite(q)
        if finite.any():
            mu, sd = float(q[finite].mean()), float(q[finite].std())
            q = (q - mu) / (sd if sd > 1e-6 else 1.0)
        self.quality = torch.as_tensor(np.nan_to_num(q))
        self.config = config
        self.device = device

    # ---- inference ----
    def _resolve(self, card_names):
        idx, unknown = [], []
        for n in card_names:
            j = self.name_to_idx.get(n) or self.name_to_idx.get(n.split(" // ", 1)[0] if " // " in n else n)
            (idx.append(j) if j is not None else unknown.append(n))
        return idx, unknown

    @torch.no_grad()
    def pick(self, pool: list[str], pack: list[str], aggressiveness: float = 0.0,
             top_k: int | None = None):
        """Rank `pack` given `pool`. Returns list of {card, prob, score} sorted best-first."""
        pack_idx, unknown = self._resolve(pack)
        if not pack_idx:
            raise ValueError(f"no known cards in pack (unknown: {unknown})")
        pool_idx, _ = self._resolve(pool)
        dev = self.device
        pool_t = torch.tensor([pool_idx or [0]], device=dev)
        pool_mask = torch.tensor([[bool(pool_idx)] * len(pool_idx or [0])], device=dev)
        if not pool_idx:
            pool_mask = torch.tensor([[False]], device=dev)
        pack_t = torch.tensor([pack_idx], device=dev)
        pack_mask = torch.ones((1, len(pack_idx)), dtype=torch.bool, device=dev)
        logits = self.model(pool_t, pool_mask, pack_t, pack_mask)[0]          # [P]
        if aggressiveness:
            logits = logits + aggressiveness * self.quality.to(dev)[torch.tensor(pack_idx, device=dev)]
        probs = torch.softmax(logits, dim=-1).tolist()
        ranked = sorted(
            ({"card": self.names[c], "prob": probs[k], "score": float(logits[k])}
             for k, c in enumerate(pack_idx)),
            key=lambda r: -r["prob"])
        return ranked[: top_k] if top_k else ranked

    # ---- persistence ----
    def save(self, bundle_dir: str | pathlib.Path):
        d = pathlib.Path(bundle_dir)
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "config": self.config}, d / "model.pt")
        np.save(d / "content.npy", self.model.content.cpu().numpy())
        np.save(d / "quality.npy", self.quality.cpu().numpy())
        json.dump({"names": self.names}, open(d / "cards.json", "w"))
        return d

    @classmethod
    def load(cls, bundle_dir: str | pathlib.Path, device: str = "cpu") -> "Drafter":
        d = pathlib.Path(bundle_dir)
        ckpt = torch.load(d / "model.pt", map_location=device, weights_only=False)
        cfg = ckpt["config"]
        content = torch.from_numpy(np.load(d / "content.npy"))
        model = ContentDraftModel(content, emb_dim=cfg["emb_dim"], enc_hidden=cfg["enc_hidden"],
                                  enc_layers=cfg["enc_layers"], pool=cfg.get("pool", "set_transformer"),
                                  n_heads=cfg.get("n_heads", 4), n_sab=cfg.get("n_sab", 1),
                                  aux_wr=cfg.get("aux_wr", False))
        model.load_state_dict(ckpt["state_dict"])
        names = json.load(open(d / "cards.json"))["names"]
        quality = np.load(d / "quality.npy")
        return cls(model, names, quality, cfg, device=device)

    def retarget(self, content_matrix, names, quality):
        """Point the (content-based) model at a different set's cards — generalizes to unseen cards."""
        self.model.set_content(torch.from_numpy(np.asarray(content_matrix, dtype="float32")))
        self.__init__(self.model, names, quality, self.config, self.device)
        return self
