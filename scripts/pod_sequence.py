"""Phase-5 sequence model LOSO on the pod: does signal-reading beat the ~0.58 ceiling?

Same 4-set -> DSK setup as the Phase-1/2/3 baselines (set-model: mean-pool 0.552, Set Transformer
0.563, +aux-WR 0.574), so the comparison is apples-to-apples on identical data. MiniLM text encoder.
"""
from mtg_draft_ml.eval.sequence_eval import run_loso_sequence

D = "data/hf"; SIZE = "60000"
TRAIN = ["BLB", "OTJ", "WOE", "MKM"]
HOLDOUT = "DSK"


def spec(s):
    return {"parquet": f"{D}/draft/{s}.PremierDraft.sample{SIZE}.parquet",
            "manifest": f"{D}/manifests/{s}.PremierDraft.sample{SIZE}.json",
            "scryfall": f"{D}/scryfall/{s.lower()}.json"}


run_loso_sequence(
    [spec(s) for s in TRAIN], spec(HOLDOUT),
    embedder="all-MiniLM-L6-v2", emb_dim=256, enc_hidden=512, enc_layers=3,
    n_layers=2, n_heads=4, epochs=12, lr=1e-3, batch_size=16, warmup_frac=0.1, grad_clip=1.0,
    seed=0, out_json="/workspace/sequence_result.json",
)
