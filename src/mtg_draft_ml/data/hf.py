"""Hugging Face Hub integration: push/pull the compact dataset (DD-007, docs/data-infra.md).

HF is the hub between CPU preprocessing and GPU training. This module is a thin wrapper around
`huggingface_hub`; it never touches the model code. The dataset format and `DraftPickDataset`
stay path-based and oblivious to HF.

Repo layout (a *dataset* repo):
    draft/<SET>.<EVENT>.parquet
    manifests/<SET>.<EVENT>.json
    cards/{features,text_embeddings}.parquet   # Phase 1

Auth: reads of a public repo need no token; pushes need one (`huggingface-cli login` or HF_TOKEN).
"""
from __future__ import annotations

import pathlib

DEFAULT_REPO = "mtg-draft"  # set to "<user>/mtg-draft" in configs/CLI
# Canonical dataset layout. Precise per-dir patterns (NOT a broad top-level *.json, which would
# sweep in experiment-output cruft and miss ratings/).
_ALL_PATTERNS = [
    "draft/*.parquet",      # compact integer-index pick shards
    "manifests/*.json",     # per-set card vocab
    "scryfall/*.json",      # per-set Scryfall card records (for the content encoder)
    "ratings/*.json",       # 17lands GIH-WR ratings (for WR-agreement + aux-WR)
    "gamevalue/*.json",     # game_data per-card marginal-value field (eval.game_value, Step 1)
    "cards/*.parquet",      # precomputed feature/embedding tables (future)
]


def shard_patterns(tag: str) -> list[str]:
    """allow_patterns for a single set shard + its manifest (tag = '<SET>.<EVENT>[.sampleN]')."""
    return [f"draft/{tag}.parquet", f"manifests/{tag}.json"]


def all_patterns() -> list[str]:
    return list(_ALL_PATTERNS)


def _require_hub():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'huggingface_hub is required: uv pip install -e ".[hub]"'
        ) from e
    import huggingface_hub
    return huggingface_hub


def push_dataset(
    processed_dir: str | pathlib.Path,
    repo_id: str = DEFAULT_REPO,
    patterns: list[str] | None = None,
    private: bool = False,
    token: str | None = None,
    commit_message: str = "update compact draft dataset",
) -> str:
    """Upload processed_dir contents (matching `patterns`) to a HF dataset repo. Returns repo URL."""
    hub = _require_hub()
    api = hub.HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(processed_dir),
        allow_patterns=patterns or all_patterns(),
        commit_message=commit_message,
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def pull_dataset(
    repo_id: str = DEFAULT_REPO,
    local_dir: str | pathlib.Path = "data/processed",
    patterns: list[str] | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> pathlib.Path:
    """Download dataset files (matching `patterns`) to local_dir. Returns the local path.

    Pin `revision` (a commit SHA or tag) for reproducible splits. Public repos read tokenlessly.
    """
    hub = _require_hub()
    local_dir = pathlib.Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    hub.snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=patterns or all_patterns(),
        revision=revision,
        token=token,
    )
    return local_dir


def push_file(
    local_path: str | pathlib.Path,
    path_in_repo: str,
    repo_id: str,
    repo_type: str = "model",
    token: str | None = None,
    commit_message: str = "upload",
) -> str:
    """Upload a single file (e.g. a training checkpoint to a model repo)."""
    hub = _require_hub()
    api = hub.HfApi(token=token)
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"


def main(argv=None):  # pragma: no cover - thin CLI
    import argparse

    ap = argparse.ArgumentParser(description="Push/pull the compact draft dataset to/from HF")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("push", help="upload processed_dir to a HF dataset repo")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--private", action="store_true")

    q = sub.add_parser("pull", help="download a HF dataset repo to a local dir")
    q.add_argument("--repo", default=DEFAULT_REPO)
    q.add_argument("--out", default="data/processed")
    q.add_argument("--revision", default=None)

    a = ap.parse_args(argv)
    if a.cmd == "push":
        print(push_dataset(a.processed_dir, a.repo, private=a.private))
    else:
        print(pull_dataset(a.repo, a.out, revision=a.revision))


if __name__ == "__main__":  # pragma: no cover
    main()
