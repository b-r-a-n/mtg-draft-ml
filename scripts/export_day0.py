"""Export a day-0 JSON for the webapp from LLM teacher scores.

Usage
-----
# From the teacher_cache subdir (dict shape: {name: score})
python3 scripts/export_day0.py --set DSK --teacher agent-claude

# From a ratings file (list shape: [{name, llm_quality}])
python3 scripts/export_day0.py --set DSK --ratings-file docs/results/ws24/agent-claude.DSK.ratings.json

Output: webapp/data/<SET>.day0.json
  {
    "set": "DSK",
    "source": "agent-claude",          # or "file:<basename>"
    "alpha_recommended": 4.0,          # WS2.4 finding: alpha=4 closes ~20% of the day-0 gap
    "scores": {"<card name>": 7.0, …}  # raw 0-10 scores; app z-scores at load time
  }

The alpha_recommended default (4.0) is the WS2.4 finding: blending z-scored LLM teacher
quality at alpha≈4 closes ~20% of the oracle–baseline WR-agreement gap on DSK.
"""
import argparse, json, os, pathlib, statistics


ALPHA_RECOMMENDED = 4.0   # WS2.4 best result; see docs/results/ws24-coldstart.md


def load_dict_shape(path: str) -> dict[str, float]:
    """Load {name: score} dict directly."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict shape at {path}, got {type(data)}")
    return {k: float(v) for k, v in data.items()}


def load_list_shape(path: str) -> dict[str, float]:
    """Load [{name, llm_quality}] list shape (ws24 ratings format)."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list shape at {path}, got {type(data)}")
    out = {}
    for entry in data:
        name = entry.get("name") or entry.get("card_name")
        score = entry.get("llm_quality") if entry.get("llm_quality") is not None else entry.get("score")
        if name is None or score is None:
            raise ValueError(f"Entry missing name or score/llm_quality: {entry}")
        out[name] = float(score)
    return out


def main():
    repo_root = pathlib.Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Export day-0 webapp JSON from LLM teacher scores.")
    parser.add_argument("--set", required=True, metavar="SET", help="Set code, e.g. DSK")
    parser.add_argument("--teacher", default=None, metavar="TEACHER",
                        help="Teacher name under data/teacher_cache/<teacher>/<SET>.json (dict shape)")
    parser.add_argument("--ratings-file", default=None, metavar="PATH",
                        help="Path to a ratings file in list shape [{name, llm_quality}]")
    parser.add_argument("--alpha", type=float, default=ALPHA_RECOMMENDED,
                        help=f"Recommended alpha (default: {ALPHA_RECOMMENDED}; WS2.4 finding)")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Output path (default: webapp/data/<SET>.day0.json)")
    args = parser.parse_args()

    if args.teacher and args.ratings_file:
        parser.error("Specify either --teacher or --ratings-file, not both.")
    if not args.teacher and not args.ratings_file:
        parser.error("Specify one of --teacher or --ratings-file.")

    # Load scores
    if args.teacher:
        cache_path = repo_root / "data" / "teacher_cache" / args.teacher / f"{args.set}.json"
        if not cache_path.exists():
            # Try the flat naming convention used by distill/teacher.py
            alt = repo_root / "data" / "teacher_cache" / f"{args.set}.cached_{args.teacher}.ratings.json"
            if alt.exists():
                scores = load_list_shape(str(alt))
            else:
                raise FileNotFoundError(
                    f"Could not find teacher cache at {cache_path} or {alt}.\n"
                    "Run the teacher scoring workflow first (scripts/pod_coldstart.py teacher build)."
                )
        else:
            scores = load_dict_shape(str(cache_path))
        source = args.teacher
    else:
        ratings_path = pathlib.Path(args.ratings_file)
        if not ratings_path.is_absolute():
            ratings_path = repo_root / ratings_path
        if not ratings_path.exists():
            raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
        # Detect shape
        with open(ratings_path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            scores = load_list_shape(str(ratings_path))
        else:
            scores = load_dict_shape(str(ratings_path))
        source = f"file:{ratings_path.name}"

    # Output path
    out_path = pathlib.Path(args.out) if args.out else repo_root / "webapp" / "data" / f"{args.set}.day0.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "set": args.set,
        "source": source,
        "alpha_recommended": args.alpha,
        "scores": scores,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Print stats
    vals = list(scores.values())
    print(f"Wrote {out_path}")
    print(f"  set={args.set}  source={source}  alpha={args.alpha}")
    print(f"  count={len(vals)}  min={min(vals):.2f}  mean={statistics.mean(vals):.2f}  max={max(vals):.2f}")


if __name__ == "__main__":
    main()
