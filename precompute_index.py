#!/usr/bin/env python3
"""
precompute_index.py - build and cache the semantic (TF-IDF/SVD) index ahead
of the timed ranking step.

This is OPTIONAL. `rank.py` builds the index fresh by default and is still
well within the 5-minute budget (~76s total observed for the full pipeline,
~50-60s of which is the index build). Use this script if you want to
demonstrate the offline-precomputation / timed-ranking-step split explicitly
(e.g. for the methodology writeup, or if pre-computation needs to happen on
a different/faster machine than the timed reproduction run).

Usage:
    python precompute_index.py --candidates ./candidates.jsonl --out ./artifacts/semantic_index.pkl
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv --precomputed-index ./artifacts/semantic_index.pkl
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from semantic import build_semantic_index  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Pre-build the semantic similarity index")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    t0 = time.time()
    print(f"[precompute_index] Loading candidates from {args.candidates} ...", file=sys.stderr)
    candidates = []
    opener = open
    if args.candidates.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(args.candidates, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"[precompute_index] Loaded {len(candidates)} in {time.time()-t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    index = build_semantic_index(candidates)
    print(f"[precompute_index] Built index in {time.time()-t0:.1f}s", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(index, f)
    print(f"[precompute_index] Saved to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
