#!/usr/bin/env python3
"""
rank.py - Redrob Hackathon ranking entry point.

Single command per submission_spec.md Section 10.3:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Compute budget (validated against the real 100K dataset on a single CPU
core in development - see VALIDATION_LOG.md):
  - Loading 100K candidates from JSONL: ~5-13s
  - Building semantic index (TF-IDF + SVD, offline pre-computation): ~50-60s
  - Per-candidate scoring (structured + trust + behavioral + honeypot +
    semantic lookup + reasoning generation): ~20-25s for all 100K
  - Total observed: well under the 5-minute ceiling, with margin for a
    slower/more loaded machine. No GPU used. No network calls made at any
    point in this script.

If `--precomputed-index` is not given, the semantic index is built fresh
inside the timed run (still comfortably inside budget per the timing
above). If you want to demonstrate the offline-precomputation/ranking-step
split explicitly (recommended for the methodology writeup), run
`precompute_index.py` first and pass its output here with
`--precomputed-index`, which skips straight to the ~20s scoring step.
"""
from __future__ import annotations
import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from semantic import build_semantic_index, semantic_scores  # noqa: E402
from final import combine_scores  # noqa: E402
from reasoning import generate_reasoning  # noqa: E402

TOP_N = 100


def load_candidates(path: str) -> list[dict]:
    candidates = []
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Redrob Hackathon candidate ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl (or .jsonl.gz)")
    parser.add_argument("--out", required=True, help="Output path for submission CSV")
    parser.add_argument(
        "--precomputed-index", default=None,
        help="Optional path to a pre-built semantic index pickle (from precompute_index.py). "
             "If omitted, the index is built fresh inside this run (still within the 5-minute budget).",
    )
    args = parser.parse_args()

    t_start = time.time()

    print(f"[rank.py] Loading candidates from {args.candidates} ...", file=sys.stderr)
    candidates = load_candidates(args.candidates)
    print(f"[rank.py] Loaded {len(candidates)} candidates in {time.time()-t_start:.1f}s", file=sys.stderr)

    if args.precomputed_index:
        print(f"[rank.py] Loading pre-computed semantic index from {args.precomputed_index} ...", file=sys.stderr)
        with open(args.precomputed_index, "rb") as f:
            index = pickle.load(f)
    else:
        print("[rank.py] Building semantic index (TF-IDF + SVD) ...", file=sys.stderr)
        t0 = time.time()
        index = build_semantic_index(candidates)
        print(f"[rank.py] Semantic index built in {time.time()-t0:.1f}s", file=sys.stderr)

    sem_scores = semantic_scores(index)

    print("[rank.py] Scoring all candidates ...", file=sys.stderr)
    t0 = time.time()
    cand_by_id = {}
    results = []
    for c in candidates:
        cid = c["candidate_id"]
        cand_by_id[cid] = c
        r = combine_scores(c, sem_scores.get(cid, 0.0))
        results.append(r)
    print(f"[rank.py] Scored {len(results)} candidates in {time.time()-t0:.1f}s", file=sys.stderr)

    results.sort(key=lambda r: -r.final_score)
    top = results[:TOP_N]

    # Tie-break per spec Section 3: equal scores -> candidate_id ascending.
    # Our final_score values are continuous floats from a multi-factor blend
    # so exact ties are unlikely, but we sort defensively to satisfy the
    # validator's monotonicity + tie-break check regardless.
    top.sort(key=lambda r: (-round(r.final_score, 6), r.candidate_id))

    print(f"[rank.py] Writing top {TOP_N} to {args.out} ...", file=sys.stderr)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, r in enumerate(top, start=1):
            c = cand_by_id[r.candidate_id]
            reasoning_text = generate_reasoning(c, r)
            # Scores must be non-increasing by rank and in a sane display
            # range; we expose a rescaled [0,1] display score independent
            # of the internal honeypot sentinel range (final.py uses
            # negative values to force honeypots to the bottom - we clip
            # those to a small positive floor here purely for a clean,
            # monotonic, human-readable score column; rank order itself is
            # unaffected since sorting already happened above).
            display_score = max(0.0001, r.final_score)
            writer.writerow([r.candidate_id, rank, f"{display_score:.4f}", reasoning_text])

    n_honeypots = sum(1 for r in top if r.is_honeypot)
    elapsed = time.time() - t_start
    print(f"[rank.py] Done in {elapsed:.1f}s total. Honeypots in top {TOP_N}: {n_honeypots}", file=sys.stderr)
    if n_honeypots > TOP_N * 0.10:
        print(
            f"[rank.py] WARNING: honeypot rate {n_honeypots}/{TOP_N} exceeds the 10% "
            f"disqualification threshold!",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
