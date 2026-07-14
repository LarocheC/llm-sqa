"""
Regenerate the per-clip `target_text` for an existing Plan B corpus using the
current (v2) description generator, WITHOUT re-rendering audio or recomputing
metrics. The per-dimension scores and the fused MOS are unchanged — only the
free-text description (and thus the assembled target) is rewritten with the richer,
more diverse, fully-grounded phrasing.

This is the cheap lever for the description-quality iteration: scores/Stage 1 are
fine, so we only rebuild the Stage 2 targets and retrain Stage 2.

Usage:
  python -m experiments.planb.rebuild_targets \
      --in experiments/results/planb/corpus_train.jsonl \
      --out experiments/results/planb/corpus_train_v2.jsonl
"""

import argparse
import json
import os

import numpy as np

from experiments.planb import targets as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n, lens = 0, []
    with open(args.out, "w") as out:
        for line in open(args.inp):
            r = json.loads(line)
            ctx = {"noise_type": (r["params"].get("noise") or {}).get("noise_type")}
            rng = np.random.default_rng(hash(r["id"]) % 2**32)
            desc = T.describe(r["scores"], rng, ctx)
            r["target_text"] = T.build_target(r["scores"], r["mos"], desc)
            out.write(json.dumps(r) + "\n")
            n += 1
            lens.append(len(desc.split()))
    print(f"rebuilt {n} targets -> {args.out}")
    print(f"description length: mean {np.mean(lens):.1f} words, range {min(lens)}-{max(lens)}")
    # show a few
    recs = [json.loads(l) for l in open(args.out)]
    for r in recs[:3]:
        print("  ", r["target_text"].replace("\n", " || "))


if __name__ == "__main__":
    main()
