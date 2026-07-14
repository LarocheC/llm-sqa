#!/usr/bin/env python3
"""Download the Plan B v3 checkpoints from the Hugging Face Hub.

    uv run python scripts/fetch_checkpoints.py            # stage 2 (the final model)
    uv run python scripts/fetch_checkpoints.py --all      # stage 1 + stage 2

Stage 2 is the model you want: calibrated per-dimension scores + description + MOS.
Stage 1 is the intermediate calibration-only checkpoint, published for completeness
(it is what Stage 2 was initialized from).

Lands in $SQA_CKPT_V3_DIR (default experiments/results/planb/ckpt_v3/), where
experiments/config.py:ckpt_v3() looks for it.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import config as cfg  # noqa: E402

FILES = {
    2: "stage2_checkpoint_best.pth",
    1: "stage1_checkpoint_best.pth",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also fetch the stage-1 checkpoint")
    ap.add_argument("--repo", default=cfg.HF_CKPT_REPO)
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("huggingface_hub is missing -> uv sync --extra experiments")

    stages = [2, 1] if args.all else [2]
    cfg.CKPT_V3_DIR.mkdir(parents=True, exist_ok=True)

    for stage in stages:
        name = FILES[stage]
        dest = cfg.CKPT_V3_DIR / name
        if dest.exists():
            print(f"  already have  {dest}")
            continue
        print(f"  downloading   {args.repo}/{name}  (~121 MB) ...")
        path = hf_hub_download(repo_id=args.repo, filename=name)
        # hf_hub_download returns a path in the HF cache; symlink it into place so the
        # repo layout is self-describing and config.ckpt_v3() finds it.
        dest.symlink_to(path)
        print(f"  -> {dest}")

    print("\nReady. Evaluate with:")
    print("  uv run python -m experiments.planb.eval_compare --n-clips 6")


if __name__ == "__main__":
    main()
