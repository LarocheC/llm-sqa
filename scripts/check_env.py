#!/usr/bin/env python3
"""Environment doctor for the SQA repo.

Reports what you can do RIGHT NOW, per reproduction tier, and for anything missing
prints the exact command that fixes it. Nothing here mutates the machine.

    uv run python scripts/check_env.py
"""

import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import config as cfg  # noqa: E402

OK, BAD, WARN = "\033[32mok\033[0m  ", "\033[31mMISS\033[0m", "\033[33mwarn\033[0m"

# tier -> modules it needs
RUNTIME = ["torch", "torchaudio", "transformers", "peft", "soundfile", "librosa", "omegaconf"]
EXPERIMENTS = ["numpy", "scipy", "pandas", "matplotlib", "pyarrow",
               "pesq", "pystoi", "speechmos", "onnxruntime"]
OPTIONAL = {
    "anthropic": "only for the corpus paraphrase step (uv sync --extra experiments)",
    "tensorboardX": "only for training (uv sync --extra experiments)",
    "mlflow": "only for the serving/batch tools (uv sync --extra api --extra tracking)",
    "pptx": "only for build_deck.py (uv sync --extra deck)",
    "fastapi": "only for the API server (uv sync --extra api)",
}


def mods(names) -> bool:
    all_ok = True
    for m in names:
        try:
            importlib.import_module(m)
            print(f"  {OK} {m}")
        except Exception as e:  # noqa: BLE001
            all_ok = False
            print(f"  {BAD} {m}  ({type(e).__name__})")
    return all_ok


def path(label: str, p, how: str) -> bool:
    p = Path(p)
    if p.exists():
        print(f"  {OK} {label}")
        return True
    print(f"  {BAD} {label}\n         not at {p}\n         -> {how}")
    return False


def main() -> None:
    print("\n=== Python deps: model runtime ===")
    runtime_ok = mods(RUNTIME)
    if not runtime_ok:
        print("     -> uv sync --extra experiments")

    print("\n=== Python deps: experiments ===")
    exp_ok = mods(EXPERIMENTS)
    if not exp_ok:
        print("     -> uv sync --extra experiments")

    print("\n=== Python deps: optional ===")
    for m, why in OPTIONAL.items():
        try:
            importlib.import_module(m)
            print(f"  {OK} {m}")
        except Exception:  # noqa: BLE001
            print(f"  {WARN} {m}  — {why}")

    try:
        import torch
        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "—"
        print(f"\n=== GPU ===\n  {OK if cuda else WARN} CUDA={cuda}  {name}")
        if not cuda:
            print("         inference/training need a GPU (~24 GB for training)")
    except Exception:  # noqa: BLE001
        pass

    # ---- Tier 1: analysis only (no GPU, no datasets) ----
    print("\n=== Tier 1 — re-run analysis + plots (no GPU, no datasets) ===")
    t1 = all([
        path("v3 training corpus", cfg.PLANB_DIR / "corpus_train_v3.jsonl",
             "ships with the repo — did the clone succeed?"),
        path("cached eval outputs", cfg.RESULTS_DIR / "voicebank_sqa.jsonl",
             "ships with the repo"),
        path("objective metrics", cfg.RESULTS_DIR / "objective_metrics.csv",
             "ships with the repo"),
    ]) and exp_ok

    # ---- Tier 2: evaluate the published v3 model ----
    print("\n=== Tier 2 — evaluate the v3 model ===")
    try:
        ck = cfg.ckpt_v3(stage=2)
        print(f"  {OK} v3 checkpoint ({ck.name})")
        have_ckpt = True
    except cfg.MissingInput:
        print(f"  {BAD} v3 checkpoint\n         -> uv run python scripts/fetch_checkpoints.py")
        have_ckpt = False
    base = all([
        path("SALMONN model code", cfg.SQA_ROOT / "salmonn_sqa" / "SALMONN" / "models",
             "bash scripts/setup_salmonn.sh"),
        path("Vicuna-7B-v1.5", cfg.SQA_ROOT / "salmonn_sqa" / "models" / "vicuna-7b-v1_5",
             "bash scripts/setup_salmonn.sh"),
        path("Whisper-large-v2", cfg.SQA_ROOT / "salmonn_sqa" / "models" / "whisper-large-v2",
             "bash scripts/setup_salmonn.sh"),
    ])
    beats = list((cfg.SQA_ROOT / "salmonn_sqa" / "models").glob("BEATs_*.pt"))
    if beats:
        print(f"  {OK} BEATs encoder")
    else:
        print(f"  {BAD} BEATs encoder\n         -> manual download, see scripts/setup_salmonn.sh")
    t2 = base and have_ckpt and bool(beats) and runtime_ok

    # ---- Tier 3: full retrain ----
    print("\n=== Tier 3 — regenerate the corpus / retrain ===")
    t3 = all([
        path("LibriTTS-R", cfg.LIBRITTS_ROOT,
             "https://www.openslr.org/141/  (set SQA_LIBRITTS_ROOT)"),
        path("MUSAN", cfg.MUSAN_ROOT,
             "https://www.openslr.org/17/  (set SQA_MUSAN_ROOT)"),
        path("measured RIRs", cfg.RIR_ROOT,
             "measured RIR set  (set SQA_RIR_ROOT)"),
        path("NISQA + weights", cfg.NISQA_WEIGHTS,
             "bash scripts/setup_salmonn.sh  (clones NISQA)"),
    ])
    try:
        cfg.voicebank_parquet()
        print(f"  {OK} VoiceBank-DEMAND-16k")
    except cfg.MissingInput:
        t3 = False
        print(f"  {BAD} VoiceBank-DEMAND-16k\n         -> huggingface-cli download "
              "JacobLinCool/VoiceBank-DEMAND-16k --repo-type dataset")
    if os.environ.get("ANTHROPIC_API_KEY"):
        print(f"  {OK} ANTHROPIC_API_KEY set")
    else:
        print(f"  {WARN} ANTHROPIC_API_KEY not set — needed only to regenerate descriptions.\n"
              f"         The paraphrase cache ships with the repo, so retraining works without it.")

    print("\n" + "=" * 58)
    for label, ready in [("Tier 1  analysis + plots ", t1),
                         ("Tier 2  evaluate v3      ", t2),
                         ("Tier 3  corpus + retrain ", t3)]:
        print(f"  {label}  {'READY' if ready else 'not ready'}")
    print("=" * 58)
    print("See REPRODUCING.md for the full recipe.\n")
    sys.exit(0 if t1 else 1)


if __name__ == "__main__":
    main()
