"""
Plan B — Stage 1 synthetic corpus + target generator.

Takes a directory of leakage-safe clean speech (LibriTTS-R), applies graded,
parameter-known degradation recipes biased toward SALMONN-SQA's blind spots
(reverb / bandwidth / clipping / discontinuity), and emits, per clip:
  - the degraded wav,
  - exact per-axis parameters,
  - per-dimension 1-5 scores derived from those params,
  - a fused, de-compressed Overall MOS (PESQ + NISQA + DNSMOS),
  - a grounded 1-3 sentence description,
  - the assembled `target_text` the model is trained to emit.

NISQA is run in one batch over the whole wav dir at the end and merged back, then
the fused MOS is (re)computed. Run with --no-metrics for a fast schema dry-run
(MOS falls back to a param proxy).

Usage:
  python -m experiments.planb.generate_corpus --clean-dir <dir> --n 300 \
      --out experiments/results/planb/corpus.jsonl --wav-dir /tmp/planb_wav
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pathlib import Path  # noqa: E402

from experiments import config as cfg  # noqa: E402
from experiments.planb import degradations as D  # noqa: E402
from experiments.planb import targets as T  # noqa: E402

SR = 16000


# ----------------------------------------------------------------------------- clean pool
def _rel_clean(p):
    """Provenance only (nothing reads it back). Store it RELATIVE to the LibriTTS root so
    the corpus is portable — it used to embed the generating machine's absolute path."""
    try:
        return str(Path(p).resolve().relative_to(Path(cfg.LIBRITTS_ROOT).resolve()))
    except ValueError:
        return os.path.basename(p)


def load_clean_paths(clean_dir, rng, limit_pool=4000):
    paths = glob.glob(os.path.join(clean_dir, "**", "*.wav"), recursive=True)
    paths.sort()
    rng.shuffle(paths)
    return paths[:limit_pool]


def read_clean(path, max_s=6.0, min_s=2.0):
    a, sr = sf.read(path)
    if a.ndim == 2:
        a = a[:, 0]
    a = a.astype(np.float64)
    if sr != SR:
        a = resample_poly(a, SR, sr)
    if len(a) < min_s * SR:
        return None
    if len(a) > max_s * SR:
        a = a[: int(max_s * SR)]
    peak = np.max(np.abs(a)) + 1e-9
    return a / peak * 0.95  # normalize headroom so degradations are comparable


# ----------------------------------------------------------------------------- recipe sampling
# severity grids per axis (each entry produces a target band when applied alone)
NOISE_SNR = [25, 18, 10, 4]
BW_CUTOFF = [6500, 5000, 3400, 2200, 1200]
CLIP_KNOB = [0.6, 0.4, 0.25, 0.12]
LOSS_RATE = [0.02, 0.05, 0.10, 0.20]
GAIN_DB = [-6, -9, -14, -22, 5]  # -9 fills the loudness:3 band (|6-12| dB)
# lossy codecs, mild -> harsh (fmt, encoder, bitrate). Fold into bandwidth/coloration.
CODEC_OPTS = [("mp3", None, 32000), ("ogg", "libopus", 10000),
              ("mp3", None, 16000), ("ogg", "libopus", 6000)]
# blind-spot axes get more primary slots than the already-strong noise axis
PRIMARY_WEIGHTS = {"reverb": 4, "bandwidth": 4, "clip": 4, "discontinuity": 3,
                   "codec": 3, "noise": 2, "loudness": 2}
SECONDARY_AXES = ["reverb", "bandwidth", "clip", "noise", "discontinuity", "loudness", "codec"]


def _grid(axis, rir_bank):
    return {"reverb": rir_bank, "bandwidth": BW_CUTOFF, "clip": CLIP_KNOB,
            "discontinuity": LOSS_RATE, "noise": NOISE_SNR, "loudness": GAIN_DB,
            "codec": CODEC_OPTS}[axis]


def build_recipes(n, rir_bank, rng):
    """Round-robin over (primary axis, severity) cells for balance, then add 0-2
    mild secondary axes. ~10% clean clips. Returns list of recipe dicts."""
    cells = []
    for axis, w in PRIMARY_WEIGHTS.items():
        for sev in _grid(axis, rir_bank):
            cells += [(axis, sev)] * w
    rng.shuffle(cells)

    recipes = []
    for i in range(n):
        rec = {}
        if rng.random() < 0.10:  # clean anchor
            recipes.append(rec)
            continue
        axis, sev = cells[i % len(cells)]
        rec[axis] = sev
        n_sec = rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
        for _ in range(n_sec):
            cand = [a for a in SECONDARY_AXES if a not in rec] or SECONDARY_AXES
            sa = rng.choice(cand)
            grid = _grid(sa, rir_bank)
            # secondaries skew mild (first half of each grid, which is ordered mild-first)
            hi = max(1, len(grid) // 2) if sa == "reverb" else max(1, len(grid) // 2 + 1)
            rec[sa] = grid[rng.integers(0, hi)]
        recipes.append(rec)
    return recipes


# ----------------------------------------------------------------------------- render
def render(clean, recipe, rng, noise_bank=None):
    """Apply the recipe in acoustic order; return (degraded, params)."""
    x = clean.copy()
    params = {}
    if "reverb" in recipe:
        rt60, path = recipe["reverb"]
        if rng.random() < 0.30:  # synthetic exp reverb alongside the real RIRs
            x, info = D.synth_reverb(x, rt60, rng)
        else:
            rir, _ = sf.read(path)
            x, info = D.reverberate(x, rir)
        info["rt60"] = float(rt60)
        params["reverb"] = info
    if "noise" in recipe:
        snr = recipe["noise"]
        # span both distributions: ~half real MUSAN, ~half synthetic white/pink/brown
        if noise_bank and rng.random() < 0.5:
            x, params["noise"] = D.add_real_noise(x, noise_bank[rng.integers(len(noise_bank))], snr, rng)
        else:
            color = rng.choice(["white", "pink", "brown"])
            x, params["noise"] = D.add_colored_noise(x, snr, rng, color)
    if "clip" in recipe:
        x, params["clip"] = D.clip_frac(x, recipe["clip"])
    if "bandwidth" in recipe:
        x, params["bandwidth"] = D.lowpass(x, recipe["bandwidth"])
    if "codec" in recipe:
        fmt, enc, br = recipe["codec"]
        x, params["codec"] = D.apply_codec(x, fmt, enc, br)
    if "loudness" in recipe:
        x, params["loudness"] = D.regain(x, recipe["loudness"])
    if "discontinuity" in recipe:
        x, params["discontinuity"] = D.packet_loss(x, recipe["discontinuity"], rng)
    return x, params


# ----------------------------------------------------------------------------- metrics
def compute_metrics(clean, deg):
    from pesq import pesq
    from speechmos import dnsmos
    m = {}
    n = min(len(clean), len(deg))
    cr, dr = clean[:n].astype(np.float32), np.clip(deg[:n], -1, 1).astype(np.float32)
    try:
        m["pesq"] = float(pesq(SR, cr, dr, "wb"))
    except Exception:
        m["pesq"] = None
    try:
        m["dnsmos_ovrl"] = float(dnsmos.run(dr, sr=SR)["ovrl_mos"])
    except Exception:
        m["dnsmos_ovrl"] = None
    return m


def run_nisqa(wav_dir, results_dir):
    outdir = os.path.join(results_dir, "nisqa_planb")
    os.makedirs(outdir, exist_ok=True)
    subprocess.run(
        [sys.executable, "experiments/NISQA/run_predict.py", "--mode", "predict_dir",
         "--pretrained_model", "experiments/NISQA/weights/nisqa.tar", "--data_dir", wav_dir,
         "--num_workers", "0", "--bs", "20", "--output_dir", outdir],
        check=True, capture_output=True, text=True)
    import pandas as pd
    csv = sorted(glob.glob(f"{outdir}/*.csv"), key=os.path.getmtime)[-1]
    return pd.read_csv(csv).set_index("deg")["mos_pred"].to_dict()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="experiments/results/planb/corpus.jsonl")
    ap.add_argument("--wav-dir", default=str(cfg.CORPUS_WAV_DIR / "wav"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--musan-root", default=D.MUSAN_ROOT, help="MUSAN root for real noise (falls back to white)")
    ap.add_argument("--limit-pool", type=int, default=20000, help="max clean wavs to draw from")
    ap.add_argument("--no-metrics", action="store_true", help="skip PESQ/DNSMOS/NISQA (schema dry-run)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.wav_dir, exist_ok=True)
    results_dir = os.path.dirname(args.out)
    rng = np.random.default_rng(args.seed)

    rir_bank = D.load_rir_bank(rng, cache=cfg.RIR_CACHE)
    print(f"RIR bank: {len(rir_bank)} RIRs, RT60 {rir_bank[0][0]}–{rir_bank[-1][0]}")
    noise_bank = D.load_noise_bank(args.musan_root)
    print(f"noise bank: {len(noise_bank)} MUSAN files"
          + ("" if noise_bank else " -> NONE, falling back to white noise"))
    clean_paths = load_clean_paths(args.clean_dir, rng, args.limit_pool)
    print(f"clean pool: {len(clean_paths)} wavs from {args.clean_dir}")
    if not clean_paths:
        sys.exit("no clean wavs found")

    recipes = build_recipes(args.n, rir_bank, rng)
    records, ci = [], 0
    for i, recipe in enumerate(recipes):
        clean = None
        while clean is None and ci < len(clean_paths):
            clean = read_clean(clean_paths[ci]); ci += 1
        if clean is None:
            print("ran out of usable clean clips"); break
        deg, params = render(clean, recipe, rng, noise_bank)
        scores = T.scores_from_params(params)
        rid = f"planb_{i:05d}"
        wav_name = f"{rid}.wav"
        sf.write(os.path.join(args.wav_dir, wav_name),
                 np.clip(deg, -1, 1).astype(np.float32), SR)

        metrics = {} if args.no_metrics else compute_metrics(clean, deg)
        mos, mos_src = T.overall_mos(scores, metrics)

        records.append({
            "id": rid, "wav": wav_name, "clean_path": _rel_clean(clean_paths[ci - 1]),
            "params": params, "scores": scores, "metrics": metrics,
            "mos": mos, "mos_source": mos_src,
        })
        if (i + 1) % 50 == 0:
            print(f"  rendered {i+1}/{len(recipes)}")

    # batch NISQA, re-fuse MOS
    if not args.no_metrics:
        print("running NISQA over the corpus ...")
        try:
            nis = run_nisqa(args.wav_dir, results_dir)
            for r in records:
                nm = nis.get(r["wav"])
                if nm is not None:
                    r["metrics"]["nisqa"] = float(nm)
                    r["mos"], r["mos_source"] = T.overall_mos(r["scores"], r["metrics"])
        except Exception as e:
            print(f"NISQA failed ({e}); keeping PESQ+DNSMOS fusion")

    # assemble target_text last (after final MOS)
    for r in records:
        r["prompt"] = T.PLANB_PROMPT
        ctx = {"noise_type": (r["params"].get("noise") or {}).get("noise_type")}
        desc = T.describe(r["scores"], np.random.default_rng(hash(r["id"]) % 2**32), ctx)
        r["target_text"] = T.build_target(r["scores"], r["mos"], desc)

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} records -> {args.out}")

    # summary
    print("\n=== per-dimension score distribution ===")
    for d in T.DIMS:
        c = np.bincount([r["scores"][d] for r in records], minlength=6)[1:]
        print(f"  {d:14s} 1..5: {c.tolist()}")
    mos_all = np.array([r["mos"] for r in records])
    print(f"\nMOS: mean {mos_all.mean():.2f} ± {mos_all.std():.2f}, "
          f"range {mos_all.min():.2f}–{mos_all.max():.2f}, "
          f"distinct {len(set(np.round(mos_all,2)))}")
    from collections import Counter
    print("mos_source:", dict(Counter(r["mos_source"] for r in records)))


if __name__ == "__main__":
    main()
