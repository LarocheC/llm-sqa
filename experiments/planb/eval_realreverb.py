"""
Real-reverb control: does the model's MOS track perceived quality on REAL room
impulse responses, rather than the synthetic exponential-decay reverb the held-out
sweep uses?

This matters because `synth_reverb` injects an explicit direct path, so synthetic
reverb has an artificially high DRR at every RT60. A model calibrated on real rooms
(where DRR varies with mic distance) will under-penalize it. The degradation sweep
therefore cannot tell "worse at reverb" apart from "calibrated to real rooms".

Method: convolve held-out clean speech with REAL RIRs that were *not* in the training
bank, then correlate each model's MOS against PESQ — an independent reference that
knows nothing about either model's severity map.

  uv run python -m experiments.planb.eval_realreverb --ckpt-a <v3> --ckpt-b <open>
"""

import argparse
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from pesq import pesq as PESQ
from scipy.signal import fftconvolve
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments import config as cfg  # noqa: E402
from experiments.planb import degradations as D  # noqa: E402


def held_out_rirs(n, train_seed=1):
    """Real RIRs the OPEN model never trained on.

    The training bank is drawn by load_rir_bank() from an rng seeded with the corpus
    seed, so replaying that seed reproduces it exactly; everything else is held out.
    (v3 trained on a different RIR corpus entirely, so all of these are unseen by it too.)
    """
    train_bank = {f for _, f in D.load_rir_bank(np.random.default_rng(train_seed), cache=cfg.RIR_CACHE)}
    cache = json.load(open(cfg.RIR_CACHE))
    pool = [(v[0], v[1], f) for f, v in cache.items()
            if isinstance(v, list) and v[0] is not None and f not in train_bank]
    # spread across the reverb severity range so the correlation has something to grip
    from experiments.planb.targets import score_reverb
    by_s = {}
    for rt, dr, f in pool:
        by_s.setdefault(int(score_reverb(rt, dr)), []).append((rt, dr, f))
    rng = np.random.default_rng(123)
    out = []
    per = max(1, n // max(1, len(by_s)))
    for s in sorted(by_s):
        picks = by_s[s]
        idx = rng.choice(len(picks), size=min(per, len(picks)), replace=False)
        out += [picks[i] for i in idx]
    return out, len(train_bank)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", required=True, help="model A (e.g. v3)")
    ap.add_argument("--ckpt-b", required=True, help="model B (e.g. open)")
    ap.add_argument("--name-a", default="v3")
    ap.add_argument("--name-b", default="open")
    ap.add_argument("--n-clips", type=int, default=6)
    ap.add_argument("--n-rirs", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import salmonn_core as sc
    from experiments.planb.eval_compare import load_with_ckpt, pick_ids

    rirs, n_train = held_out_rirs(args.n_rirs)
    print(f"held-out real RIRs: {len(rirs)}  (excluded {n_train} used in training)")
    for rt, dr, f in rirs:
        print(f"   RT60 {rt:.2f}  DRR {dr:+6.1f} dB   {os.path.basename(f)}")

    rows = pq.read_table(str(cfg.voicebank_parquet())).to_pylist()
    by_id = {r["id"]: r for r in rows}
    ids = pick_ids(rows, 50)[: args.n_clips]

    # build the reverberant clips + their PESQ (independent reference)
    clips = []
    for rid in ids:
        clean, _ = sf.read(io.BytesIO(by_id[rid]["clean"]["bytes"]))
        if clean.ndim > 1:
            clean = clean[:, 0]
        clean = clean.astype(np.float64)
        for rt, dr, f in rirs:
            h, sr = sf.read(f)
            if np.ndim(h) > 1:
                h = h[:, 0]
            h = h.astype(np.float64)
            h /= (np.abs(h).max() + 1e-9)
            y = fftconvolve(clean, h)[: len(clean)]
            y *= np.sqrt((np.mean(clean**2) + 1e-12) / (np.mean(y**2) + 1e-12))
            try:
                p = PESQ(16000, clean.astype(np.float32), y.astype(np.float32), "wb")
            except Exception:
                p = None
            if p is not None:
                clips.append((rid, rt, dr, p, y))
    print(f"\n{len(clips)} reverberant clips (PESQ {min(c[3] for c in clips):.2f}-{max(c[3] for c in clips):.2f})")

    tmp = cfg.work_dir("realreverb")
    prompt = json.load(open("experiments/planb/train/test_prompt_planb.json"))["sqa_full"]
    results = {}
    for name, ckpt in [(args.name_a, args.ckpt_a), (args.name_b, args.ckpt_b)]:
        print(f"\n=== {name} ===")
        sqa = load_with_ckpt(ckpt, args.device)
        mos, rev = [], []
        for i, (rid, rt, dr, p, y) in enumerate(clips):
            wav = str(tmp / f"{rid}_{i}.wav")
            sf.write(wav, np.clip(y, -1, 1).astype(np.float32), 16000)
            raw = sc.generate_sqa(sqa, prompt=prompt, wav_path=wav)
            m = sc.extract_mos(raw)
            text = sc.clean_output(raw)
            import re
            mm = re.search(r"reverberation\s*:\s*([1-5])", text, re.I)
            mos.append(m)
            rev.append(int(mm.group(1)) if mm else None)
            if (i + 1) % 16 == 0:
                print(f"   {i+1}/{len(clips)}", flush=True)
        results[name] = (mos, rev)
        del sqa
        import torch
        torch.cuda.empty_cache()

    P = [c[3] for c in clips]
    RT = [c[1] for c in clips]
    print("\n" + "=" * 62)
    print("REAL-REVERB CONTROL — held-out real RIRs, PESQ as independent reference")
    print("=" * 62)
    print(f"{'model':8} {'rho(MOS, PESQ)':>16} {'rho(revScore, PESQ)':>21}")
    for name in (args.name_a, args.name_b):
        mos, rev = results[name]
        ok = [i for i, m in enumerate(mos) if m is not None]
        r1 = spearmanr([mos[i] for i in ok], [P[i] for i in ok])[0]
        okr = [i for i, v in enumerate(rev) if v is not None]
        r2 = spearmanr([rev[i] for i in okr], [P[i] for i in okr])[0] if len(okr) > 5 else float("nan")
        print(f"{name:8} {r1:+16.3f} {r2:+21.3f}")
    print("\n(higher = the model's judgement agrees better with an independent quality metric)")
    print(f"{'model':8} {'rho(MOS, RT60)':>16}")
    for name in (args.name_a, args.name_b):
        mos, _ = results[name]
        ok = [i for i, m in enumerate(mos) if m is not None]
        print(f"{name:8} {spearmanr([mos[i] for i in ok], [RT[i] for i in ok])[0]:+16.3f}")


if __name__ == "__main__":
    main()
