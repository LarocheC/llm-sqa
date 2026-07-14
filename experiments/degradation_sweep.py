"""
Controlled degradation sweep — probe SALMONN's sensitivity, monotonicity and
degradation *vocabulary* by applying ONE degradation at a time at controlled
levels to a fixed set of clean utterances.

Unlike the VoiceBank test set (which confounds SNR with speaker/noise/utterance),
here every variable but one is held fixed, so we can ask:
  * does MOS fall monotonically as each degradation worsens, and how sensitively?
  * does the description correctly NAME each degradation type (not just "noise")?
  * how does SALMONN's response compare to DNSMOS / NISQA / PESQ on the same clips?

Four synthesizable families (no external codecs needed):
  noise   — additive white noise at fixed SNRs
  lowpass — bandwidth limiting (telephone-like) at fixed cutoffs
  clip    — hard clipping at fixed peak fractions
  reverb  — synthetic exponential-decay reverberation at fixed RT60s
"""

import argparse
import glob
import io
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from pesq import pesq
from scipy.signal import butter, sosfiltfilt
from speechmos import dnsmos

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import config as cfg  # noqa: E402
import salmonn_core  # noqa: E402

SR = 16000
RESULTS = str(cfg.RESULTS_DIR)
DEG_DIR = str(cfg.work_dir("vb_degraded"))
rng = np.random.default_rng(0)

# severity rank grows with degradation; level value is the family-specific knob.
FAMILIES = {
    "noise": [("snr20", 20), ("snr15", 15), ("snr10", 10), ("snr5", 5), ("snr0", 0)],
    "lowpass": [("lp6000", 6000), ("lp4000", 4000), ("lp3000", 3000), ("lp2000", 2000)],
    "clip": [("clip0.5", 0.5), ("clip0.3", 0.3), ("clip0.2", 0.2), ("clip0.1", 0.1)],
    "reverb": [("rt0.3", 0.3), ("rt0.6", 0.6), ("rt0.9", 0.9), ("rt1.2", 1.2)],
}


def add_white_noise(x, snr_db):
    sigp = np.mean(x**2)
    if sigp < 1e-12:
        return x.copy()
    noise = rng.standard_normal(len(x))
    noise /= np.sqrt(np.mean(noise**2)) + 1e-12
    noise *= np.sqrt(sigp / (10 ** (snr_db / 10)))
    return x + noise


def lowpass(x, cutoff):
    sos = butter(8, cutoff / (SR / 2), btype="low", output="sos")
    return sosfiltfilt(sos, x)


def clip_frac(x, frac):
    peak = np.max(np.abs(x)) + 1e-9
    thr = frac * peak
    return np.clip(x, -thr, thr) * (peak / thr)  # flat-top clip, peak restored


def reverb(x, rt60):
    L = int(rt60 * SR)
    t = np.arange(L)
    rir = rng.standard_normal(L) * np.exp(-6.9 * t / (rt60 * SR))
    rir[0] += 1.0
    y = np.convolve(x, rir)[: len(x)]
    return y * np.sqrt((np.mean(x**2) + 1e-12) / (np.mean(y**2) + 1e-12))


def apply(family, val, x):
    if family == "noise":
        return add_white_noise(x, val)
    if family == "lowpass":
        return lowpass(x, val)
    if family == "clip":
        return clip_frac(x, val)
    if family == "reverb":
        return reverb(x, val)
    raise ValueError(family)


def pick_ids(rows, per_speaker):
    by_spk = {}
    for r in rows:
        by_spk.setdefault(r["id"].split("_")[0], []).append(r["id"])
    chosen = []
    for spk, ids in sorted(by_spk.items()):
        chosen += sorted(ids)[:per_speaker]
    return chosen


def run(args):
    os.makedirs(DEG_DIR, exist_ok=True)
    rows = pq.read_table(str(cfg.voicebank_parquet())).to_pylist()
    by_id = {r["id"]: r for r in rows}
    ids = pick_ids(rows, args.per_speaker)
    print(f"{len(ids)} utterances: {ids}")

    sqa = salmonn_core.load_model(device_name=args.device)
    out = open(args.out, "w")
    n = 0

    def infer(rid, family, label, sev, val, audio, clean_ref):
        nonlocal n
        path = f"{DEG_DIR}/{rid}__{family}__{label}.wav"
        clipped = np.clip(audio, -1.0, 1.0).astype(np.float32)
        sf.write(path, clipped, SR)
        samples = salmonn_core.prepare_audio_sample(path, sqa.wav_processor, device=sqa.device)
        raw = salmonn_core.generate_sqa(sqa, samples=samples)
        desc = salmonn_core.clean_output(raw)
        try:
            m = min(len(clean_ref), len(clipped))
            pq_score = pesq(SR, clean_ref[:m].astype(np.float32), clipped[:m], "wb")
        except Exception:
            pq_score = None
        try:
            d = dnsmos.run(clipped, sr=SR)
        except Exception:
            d = {}
        def f(v):
            return float(v) if v is not None else None

        rec = {
            "id": rid, "speaker": rid.split("_")[0], "family": family,
            "level": label, "level_val": val, "severity": sev,
            "salmonn_mos": salmonn_core.extract_mos(raw), "description": desc,
            "n_words": len(desc.split()), "degenerate": salmonn_core.is_degenerate(raw),
            "pesq": f(pq_score), "dnsmos_ovrl": f(d.get("ovrl_mos")), "dnsmos_p808": f(d.get("p808_mos")),
            "wav": os.path.basename(path),
        }
        out.write(json.dumps(rec) + "\n")
        out.flush()
        n += 1

    for rid in ids:
        clean, _ = sf.read(io.BytesIO(by_id[rid]["clean"]["bytes"]))
        if clean.ndim == 2:
            clean = clean[:, 0]
        clean = clean.astype(np.float64)
        clean_ref = np.clip(clean, -1, 1).astype(np.float32)
        infer(rid, "clean", "clean", 0, None, clean, clean_ref)  # baseline
        for family, levels in FAMILIES.items():
            for sev, (label, val) in enumerate(levels, start=1):
                infer(rid, family, label, sev, val, apply(family, val, clean), clean_ref)
        print(f"  {rid} done ({n} clips)")

    out.close()
    print(f"Generated {n} clips -> {args.out}")

    # NISQA over all degraded clips at once
    print("Running NISQA ...")
    outdir = f"{RESULTS}/nisqa_deg"
    os.makedirs(outdir, exist_ok=True)
    subprocess.run(
        [sys.executable, "experiments/NISQA/run_predict.py", "--mode", "predict_dir",
         "--pretrained_model", "experiments/NISQA/weights/nisqa.tar", "--data_dir", DEG_DIR,
         "--num_workers", "0", "--bs", "20", "--output_dir", outdir],
        check=True, capture_output=True, text=True)
    import pandas as pd
    csv = sorted(glob.glob(f"{outdir}/*.csv"), key=os.path.getmtime)[-1]
    nis = pd.read_csv(csv).set_index("deg")["mos_pred"].to_dict()
    # merge nisqa_mos back into the jsonl
    recs = [json.loads(l) for l in open(args.out)]
    for r in recs:
        r["nisqa_mos"] = nis.get(r["wav"])
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"Merged NISQA into {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{RESULTS}/degradation.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-speaker", type=int, default=8, help="utterances per speaker")
    run(ap.parse_args())
