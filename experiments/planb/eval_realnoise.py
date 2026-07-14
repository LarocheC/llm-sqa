"""
Does the noise-naming gap survive on REAL noise? The main eval used synthetic white
noise; training used real MUSAN. This builds a noise-only sweep with **held-out
MUSAN noise** (files never used in training) on VoiceBank-DEMAND clean clips, and
runs orig / v1 / v2, reporting naming rate + rho(MOS,sev) + rho(noise-dim,sev).

If v2 names real noise far better than the 50% it scored on synthetic white noise,
the synthetic sweep was unfair and the model is fine on the distribution it was
trained for.

Usage:  python -m experiments.planb.eval_realnoise --n-clips 8
"""

import argparse
import glob
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments import config as cfg  # noqa: E402
import salmonn_core as sc  # noqa: E402
from experiments.degradation_sweep import pick_ids  # noqa: E402
from experiments.planb import degradations as D  # noqa: E402
from experiments.planb.eval_compare import (  # noqa: E402
    KEYWORDS, free, load_with_ckpt, parse_dim_scores,
)

SR = 16000


def prose(text, tag):
    """Natural-language description only — strip the structured score block + MOS
    line for every fine-tuned model (tags v1/v2/v3, not just 'planb'). The original
    model ('orig') emits prose already. (eval_compare.description_only only strips
    for tag=='planb', which silently inflated naming when tags were v1/v2/v3.)"""
    if tag == "orig":
        return text
    import re as _re
    return " ".join(l for l in text.split("\n")
                    if not _re.search(r"(noise|reverberation|bandwidth|clipping|discontinuity|loudness)\s*:\s*[1-5]", l, _re.I)
                    and not l.lower().strip().startswith("overall mos"))


TMP = str(cfg.work_dir("planb_realnoise"))
RESULTS = str(cfg.PLANB_DIR)
SNRS = [20, 15, 10, 5, 0]  # severity 1..5, matches the synthetic noise sweep
# Held-out = every MUSAN noise file the model under test never trained on. Read this
# from the corpus the model was actually trained on (v3); it used to point at the v1
# corpus, which is superseded and no longer shipped.
TRAIN_CORPUS = os.environ.get("SQA_TRAIN_CORPUS", f"{RESULTS}/corpus_train_v3.jsonl")


def heldout_noise(rng):
    cfg.require(TRAIN_CORPUS, "the v3 training corpus (to determine held-out MUSAN noise)",
                "it ships with the repo; or regenerate with experiments/planb/generate_corpus.py")
    used = set()
    for l in open(TRAIN_CORPUS):
        nf = (json.loads(l)["params"].get("noise") or {}).get("noise_file")
        if nf:
            used.add(nf)
    files = [p for p in glob.glob(f"{cfg.MUSAN_ROOT}/noise/**/*.wav", recursive=True)
             if os.path.basename(p) not in used]
    rng.shuffle(files)
    return files


def build_clips(n_clips, rng):
    rows = pq.read_table(str(cfg.voicebank_parquet())).to_pylist()
    by_id = {r["id"]: r for r in rows}
    ids = pick_ids(rows, 50)[:n_clips]
    noise = heldout_noise(rng)
    clips, ni = [], 0
    for rid in ids:
        clean, _ = sf.read(io.BytesIO(by_id[rid]["clean"]["bytes"]))
        clean = (clean[:, 0] if clean.ndim == 2 else clean).astype(np.float64)
        clips.append((rid, 0, "clean", clean))
        for sev, snr in enumerate(SNRS, start=1):
            noisy, info = D.add_real_noise(clean, noise[ni % len(noise)], snr, rng); ni += 1
            clips.append((rid, sev, f"snr{snr}", noisy))
    return ids, clips


def run(tag, ckpt, prompt, clips, device, fh):
    print(f"\n=== {tag} ({os.path.basename(ckpt) if ckpt else 'BASE'}) ===", flush=True)
    sqa = load_with_ckpt(ckpt, device)
    recs = []
    for i, (rid, sev, label, audio) in enumerate(clips):
        path = f"{TMP}/{tag}_{rid}_{label}.wav"
        sf.write(path, np.clip(audio, -1, 1).astype(np.float32), SR)
        raw = sc.generate_sqa(sqa, prompt=prompt, wav_path=path)
        text = sc.clean_output(raw)
        rec = {"tag": tag, "id": rid, "severity": sev, "level": label,
               "mos": sc.extract_mos(raw), "dims": parse_dim_scores(text), "desc": text}
        recs.append(rec); fh.write(json.dumps(rec) + "\n"); fh.flush()
        if i < 2:
            print(f"  [{tag} {label}] mos={rec['mos']} noise_dim={rec['dims'].get('noise')}"
                  f"\n    {prose(text, tag)[:140]}", flush=True)
        if (i + 1) % 24 == 0:
            print(f"  {tag}: {i+1}/{len(clips)}", flush=True)
    free(sqa)
    return recs


def stats(recs, tag):
    sev = np.array([r["severity"] for r in recs])
    mos = np.array([r["mos"] if r["mos"] is not None else np.nan for r in recs])
    ok = ~np.isnan(mos)
    rho_mos = spearmanr(sev[ok], mos[ok])[0]
    dv = np.array([r["dims"].get("noise", np.nan) for r in recs], float)
    dok = ~np.isnan(dv)
    rho_dim = spearmanr(sev[dok], dv[dok])[0] if dok.sum() > 3 else np.nan
    deg = [r for r in recs if r["severity"] > 0]
    naming = np.mean([any(k in prose(r["desc"], tag).lower() for k in KEYWORDS["noise"]) for r in deg])
    return rho_mos, rho_dim, naming


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    rng = np.random.default_rng(123)
    ids, clips = build_clips(args.n_clips, rng)
    print(f"real-noise sweep: {len(ids)} clean clips x {len(SNRS)} SNRs (held-out MUSAN noise)")

    full = json.load(open("experiments/planb/train/test_prompt_planb.json"))["sqa_full"]
    models = [
        ("v3", str(cfg.ckpt_v3(stage=2)), full),
    ]
    recs = []
    with open(f"{RESULTS}/eval_realnoise_v3.jsonl", "w") as fh:
        for tag, ckpt, prompt in models:
            recs += run(tag, ckpt, prompt, clips, args.device, fh)

    print("\n# Real held-out MUSAN noise — naming & sensitivity\n")
    print(f"{'model':5} | rho(MOS,sev) | rho(noise-dim,sev) | noise naming (prose)")
    for tag, _, _ in models:
        rm, rd, nm = stats([r for r in recs if r["tag"] == tag], tag)
        print(f"{tag:5} | {rm:+.2f}        | {('' if np.isnan(rd) else f'{rd:+.2f}'):>6}             | {nm:.0%}")
    print("\n(compare to synthetic white noise: orig 97%, v1 23%, v2 50%)")


if __name__ == "__main__":
    main()
