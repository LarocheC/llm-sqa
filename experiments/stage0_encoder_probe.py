"""
Plan B — Stage 0 de-risk: is reverb / bandwidth / clipping linearly decodable
from SALMONN's FROZEN front-end?

If a simple linear probe can recover RT60 / low-pass cutoff / clip-fraction from
the frozen Whisper and BEATs features (and their concat, what feeds the Q-Former),
then LoRA + Q-Former fine-tuning can plausibly surface it -> Plan B is viable as
scoped. If the info is absent from the frozen features (low R^2 everywhere),
the blindness is architectural and we'd need to unfreeze the encoders.

Tap points probed:
  whisper  : Whisper-Large-v2 encoder output  (frozen, 1280-d, time-mean)
  beats    : BEATs encoder output             (frozen,  768-d, time-mean)
  concat   : [whisper | beats]                (2048-d) — what feeds the Q-Former
  qformer  : Q-Former + projection output     (TRAINED bottleneck, 4096-d)

Targets: RT60 (reverb, via local measured RIRs), low-pass cutoff (bandwidth),
clip-fraction (clipping). Reported as held-out (group-split by utterance) R^2 and
Spearman after PCA+ridge.
"""

import glob
import io
import os
import re
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from scipy.signal import fftconvolve
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import config as cfg  # noqa: E402
import salmonn_core  # noqa: E402
from experiments.degradation_sweep import clip_frac, lowpass  # noqa: E402

TMP = str(cfg.work_dir("scratch") / "_probe_feat.wav")
RIR_GLOB = cfg.RIR_GLOB
rng = np.random.default_rng(0)


def sample_rirs(n_bins=8, per_bin=1):
    files = glob.glob(RIR_GLOB)
    by_bin = {}
    for f in files:
        rt = float(re.search(r"_RT_([0-9.]+)", f).group(1))
        by_bin.setdefault(round(rt, 1), []).append((rt, f))
    chosen = []
    for b in sorted(by_bin)[:n_bins] if False else sorted(by_bin):
        picks = by_bin[b]
        idx = rng.choice(len(picks), size=min(per_bin, len(picks)), replace=False)
        chosen += [picks[i] for i in idx]
    # spread across the RT60 range
    chosen = sorted(chosen)
    step = max(1, len(chosen) // 8)
    return chosen[::step][:8]


def reverberate(clean, rir):
    rir = rir.astype(np.float64)
    rir = rir / (np.abs(rir).max() + 1e-9)
    y = fftconvolve(clean, rir)[: len(clean)]
    return y * np.sqrt((np.mean(clean**2) + 1e-12) / (np.mean(y**2) + 1e-12))


def get_features(sqa, audio):
    sf.write(TMP, np.clip(audio, -1, 1).astype(np.float32), 16000)
    s = salmonn_core.prepare_audio_sample(TMP, sqa.wav_processor, device=sqa.device)
    m = sqa.model
    with salmonn_core.inference_context(sqa.device):
        wh = m.speech_encoder(s["spectrogram"], return_dict=True).last_hidden_state
        be = m.beats.extract_features(s["raw_wav"], padding_mask=s["padding_mask"], feature_only=True)[0]
        qf = m.encode_speech(s["spectrogram"], s["raw_wav"], s["padding_mask"])[0]
    whm = wh.float().mean(1).squeeze(0).cpu().numpy()
    bem = be.float().mean(1).squeeze(0).cpu().numpy()
    qfm = qf.float().mean(1).squeeze(0).cpu().numpy()
    return {"whisper": whm, "beats": bem, "concat": np.concatenate([whm, bem]), "qformer": qfm}


def split_by_utterance(groups):
    """One held-out (by-utterance) test mask, shared across tap points for a target."""
    groups = np.asarray(groups)
    ug = np.array(sorted(set(groups.tolist())))
    rng.shuffle(ug)
    te_g = set(ug[: max(1, int(round(0.25 * len(ug))))].tolist())
    return np.array([g in te_g for g in groups])


def probe(X, y, te, k=60, lam=1.0):
    """standardize -> PCA(k) -> ridge on the train split; return R^2, Spearman, and
    (y_test, pred) on the held-out utterances."""
    X, y = np.asarray(X), np.asarray(y)
    tr = ~te
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    kk = min(k, Xtr.shape[0] - 2, Xtr.shape[1])
    U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
    Vk = Vt[:kk].T
    Ztr, Zte = Xtr @ Vk, Xte @ Vk
    ymu = y[tr].mean()
    w = np.linalg.solve(Ztr.T @ Ztr + lam * np.eye(kk), Ztr.T @ (y[tr] - ymu))
    pred = Zte @ w + ymu
    r2 = 1 - ((y[te] - pred) ** 2).sum() / (((y[te] - y[te].mean()) ** 2).sum() + 1e-9)
    rho = spearmanr(y[te], pred)[0] if len(set(y[te].tolist())) > 1 else float("nan")
    return r2, rho, y[te], pred


def main():
    n_utt = int(os.environ.get("N_UTT", "20"))
    rows = pq.read_table(str(cfg.voicebank_parquet())).slice(0, n_utt * 3).to_pylist()
    cleans = []
    for r in rows:
        a, _ = sf.read(io.BytesIO(r["clean"]["bytes"]))
        cleans.append((a[:, 0] if a.ndim == 2 else a).astype(np.float64))
        if len(cleans) >= n_utt:
            break
    rirs = sample_rirs()
    print(f"{len(cleans)} clean utts; RIR RT60s: {[round(rt,2) for rt,_ in rirs]}")
    cutoffs = [7900, 6000, 4000, 3000, 2000, 1500]
    fracs = [1.0, 0.5, 0.3, 0.2, 0.1]

    sqa = salmonn_core.load_model(device_name="cuda:0")

    data = {t: {"X": {tp: [] for tp in ["whisper", "beats", "concat", "qformer"]}, "y": [], "g": []}
            for t in ["reverb", "bandwidth", "clip"]}

    def add(task, feats, label, g):
        for tp, v in feats.items():
            data[task]["X"][tp].append(v)
        data[task]["y"].append(label)
        data[task]["g"].append(g)

    for gi, clean in enumerate(cleans):
        add("reverb", get_features(sqa, clean), 0.0, gi)  # dry anchor
        for rt, rf in rirs:
            rir, _ = sf.read(rf)
            add("reverb", get_features(sqa, reverberate(clean, rir)), rt, gi)
        for c in cutoffs:
            add("bandwidth", get_features(sqa, lowpass(clean, c) if c < 7900 else clean), np.log2(c), gi)
        for fr in fracs:
            add("clip", get_features(sqa, clip_frac(clean, fr) if fr < 1.0 else clean), -np.log10(fr), gi)
        if (gi + 1) % 5 == 0:
            print(f"  features {gi+1}/{len(cleans)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taps = ["whisper", "beats", "concat", "qformer"]
    tasks = ["reverb", "bandwidth", "clip"]
    # one held-out split per target, shared across taps (comparable R^2 + scatter)
    splits = {t: split_by_utterance(data[t]["g"]) for t in tasks}
    R2 = {t: {} for t in tasks}
    PRED = {t: {} for t in tasks}
    print("\n=== Stage 0: linear decodability of degradation from FROZEN features ===")
    print(f"{'target':10s} | " + " | ".join(f"{tp:>13}" for tp in taps))
    for t in tasks:
        cells = []
        for tp in taps:
            r2, rho, yte, pred = probe(data[t]["X"][tp], data[t]["y"], splits[t])
            R2[t][tp] = r2
            PRED[t][tp] = (yte, pred)
            cells.append(f"{r2:+.2f}/{rho:+.2f}")
        print(f"{t:10s} | " + " | ".join(f"{c:>13}" for c in cells))

    outdir = str(cfg.PLANB_DIR)
    os.makedirs(outdir, exist_ok=True)

    # (a) R^2 bar chart: tap points grouped by target
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(tasks)); w = 0.2
    for i, tp in enumerate(taps):
        ax.bar(x + (i - 1.5) * w, [R2[t][tp] for t in tasks], w, label=tp)
    ax.axhline(0.5, ls="--", c="gray", lw=1, label="0.5 (decodable threshold)")
    ax.set_xticks(x); ax.set_xticklabels(["reverb (RT60)", "bandwidth (cutoff)", "clipping (frac)"])
    ax.set_ylabel("held-out R²"); ax.set_ylim(0, 1)
    ax.set_title("Stage 0 — degradation decodability from FROZEN SALMONN features")
    ax.legend(fontsize=8, ncol=5, loc="lower center")
    fig.tight_layout(); fig.savefig(f"{outdir}/stage0_r2_bars.png", dpi=120); plt.close(fig)

    # (b) predicted-vs-true scatter, 3 targets x 4 taps
    ylab = {"reverb": "RT60 (s)", "bandwidth": "log2(cutoff Hz)", "clip": "clip severity −log10(frac)"}
    fig, ax = plt.subplots(len(tasks), len(taps), figsize=(13, 9), sharex="row", sharey="row")
    for r, t in enumerate(tasks):
        for c, tp in enumerate(taps):
            yte, pred = PRED[t][tp]
            a = ax[r][c]
            a.scatter(yte, pred, s=14, alpha=0.5)
            lo, hi = min(yte.min(), pred.min()), max(yte.max(), pred.max())
            a.plot([lo, hi], [lo, hi], "r--", lw=1)
            a.set_title(f"{t} / {tp}  (R²={R2[t][tp]:+.2f})", fontsize=9)
            if c == 0: a.set_ylabel(f"pred\n{ylab[t]}", fontsize=8)
            if r == len(tasks) - 1: a.set_xlabel(f"true {ylab[t]}", fontsize=8)
    fig.suptitle("Stage 0 — predicted vs true degradation parameter (held-out utterances)")
    fig.tight_layout(); fig.savefig(f"{outdir}/stage0_scatter.png", dpi=120); plt.close(fig)
    print(f"\nwrote {outdir}/stage0_r2_bars.png and stage0_scatter.png")
    print("Read: points on the y=x line = the degradation is linearly recoverable from the")
    print("FROZEN features (incl. the Q-Former output that reaches the LLM) -> fix is data, not encoders.")


if __name__ == "__main__":
    main()
