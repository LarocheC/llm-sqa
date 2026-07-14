"""
Plan B — Stage 1 degradation chain (graded severity, KNOWN parameters).

Each degradation returns (audio, info) where `info` carries the exact parameter
that produced it — this is the noise-free supervision Plan B trains on. The point
of Stage 1 is to bias the curriculum toward SALMONN-SQA's blind spots (reverb,
bandwidth, clipping, discontinuity) which it currently can't name, while keeping
additive noise modest (already a strong axis).

Conventions: audio is float64 mono at 16 kHz, nominal range [-1, 1]. Real reverb
uses the local measured RIRs (RT60 in the filename); everything else is DSP so the
parameter is exact. Reverb severity additionally accounts for DRR (direct-to-
reverberant ratio), which tracks *perceived* reverberation better than RT60 alone.
"""

import glob
import os
import re

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfiltfilt

from experiments import config as cfg

SR = cfg.SR
RIR_GLOB = cfg.RIR_GLOB
MUSAN_ROOT = str(cfg.MUSAN_ROOT)


# ----------------------------------------------------------------------------- noise
def _scale_noise_to_snr(x, noise, snr_db):
    sigp = np.mean(x**2)
    if sigp < 1e-12:
        return x.copy()
    noise = noise / (np.sqrt(np.mean(noise**2)) + 1e-12)
    noise = noise * np.sqrt(sigp / (10 ** (snr_db / 10)))
    return x + noise


def add_white_noise(x, snr_db, rng):
    noise = rng.standard_normal(len(x))
    return _scale_noise_to_snr(x, noise, snr_db), {"snr_db": float(snr_db), "noise_type": "white"}


def add_colored_noise(x, snr_db, rng, color="white"):
    """Synthetic noise shaped to white / pink (1/sqrt f) / brown (1/f) spectra, at
    the target SNR. Covers the synthetic-noise distribution the eval sweep uses,
    complementing the real MUSAN noise so training spans both."""
    n = len(x)
    w = rng.standard_normal(n)
    if color == "white":
        noise = w
    else:
        f = np.fft.rfftfreq(n, 1.0 / SR)
        f[0] = f[1] if len(f) > 1 else 1.0
        env = 1.0 / np.sqrt(f) if color == "pink" else 1.0 / f  # pink vs brown
        noise = np.fft.irfft(np.fft.rfft(w) * env, n=n)
    return _scale_noise_to_snr(x, noise, snr_db), {"snr_db": float(snr_db), "noise_type": color}


def load_noise_bank(root=MUSAN_ROOT):
    """MUSAN noise + music wav paths (real ambient/babble/music backgrounds)."""
    bank = []
    for sub in ("noise", "music"):
        bank += glob.glob(os.path.join(root, sub, "**", "*.wav"), recursive=True)
    return sorted(bank)


def add_real_noise(x, noise_path, snr_db, rng):
    """Add a random segment of a real MUSAN noise/music file at the target SNR."""
    nz, nsr = sf.read(noise_path)
    if nz.ndim == 2:
        nz = nz[:, 0]
    nz = nz.astype(np.float64)
    if nsr != SR:  # MUSAN is 16 kHz, but guard anyway
        from scipy.signal import resample_poly
        nz = resample_poly(nz, SR, nsr)
    if len(nz) < len(x):  # tile short clips
        nz = np.tile(nz, int(np.ceil(len(x) / len(nz))))
    start = rng.integers(0, len(nz) - len(x) + 1)
    seg = nz[start:start + len(x)]
    ntype = "music" if os.sep + "music" + os.sep in noise_path else "noise"
    return _scale_noise_to_snr(x, seg, snr_db), {
        "snr_db": float(snr_db), "noise_type": ntype,
        "noise_file": os.path.basename(noise_path),
    }


# ----------------------------------------------------------------------------- bandwidth
def lowpass(x, cutoff):
    cutoff = min(cutoff, SR / 2 - 100)
    sos = butter(8, cutoff / (SR / 2), btype="low", output="sos")
    return sosfiltfilt(sos, x), {"cutoff_hz": float(cutoff)}


# ----------------------------------------------------------------------------- clipping
def clip_frac(x, frac):
    """Hard-clip at `frac` of peak, restore peak. Returns measured clipped fraction."""
    peak = np.max(np.abs(x)) + 1e-9
    thr = frac * peak
    clipped = np.clip(x, -thr, thr)
    measured = float(np.mean(np.abs(x) >= thr - 1e-12))
    return clipped * (peak / thr), {"clip_knob": float(frac), "clipped_fraction": measured}


# ----------------------------------------------------------------------------- reverb (real RIRs)
def _drr_db(rir, sr=SR, direct_ms=2.5):
    """Direct-to-reverberant ratio: energy in a window around the peak vs the tail."""
    rir = rir.astype(np.float64)
    p = int(np.argmax(np.abs(rir)))
    w = int(sr * direct_ms / 1000)
    direct = rir[max(0, p - w): p + w + 1]
    tail = np.concatenate([rir[: max(0, p - w)], rir[p + w + 1:]])
    de, te = float(np.sum(direct**2)), float(np.sum(tail**2)) + 1e-12
    return 10 * np.log10((de + 1e-12) / te)


def measure_rt60(rir, sr=SR, decay_db=30.0):
    """RT60 from the impulse response itself, via Schroeder backward integration.

    Fits the energy-decay curve between -5 dB and -(5+decay_db) dB (i.e. T30) and
    extrapolates to a 60 dB decay. Returns None if the RIR is too short/noisy to fit.

    This replaces parsing RT60 out of the filename, which tied the pipeline to one
    specific RIR set. Measuring it works for ANY RIR corpus.
    """
    h = np.asarray(rir, dtype=np.float64)
    if h.ndim > 1:
        h = h[:, 0]
    p = int(np.argmax(np.abs(h)))
    h = h[p:]                                    # start at the direct path
    if len(h) < int(0.05 * sr):
        return None
    edc = np.cumsum((h**2)[::-1])[::-1]          # Schroeder energy decay curve
    if edc[0] <= 0:
        return None
    edc_db = 10.0 * np.log10(edc / edc[0] + 1e-20)
    lo, hi = -5.0, -(5.0 + decay_db)
    i1 = np.argmax(edc_db <= lo)
    i2 = np.argmax(edc_db <= hi)
    if i2 <= i1 or i2 == 0:                      # never decayed that far (noise floor)
        return None
    t = np.arange(i1, i2) / sr
    slope = np.polyfit(t, edc_db[i1:i2], 1)[0]   # dB per second
    if slope >= -1e-6:
        return None
    return float(np.clip(-60.0 / slope, 0.05, 3.0))


def load_rir_bank(rng, per_bin=6, cache=None):
    """Build a RIR bank spread across RT60 bins, measuring RT60 from each response.

    Dataset-agnostic: works with any set of RIR wavs matched by RIR_GLOB. RT60 is
    measured (Schroeder), not parsed from the path, so no filename convention is
    assumed. Results are cached to JSON because measuring a large bank is slow.
    """
    import json

    files = sorted(glob.glob(RIR_GLOB, recursive=True))
    # SLR28 ships noise recordings alongside the RIRs (pointsource_noises/, and 93
    # *_noise_*.wav inside real_rirs_isotropic_noises/). Convolving speech with one of
    # those would silently corrupt the reverb axis — keep only actual impulse responses.
    files = [f for f in files
             if "pointsource_noises" not in f and "noise" not in os.path.basename(f).lower()]
    if not files:
        raise FileNotFoundError(f"no RIRs matched {RIR_GLOB} (set SQA_RIR_ROOT)")

    measured = {}
    if cache and os.path.exists(cache):
        measured = json.load(open(cache))

    todo = [f for f in files if f not in measured]
    if todo:
        # Measuring the full 60k pool is wasteful, but the "slight" band (low RT60 AND
        # high DRR) is rare — ~0.6% of responses — so sample generously to populate it.
        if len(todo) > 20000:
            idx = rng.choice(len(todo), size=20000, replace=False)
            todo = [todo[i] for i in sorted(idx)]
        for f in todo:
            try:
                x, sr = sf.read(f)
                if np.ndim(x) > 1:
                    x = x[:, 0]
                if sr != SR:
                    from scipy.signal import resample_poly
                    x = resample_poly(x, SR, sr)
                x = np.asarray(x, dtype=np.float64)
                x = x / (np.abs(x).max() + 1e-9)
                rt = measure_rt60(x)
                dr = float(_drr_db(x)) if rt is not None else None
            except Exception:
                rt, dr = None, None
            measured[f] = [rt, dr]
        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            json.dump(measured, open(cache, "w"))

    # Balance the bank on the SEVERITY SCORE the RIR will actually produce, not on RT60.
    # Selecting by RT60 systematically under-samples close-mic (high-DRR) responses, and DRR
    # is the stronger predictor of perceived reverberation — so an RT60-balanced bank leaves
    # the "slight" band nearly empty. Import here to avoid a circular import at module load.
    from experiments.planb.targets import score_reverb

    by_score = {4: [], 3: [], 2: [], 1: []}
    for f, v in measured.items():
        rt, dr = (v if isinstance(v, list) else (v, None))
        if rt is None:
            continue
        by_score[int(score_reverb(rt, dr))].append((rt, f))

    bank = []
    for s in (4, 3, 2, 1):
        picks = by_score[s]
        if not picks:
            continue
        n = min(per_bin, len(picks))
        idx = rng.choice(len(picks), size=n, replace=False)
        bank += [picks[i] for i in idx]
    return sorted(bank)  # list of (rt60, path)


def reverberate(clean, rir):
    rir = rir.astype(np.float64)
    rir = rir / (np.abs(rir).max() + 1e-9)
    y = fftconvolve(clean, rir)[: len(clean)]
    y = y * np.sqrt((np.mean(clean**2) + 1e-12) / (np.mean(y**2) + 1e-12))
    return y, {"rt60": None, "drr_db": float(_drr_db(rir)), "method": "rir"}  # rt60 filled by caller


def synth_reverb(clean, rt60, rng):
    """Synthetic exponential-decay reverberation at a target RT60 (the reverb type
    the eval sweep uses), complementing the real measured RIRs so training spans both."""
    L = max(1, int(rt60 * SR))
    t = np.arange(L)
    rir = rng.standard_normal(L) * np.exp(-6.9 * t / (rt60 * SR))
    rir[0] += 1.0  # direct path
    y = fftconvolve(clean, rir)[: len(clean)]
    y = y * np.sqrt((np.mean(clean**2) + 1e-12) / (np.mean(y**2) + 1e-12))
    return y, {"rt60": None, "drr_db": float(_drr_db(rir)), "method": "synthetic"}


# ----------------------------------------------------------------------------- discontinuity
def packet_loss(x, rate, rng, frame_ms=20):
    """Drop frames (zero-fill) at the given rate -> gaps/choppiness."""
    n = int(SR * frame_ms / 1000)
    y = x.copy()
    nf = len(x) // n
    dropped = 0
    for i in range(nf):
        if rng.random() < rate:
            y[i * n:(i + 1) * n] = 0.0
            dropped += 1
    measured = dropped / max(1, nf)
    return y, {"loss_rate": float(rate), "measured_loss": float(measured)}


# ----------------------------------------------------------------------------- loudness
def regain(x, gain_db):
    """Scale level. Negative = too quiet (the common real-world fault)."""
    return x * (10 ** (gain_db / 20)), {"gain_db": float(gain_db)}


# ----------------------------------------------------------------------------- codec
def measure_cutoff(x, energy_frac=0.995):
    """Effective bandwidth: frequency below which `energy_frac` of spectral energy
    lies. ~7-8 kHz for fullband speech, lower for codec/lowpass colored audio."""
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    c = np.cumsum(X)
    if c[-1] < 1e-12:
        return float(SR / 2)
    idx = int(np.searchsorted(c, energy_frac * c[-1]))
    return float(freqs[min(idx, len(freqs) - 1)])


def apply_codec(x, fmt, encoder, bitrate):
    """Encode->decode through a lossy codec (real coloration/artifacts), length-
    matched. Returns (audio, info). info['eff_cutoff_hz'] is measured post-codec."""
    import torch
    from torchaudio.io import AudioEffector, CodecConfig
    wav = torch.from_numpy(np.clip(x, -1, 1).astype(np.float32))[:, None]  # (T, 1)
    cfg = CodecConfig(bit_rate=int(bitrate)) if bitrate else None
    y = AudioEffector(format=fmt, encoder=encoder, codec_config=cfg).apply(wav, SR)
    y = y.numpy()[:, 0].astype(np.float64)
    if len(y) >= len(x):  # codecs add encoder delay/padding -> length-match
        y = y[:len(x)]
    else:
        y = np.concatenate([y, np.zeros(len(x) - len(y))])
    y = y * np.sqrt((np.mean(x**2) + 1e-12) / (np.mean(y**2) + 1e-12))
    return y, {"codec": encoder or fmt, "bitrate": int(bitrate) if bitrate else None,
               "eff_cutoff_hz": measure_cutoff(y)}
