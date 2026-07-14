"""
Plan B — Stage 1 target derivation.

Turns the KNOWN degradation parameters of a clip into the supervised target the
model learns to emit:

    noise:4 reverberation:2 bandwidth:3 clipping:5 discontinuity:5 loudness:4.
    The speech is noticeably reverberant, as if in a large room, and slightly
    band-limited; little background noise.
    Overall MOS: 2.73

Three pieces, in the scoping doc's "calibrate-then-describe" order:
  1. Per-dimension 1-5 scores  <- from params (noise-free, exact). This is the
     direct fix for "the quality prior overrides the named dimension".
  2. Grounded description       <- templated from the SAME scores; every clause is
     backed by an applied degradation (never hallucinate an axis that's clean).
  3. Overall MOS (2 decimals)   <- fused from PESQ + NISQA + DNSMOS mapped to a
     common 1-5 axis, de-compressed (never copy a single 5-level teacher).
"""

import numpy as np

DIMS = ["noise", "reverberation", "bandwidth", "clipping", "discontinuity", "loudness"]

# The instruction the model is trained to answer (Plan B schema).
PLANB_PROMPT = (
    "Assess the quality of this speech recording. First rate each dimension from "
    "1 (very bad) to 5 (excellent): noise, reverberation, bandwidth, clipping, "
    "discontinuity, loudness. Then describe the main problems you hear in 1-3 "
    "sentences. Finally give an overall MOS from 1.00 to 5.00."
)


# --------------------------------------------------------------- param -> 1-5 score
def _bucket(value, thresholds, scores):
    """thresholds ascending in *severity*; return the score for the first match."""
    for t, s in zip(thresholds, scores):
        if value <= t:
            return s
    return scores[-1]


def score_noise(snr_db):
    if snr_db is None:
        return 5
    # higher SNR = better; invert by scoring on -snr
    return _bucket(-snr_db, [-30, -20, -12, -6, 1e9], [5, 4, 3, 2, 1])


def score_reverb(rt60, drr_db):
    """Reverberation severity from BOTH the decay tail (RT60) and the apparent distance
    (DRR), taking the worse of the two. 4 = slight ... 1 = severe; 5 only if no room was
    applied at all.

    Only un-convolved audio is 'dry' (5). This was verified, not assumed: convolving clean
    speech with every RIR in the bank and measuring PESQ, *none* came out transparent —
    even the most benign response (RT60 0.05 s, DRR +18 dB) scores PESQ 3.75. So any
    applied RIR floors the score at 4.

    Why DRR is co-equal with RT60 rather than a small correction: on a RIR set with varying
    mic distance, DRR predicts perceived degradation far better than RT60 does
    (rho(PESQ,DRR) = +0.67 vs rho(PESQ,RT60) = -0.27) — a long-RT60 room still sounds fairly
    dry when the mic is close. Scoring on RT60 alone tracked perception at rho ~ +0.2;
    taking the worse of the two bands reaches ~ +0.6. (An earlier version keyed off RT60
    with a `DRR < -15 dB` correction, which was tuned to a set of uniformly distant
    measurements and never fires on close-mic RIRs.)
    """
    if rt60 is None:
        return 5  # no room at all
    s_rt = _bucket(rt60, [0.25, 0.45, 0.70, 1e9], [4, 3, 2, 1])
    if drr_db is None:
        return int(np.clip(s_rt, 1, 4))
    # DRR: >=10 dB close/direct ... < -3 dB clearly distant
    s_drr = _bucket(-drr_db, [-10, -2, 3, 1e9], [4, 3, 2, 1])
    return int(np.clip(min(s_rt, s_drr), 1, 4))


def score_bandwidth(cutoff_hz):
    if cutoff_hz is None or cutoff_hz >= 7500:
        return 5
    return _bucket(-cutoff_hz, [-7500, -6000, -4000, -2500, 1e9], [5, 4, 3, 2, 1])


def score_clipping(clipped_fraction):
    if clipped_fraction is None or clipped_fraction <= 0:
        return 5
    return _bucket(clipped_fraction, [0.01, 0.05, 0.15, 0.30, 1e9], [4, 3, 2, 1, 1])


def score_discontinuity(measured_loss):
    if measured_loss is None or measured_loss <= 0:
        return 5
    return _bucket(measured_loss, [0.02, 0.05, 0.10, 0.20, 1e9], [4, 3, 2, 1, 1])


def score_loudness(gain_db):
    if gain_db is None:
        return 5
    return _bucket(abs(gain_db), [3, 6, 12, 20, 1e9], [5, 4, 3, 2, 1])


def scores_from_params(params):
    """params: dict with optional keys noise/reverb/bandwidth/clip/discontinuity/loudness."""
    p = params
    # bandwidth/coloration may come from an explicit low-pass and/or a lossy codec;
    # take the worse of the two. A codec is never transparent -> floor its band at 4.
    bw = score_bandwidth((p.get("bandwidth") or {}).get("cutoff_hz"))
    if p.get("codec"):
        bw = min(bw, max(score_bandwidth(p["codec"].get("eff_cutoff_hz")), 1), 4)
    return {
        "noise": score_noise((p.get("noise") or {}).get("snr_db")),
        "reverberation": score_reverb((p.get("reverb") or {}).get("rt60"),
                                      (p.get("reverb") or {}).get("drr_db")),
        "bandwidth": bw,
        "clipping": score_clipping((p.get("clip") or {}).get("clipped_fraction")),
        "discontinuity": score_discontinuity((p.get("discontinuity") or {}).get("measured_loss")),
        "loudness": score_loudness((p.get("loudness") or {}).get("gain_db")),
    }


# --------------------------------------------------------------- overall MOS
# Design note: pure metric fusion (the original scoping plan) was tested and
# rejected — PESQ floors at ~1.0 for ANY reverb (mild and severe alike) and DNSMOS
# is reverb-blind, so a fused-metric MOS cannot rank the very axes Plan B exists to
# fix. Instead we ANCHOR the MOS on the exact per-dimension scores (which do rank
# severity, noise-free) and use the metrics only as a 30% realism + de-compression
# nudge. This keeps severity ordering while spreading the scale to 2 decimals.
PARAM_WEIGHT = 0.70
MIN_WEIGHT = 0.55  # how much the worst axis dominates the param anchor


def pesq_to_mos(pesq):
    if pesq is None or not np.isfinite(pesq):
        return None
    return float(np.clip(1 + (pesq - 1.04) * (4.0 / (4.64 - 1.04)), 1.0, 5.0))


def fuse_metric_mos(metrics):
    """Mean of available reference MOS predictors on a common 1-5 axis, or None."""
    vals = []
    pm = pesq_to_mos(metrics.get("pesq"))
    if pm is not None:
        vals.append(pm)
    for k in ("nisqa", "dnsmos_ovrl"):
        v = metrics.get(k)
        if v is not None and np.isfinite(v):
            vals.append(float(np.clip(v, 1.0, 5.0)))
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def overall_mos(scores, metrics):
    """Param-anchored MOS (worst-axis-dominated) blended with metric realism.
    Returns (mos_2dp, source_tag)."""
    vals = list(scores.values())
    mos_param = MIN_WEIGHT * min(vals) + (1 - MIN_WEIGHT) * (sum(vals) / len(vals))
    mm, n = fuse_metric_mos(metrics or {})
    if mm is not None:
        mos = PARAM_WEIGHT * mos_param + (1 - PARAM_WEIGHT) * mm
        tag = f"param+metric({n})"
    else:
        mos, tag = mos_param, "param"
    return round(float(np.clip(mos, 1.0, 5.0)), 2), tag


# --------------------------------------------------------------- grounded description
# v2: 5 phrasings per (dimension, score) and many sentence frames, so the corpus
# carries combinatorial lexical variety. The v1 generator had ~2 phrasings/axis and
# the model memorized rigid templates -> on held-out audio it produced the right
# SCORE but a drifting DESCRIPTION. More diversity forces a robust acoustic->prose
# mapping. score 5 = clean -> never mentioned (grounding: only describe what's there).
_PHRASES = {
    "noise": {
        4: ["faint background noise", "a little background hiss", "slight background noise",
            "a low level of background noise", "mild background hiss"],
        3: ["audible background noise", "clearly noticeable background noise", "background hiss that is easy to hear",
            "a moderate level of background noise", "noticeable background noise behind the voice"],
        2: ["strong background noise", "loud, intrusive background noise", "heavy background noise",
            "prominent background noise that competes with the voice", "a high level of background noise"],
        1: ["severe background noise that dominates the signal", "overwhelming background noise",
            "extreme background noise burying the speech", "very loud background noise drowning the voice",
            "background noise so strong the speech is hard to follow"],
    },
    "reverberation": {
        4: ["a hint of room reverberation", "slight reverberation", "a faint sense of room ambience",
            "mild reverberation", "a touch of room echo"],
        3: ["moderate reverberation, as if in a small room", "noticeable reverberation", "clear room reverberation",
            "a moderately reverberant, roomy sound", "reverberation suggesting a small room"],
        2: ["strong reverberation suggesting a large, echoey room", "heavy room reverberation",
            "pronounced reverberation, as if in a hall", "a strongly reverberant, distant sound", "a lot of room echo"],
        1: ["very heavy reverberation, distant and cavernous", "extreme reverberation, as if far away in a large hall",
            "overwhelming reverberation that smears the speech", "severe reverberation making the voice sound far away",
            "cavernous reverberation that drowns the clarity"],
    },
    "bandwidth": {
        4: ["slightly muffled high frequencies", "a mild loss of high-frequency detail", "a slightly dull, muffled tone",
            "gently rolled-off high frequencies", "a faint muffling of the treble"],
        3: ["a noticeably band-limited, telephone-like quality", "a clearly muffled, narrow-band sound",
            "noticeably reduced bandwidth", "a dull, telephone-like timbre", "a muffled sound with limited high frequencies"],
        2: ["a strongly band-limited and muffled sound", "a heavily muffled, narrow tone",
            "strongly reduced bandwidth and quite muffled", "a thick, muffled quality with little high end",
            "a narrow, boxy, muffled sound"],
        1: ["a severely band-limited, very muffled and narrow sound", "an extremely narrow-band, dull tone",
            "severely reduced bandwidth and very muffled", "an extremely muffled, telephone-like sound with no highs",
            "a very narrow, heavily muffled quality"],
    },
    "clipping": {
        4: ["occasional clipping", "light clipping distortion on peaks", "a little clipping on the loudest parts",
            "slight harshness from clipping", "minor clipping distortion"],
        3: ["audible clipping distortion", "clearly audible clipping", "noticeable clipping on the louder passages",
            "a moderately distorted, clipped sound", "clipping that adds a rough edge"],
        2: ["strong clipping distortion", "heavy clipping that harshens the voice", "prominent clipping distortion",
            "a strongly distorted, clipped sound", "clipping that makes the voice crackle"],
        1: ["severe clipping, harshly distorted", "extreme clipping making the voice crackle and break up",
            "very heavy clipping distortion", "a severely clipped, harsh and distorted sound",
            "clipping so strong the voice is badly distorted"],
    },
    "discontinuity": {
        4: ["a few brief dropouts", "occasional short gaps", "a couple of brief interruptions",
            "sporadic short dropouts", "the odd brief gap"],
        3: ["intermittent dropouts and gaps", "noticeable choppiness", "several dropouts breaking the flow",
            "a somewhat choppy, interrupted sound", "periodic gaps in the speech"],
        2: ["frequent dropouts breaking up the speech", "frequent gaps and stutter", "a badly choppy sound with many gaps",
            "frequent interruptions in the audio", "a lot of dropouts disrupting the speech"],
        1: ["severe choppiness with many gaps", "constant dropouts making it hard to follow",
            "extreme stuttering and dropouts", "near-constant gaps breaking up the speech",
            "so many dropouts the speech is hard to follow"],
    },
    "loudness": {
        4: ["a slightly off recording level", "a mildly inappropriate level", "a level that is a little too low or high",
            "a slightly poor loudness level", "a level that is somewhat off"],
        3: ["a noticeably poor recording level", "a clearly too quiet or too loud level", "a level that is clearly mis-set",
            "a noticeably inappropriate loudness", "a level that is too low or too high"],
        2: ["a badly set recording level", "a very poor loudness level", "a level that is far too quiet or too loud",
            "a strongly mis-set recording level", "a seriously inappropriate level"],
        1: ["an extreme level problem", "a severely mis-set recording level", "a level that is drastically too quiet or too loud",
            "an extremely poor loudness level", "a level so far off it harms the audio"],
    },
}

_MUSIC_PHRASES = {
    4: ["faint background music", "quiet background music", "a little background music",
        "soft background music", "low-level background music"],
    3: ["audible background music", "clearly noticeable background music", "background music that is easy to hear",
        "a moderate level of background music", "noticeable background music behind the voice"],
    2: ["loud background music", "intrusive background music", "prominent background music",
        "strong background music competing with the voice", "a high level of background music"],
    1: ["overwhelming background music drowning the speech", "background music dominating the signal",
        "extremely loud background music burying the voice", "background music so loud the speech is hard to follow",
        "music drowning out the speech"],
}

_FRAMES = [
    "The recording has {x}.", "This clip has {x}.", "The speech shows {x}.",
    "I can hear {x}.", "There is {x}.", "The audio is affected by {x}.",
    "This recording suffers from {x}.", "Listening to it, there is {x}.",
]
_CLEAN = [
    "The speech is clean and clear, with no noticeable degradations.",
    "High-quality, clear speech with no audible problems.",
    "Clean, natural speech with no obvious degradations.",
    "The recording sounds clean and natural, with nothing notable wrong.",
    "Clear speech over a clean background; no real issues.",
]


def _join(phrases):
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def describe(scores, rng, ctx=None):
    """Grounded description naming ALL degraded axes (worst first, capped at 4 to
    avoid run-ons), drawn from a large combinatorial phrase/frame space. Only
    mentions dimensions with score < 5. `ctx.noise_type='music'` names a MUSAN music
    background as 'music', not generic 'noise'."""
    ctx = ctx or {}
    degraded = sorted([(s, d) for d, s in scores.items() if s < 5])  # worst first
    if not degraded:
        return rng.choice(_CLEAN)
    phrases = []
    for s, dim in degraded[:4]:
        bank = _MUSIC_PHRASES if (dim == "noise" and ctx.get("noise_type") == "music") else _PHRASES[dim]
        phrases.append(rng.choice(bank[s]))
    return rng.choice(_FRAMES).format(x=_join(phrases))


# --------------------------------------------------------------- assemble target text
def build_target(scores, mos, description):
    score_line = " ".join(f"{d}:{scores[d]}" for d in DIMS) + "."
    return f"{score_line}\n{description}\nOverall MOS: {mos:.2f}"
