"""
Plan A — neutral forced-choice & rubric prompts.

The leading yes/no probe (prompt_probe.py) gave d' ~ 0 (the model just says "YES"
on clean and degraded alike). Here we remove the presupposition:

  * 2AFC: two balanced labels, neither a "yes" (DRY/REVERBERANT, FULLBAND/MUFFLED,
    CLEAN/DISTORTED), run in BOTH option orders to expose position bias. Scored
    with signal-detection d' = z(hit) - z(false-alarm), using clean clips as the
    false-alarm control.
  * Rubric: rate the degradation 1-5; check Spearman(rating, true severity).

If d' > 0 / rating rises with severity, the perception is latent but un-elicited;
if still ~0, the model genuinely can't hear it (-> Plan B, retrain).
"""

import argparse
import io
import json
import os
import re
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import config as cfg  # noqa: E402
import salmonn_core  # noqa: E402
from experiments.degradation_sweep import clip_frac, lowpass, reverb  # noqa: E402

RESULTS = str(cfg.RESULTS_DIR)
TMP = str(cfg.work_dir("scratch") / "_fc.wav")

DEGS = {
    "reverb": {
        "fn": reverb, "mid": ("rt0.6", 0.6), "worst": ("rt1.2", 1.2),
        "labels": ("DRY", "REVERBERANT"),
        "gloss": "DRY = close/studio sound with no echo; REVERBERANT = room/hall/distant with echo",
        "dim": "reverberation or echo", "lo": "completely dry, no echo", "hi": "very reverberant",
    },
    "lowpass": {
        "fn": lowpass, "mid": ("lp3000", 3000), "worst": ("lp2000", 2000),
        "labels": ("FULLBAND", "MUFFLED"),
        "gloss": "FULLBAND = natural, crisp high frequencies; MUFFLED = band-limited, telephone-like, missing highs",
        "dim": "muffling / missing high frequencies", "lo": "full natural bandwidth", "hi": "very muffled, telephone-like",
    },
    "clip": {
        "fn": clip_frac, "mid": ("clip0.3", 0.3), "worst": ("clip0.1", 0.1),
        "labels": ("CLEAN", "DISTORTED"),
        "gloss": "CLEAN = no distortion; DISTORTED = clipped, harsh, buzzy",
        "dim": "clipping / harsh distortion", "lo": "completely clean", "hi": "heavily clipped and distorted",
    },
}


def afc_prompt(d, order):
    a, b = d["labels"] if order == 0 else d["labels"][::-1]
    return (f"Listen to this recording. Which single word best describes it: {a} or {b}? "
            f"({d['gloss']}.) Reply with only one word — {a} or {b} — and nothing else.")


def rubric_prompt(d):
    return (f"On a scale of 1 to 5, how much {d['dim']} does this recording have? "
            f"1 = {d['lo']}, 5 = {d['hi']}. Reply with a single number from 1 to 5.")


def parse_afc(text, labels):
    t = text.lower()
    ha, hb = labels[0].lower() in t, labels[1].lower() in t
    if ha and not hb:
        return labels[0]
    if hb and not ha:
        return labels[1]
    return None


def parse_rubric(text):
    m = re.search(r"\b([1-5])(?:\.\d+)?\b", text)
    return int(m.group(1)) if m else None


def dprime(hit, nh, fa, nf):
    h = (hit + 0.5) / (nh + 1)
    f = (fa + 0.5) / (nf + 1)
    return float(norm.ppf(h) - norm.ppf(f))


def infer(sqa, audio, prompt):
    sf.write(TMP, np.clip(audio, -1, 1).astype(np.float32), 16000)
    raw = salmonn_core.generate_sqa(
        sqa, prompt=prompt, samples=salmonn_core.prepare_audio_sample(TMP, sqa.wav_processor, device=sqa.device))
    return salmonn_core.clean_output(raw)


def run(args):
    rows = pq.read_table(str(cfg.voicebank_parquet())).to_pylist()
    by_spk = {}
    for r in rows:
        by_spk.setdefault(r["id"].split("_")[0], []).append(r)
    sample = []
    for spk, rs in sorted(by_spk.items()):
        sample += rs[: args.per_speaker]

    sqa = salmonn_core.load_model(device_name=args.device)
    out = open(args.out, "w")
    for row in sample:
        rid = row["id"]
        clean, _ = sf.read(io.BytesIO(row["clean"]["bytes"]))
        clean = (clean[:, 0] if clean.ndim == 2 else clean).astype(np.float64)
        for fam, d in DEGS.items():
            variants = {"clean": clean, "mid": d["fn"](clean, d["mid"][1]), "worst": d["fn"](clean, d["worst"][1])}
            # 2AFC on clean + worst, both orders
            for sev in ("clean", "worst"):
                for order in (0, 1):
                    txt = infer(sqa, variants[sev], afc_prompt(d, order))
                    out.write(json.dumps({"id": rid, "family": fam, "task": "2afc", "severity": sev,
                                          "order": order, "first_label": (d["labels"] if order == 0 else d["labels"][::-1])[0],
                                          "choice": parse_afc(txt, d["labels"]), "text": txt[:200]}) + "\n")
            # rubric on clean/mid/worst
            for sev in ("clean", "mid", "worst"):
                txt = infer(sqa, variants[sev], rubric_prompt(d))
                out.write(json.dumps({"id": rid, "family": fam, "task": "rubric", "severity": sev,
                                      "rating": parse_rubric(txt), "text": txt[:200]}) + "\n")
            out.flush()
        print(f"  {rid} done")
    out.close()
    summarize(args.out)


def summarize(path):
    recs = [json.loads(l) for l in open(path)]
    print("\n=== 2AFC: discrimination (d') and biases ===")
    print(f"{'family':8s} | hit(worst→deg) | FA(clean→deg) |   d'   | order-consistent | picks-1st-option")
    for fam, d in DEGS.items():
        deg_label = d["labels"][1]
        af = [r for r in recs if r["family"] == fam and r["task"] == "2afc"]
        worst = [r for r in af if r["severity"] == "worst"]
        clean = [r for r in af if r["severity"] == "clean"]
        hit = sum(r["choice"] == deg_label for r in worst)
        fa = sum(r["choice"] == deg_label for r in clean)
        dp = dprime(hit, len(worst), fa, len(clean))
        # order consistency: per (id,severity) do the two orders agree?
        pairs = {}
        for r in af:
            pairs.setdefault((r["id"], r["severity"]), {})[r["order"]] = r["choice"]
        cons = [v for v in pairs.values() if 0 in v and 1 in v]
        consistent = np.mean([v[0] == v[1] for v in cons]) if cons else float("nan")
        first_bias = np.mean([r["choice"] == r["first_label"] for r in af if r["choice"]]) if af else float("nan")
        print(f"{fam:8s} | {hit:3d}/{len(worst):<3d} {100*hit/max(len(worst),1):3.0f}% | "
              f"{fa:3d}/{len(clean):<3d} {100*fa/max(len(clean),1):3.0f}% | {dp:+5.2f} | "
              f"{100*consistent:14.0f}% | {100*first_bias:.0f}%")

    print("\n=== Rubric: mean rating by severity (Spearman vs severity) ===")
    sev_rank = {"clean": 0, "mid": 1, "worst": 2}
    print(f"{'family':8s} | clean | mid | worst | Spearman | parsed")
    for fam in DEGS:
        rb = [r for r in recs if r["family"] == fam and r["task"] == "rubric" and r["rating"] is not None]
        means = {}
        for sev in ("clean", "mid", "worst"):
            vals = [r["rating"] for r in rb if r["severity"] == sev]
            means[sev] = np.mean(vals) if vals else float("nan")
        xs = [sev_rank[r["severity"]] for r in rb]
        ys = [r["rating"] for r in rb]
        from scipy.stats import spearmanr
        rho = spearmanr(xs, ys)[0] if len(set(ys)) > 1 else float("nan")
        n = len([r for r in recs if r["family"] == fam and r["task"] == "rubric"])
        print(f"{fam:8s} | {means['clean']:.2f} | {means['mid']:.2f} | {means['worst']:.2f} | "
              f"{rho:+.2f} | {len(rb)}/{n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{RESULTS}/forced_choice.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-speaker", type=int, default=6)
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    if a.summarize_only:
        summarize(a.out)
    else:
        run(a)
