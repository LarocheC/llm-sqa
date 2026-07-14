"""
Analyze the controlled degradation sweep (experiments/results/degradation.jsonl).

Per degradation family, reports how SALMONN's MOS + description respond as the
degradation worsens, alongside DNSMOS / NISQA / PESQ, and whether the description
correctly NAMES the degradation.
"""

import argparse
import json
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# words that count as correctly naming each degradation
RECOG = {
    "noise": ["noise", "background", "hiss", "static", "hum", "noisy"],
    "lowpass": ["muffl", "telephone", "dull", "bandwidth", "high frequenc", "high-frequenc",
                "lacks", "lacking", "tinny", "narrow", "low-pass", "filtered", "thin"],
    "clip": ["distort", "clipping", "clipped", "harsh", "crackl", "artifact", "robotic",
             "mechanical", "buzz", "rough", "grating"],
    "reverb": ["echo", "reverber", "reverb", "room", "hall", "distant", "cavern", "hollow"],
}
ORDER = ["noise", "lowpass", "clip", "reverb"]
LEVELS = {
    "noise": ["snr20", "snr15", "snr10", "snr5", "snr0"],
    "lowpass": ["lp6000", "lp4000", "lp3000", "lp2000"],
    "clip": ["clip0.5", "clip0.3", "clip0.2", "clip0.1"],
    "reverb": ["rt0.3", "rt0.6", "rt0.9", "rt1.2"],
}


def recognizes(text, family):
    t = text.lower()
    return any(k in t for k in RECOG[family])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="experiments/results/degradation.jsonl")
    ap.add_argument("--out", default="experiments/results/DEGRADATION.md")
    args = ap.parse_args()
    df = pd.DataFrame(json.loads(l) for l in open(args.inp))
    df["description"] = df["description"].fillna("")

    clean = df[df.family == "clean"]
    L = ["# Controlled degradation sweep — SALMONN descriptive SQA\n"]
    L.append(f"- {df.id.nunique()} utterances ({', '.join(sorted(df.speaker.unique()))}), "
             f"{len(df)} clips total")
    cm = clean.salmonn_mos.dropna()
    L.append(f"- clean baseline: SALMONN MOS {cm.mean():.2f}, DNSMOS {clean.dnsmos_ovrl.mean():.2f}, "
             f"NISQA {clean.nisqa_mos.mean():.2f}, PESQ {clean.pesq.mean():.2f}\n")

    # monotonicity per family
    L.append("## Monotonicity: Spearman ρ(severity, score) per family\n")
    L.append("(severity increases with degradation; a strong NEGATIVE ρ = score falls as quality drops = good)\n")
    L.append("| family | SALMONN | DNSMOS | NISQA | PESQ |")
    L.append("|---|---|---|---|---|")
    for fam in ORDER:
        sub = df[df.family == fam]
        cells = []
        for col in ["salmonn_mos", "dnsmos_ovrl", "nisqa_mos", "pesq"]:
            s = sub.dropna(subset=[col])
            cells.append(f"{spearmanr(s.severity, s[col])[0]:+.2f}" if len(s) > 4 else "-")
        L.append(f"| {fam} | " + " | ".join(cells) + " |")
    L.append("")

    # per-family detail tables
    for fam in ORDER:
        L.append(f"## {fam}\n")
        L.append("| level | SALMONN MOS | DNSMOS | NISQA | PESQ | names it? |")
        L.append("|---|---|---|---|---|---|")
        # clean row first
        L.append(f"| clean | {cm.mean():.2f} | {clean.dnsmos_ovrl.mean():.2f} | "
                 f"{clean.nisqa_mos.mean():.2f} | {clean.pesq.mean():.2f} | "
                 f"{100*clean.description.apply(lambda t: recognizes(t, fam)).mean():.0f}% |")
        for lvl in LEVELS[fam]:
            sub = df[(df.family == fam) & (df.level == lvl)]
            if not len(sub):
                continue
            L.append(
                f"| {lvl} | {sub.salmonn_mos.dropna().mean():.2f} | {sub.dnsmos_ovrl.mean():.2f} | "
                f"{sub.nisqa_mos.mean():.2f} | {sub.pesq.mean():.2f} | "
                f"{100*sub.description.apply(lambda t: recognizes(t, fam)).mean():.0f}% |"
            )
        # one worst-level example
        worst = df[(df.family == fam) & (df.level == LEVELS[fam][-1])]
        if len(worst):
            ex = worst.iloc[0]
            L.append(f"\n*example @ {ex['level']} (MOS {ex['salmonn_mos']}):* {ex['description'][:300]}\n")

    with open(args.out, "w") as f:
        f.write("\n".join(L))
    print(f"Wrote {args.out}")
    print("\n".join(L))
    _plots(df, clean, cm)


def _plots(df, clean, cm):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # normalized MOS vs severity, per family (SALMONN)
    for fam in ORDER:
        sub = df[df.family == fam]
        xs = [0] + [s for s in sorted(sub.severity.unique())]
        ys = [cm.mean()] + [sub[sub.severity == s].salmonn_mos.dropna().mean() for s in sorted(sub.severity.unique())]
        ax[0].plot(xs, ys, marker="o", label=fam)
    ax[0].set_xlabel("severity (0 = clean)"); ax[0].set_ylabel("SALMONN MOS")
    ax[0].set_title("SALMONN MOS vs degradation severity"); ax[0].legend()
    # recognition rate vs severity
    for fam in ORDER:
        sub = df[df.family == fam]
        sev = sorted(sub.severity.unique())
        ys = [100 * sub[sub.severity == s].description.apply(lambda t: recognizes(t, fam)).mean() for s in sev]
        ax[1].plot(sev, ys, marker="s", label=fam)
    ax[1].set_xlabel("severity"); ax[1].set_ylabel("% descriptions naming the degradation")
    ax[1].set_title("Degradation recognition vs severity"); ax[1].legend()
    fig.tight_layout(); fig.savefig("experiments/results/degradation_sweep.png", dpi=110); plt.close(fig)
    print("Wrote experiments/results/degradation_sweep.png")


if __name__ == "__main__":
    main()
