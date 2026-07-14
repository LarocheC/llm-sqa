"""
Analyze how SALMONN's descriptive SQA output varies with SNR and clean/noisy
condition over the VoiceBank-DEMAND-16k test set.

Reads the JSONL produced by voicebank_demand_sqa.py and writes a markdown report
(+ PNG plots) covering:
  * SNR distribution (sanity-check against the nominal VB-DEMAND test SNRs)
  * MOS vs SNR and description length vs SNR (with rank correlations)
  * how the prevalence of noise / quality keywords tracks SNR
  * which words most distinguish noisy descriptions from clean ones
"""

import argparse
import json
import math
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Keyword groups (lowercased substring match on the description).
NOISE_TERMS = [
    "noise", "noisy", "background", "hiss", "static", "hum", "buzz", "crackle",
    "distortion", "distorted", "muffled", "interference", "artifact", "rustling",
    "chatter", "echo", "reverberation", "clipping", "wind",
]
CLEAN_TERMS = [
    "clear", "clean", "clarity", "crisp", "smooth", "natural", "pristine",
    "intelligib", "easy to understand", "easy to follow",
]
VERDICTS = ["excellent", "good", "fair", "poor", "bad", "moderate"]
STOPWORDS = set(
    "the a an and or of to in is are with no not very this that it its on at as "
    "be has have but which while there their they i you he she we for from any "
    "some can may also more most than then so such into out up down over under "
    "speech audio voice speaker sound quality overall listener listening".split()
)


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def has(text, term):
    return term in text.lower()


def tokenize(text):
    return [w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 2 and w not in STOPWORDS]


def snr_bucket(snr):
    if snr is None or (isinstance(snr, float) and math.isinf(snr)):
        return None
    for lo, hi, label in [(-99, 5, "<5"), (5, 10, "5-10"), (10, 15, "10-15"), (15, 99, ">=15")]:
        if lo <= snr < hi:
            return label
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="experiments/results/voicebank_sqa.jsonl")
    ap.add_argument("--out", default="experiments/results/REPORT.md")
    args = ap.parse_args()

    df = load(args.inp)
    df["description"] = df["description"].fillna("")
    noisy = df[df.kind == "noisy"].copy()
    clean = df[df.kind == "clean"].copy()
    # genuine = non-degenerate descriptions (terse score-only replies excluded)
    gen_noisy = noisy[~noisy.degenerate].copy()
    gen_clean = clean[~clean.degenerate].copy()

    L = []
    L.append("# VoiceBank-DEMAND-16k — SALMONN descriptive SQA analysis\n")
    L.append(f"- rows: {len(df)} ({len(noisy)} noisy, {len(clean)} clean)")
    L.append(f"- genuine (non-degenerate) descriptions: {len(gen_noisy)} noisy, {len(gen_clean)} clean")
    L.append(f"- terse/score-only replies: noisy {int(noisy.degenerate.sum())}, clean {int(clean.degenerate.sum())}\n")

    # ---- SNR distribution ----
    snr = noisy["snr_db"].replace([np.inf, -np.inf], np.nan).dropna()
    L.append("## SNR distribution (computed per pair: 10·log10(Σclean²/Σ(noisy−clean)²))\n")
    L.append(f"- min/median/mean/max: {snr.min():.1f} / {snr.median():.1f} / {snr.mean():.1f} / {snr.max():.1f} dB")
    qs = snr.quantile([0.25, 0.5, 0.75]).round(1).tolist()
    L.append(f"- quartiles: {qs}\n")

    noisy["bucket"] = noisy["snr_db"].apply(snr_bucket)
    gen_noisy["bucket"] = gen_noisy["snr_db"].apply(snr_bucket)
    order = ["<5", "5-10", "10-15", ">=15"]

    # ---- MOS & length vs SNR ----
    def rankcorr(a, b):
        m = a.notna() & b.notna()
        if m.sum() < 5:
            return float("nan")
        return pd.Series(a[m]).corr(pd.Series(b[m]), method="spearman")

    L.append("## MOS and description length vs SNR (noisy files)\n")
    L.append(f"- Spearman corr(SNR, MOS) = {rankcorr(noisy['snr_db'], noisy['mos']):.3f}")
    L.append(f"- Spearman corr(SNR, n_words) = {rankcorr(noisy['snr_db'], noisy['n_words']):.3f}\n")
    L.append("| SNR bucket (dB) | n | mean MOS | mean #words | terse-reply rate |")
    L.append("|---|---|---|---|---|")
    for b in order:
        sub = noisy[noisy.bucket == b]
        if len(sub) == 0:
            continue
        mos = sub["mos"].dropna()
        L.append(
            f"| {b} | {len(sub)} | {mos.mean():.2f} | {sub['n_words'].mean():.0f} | "
            f"{sub['degenerate'].mean()*100:.0f}% |"
        )
    L.append("")
    clean_mos = clean["mos"].dropna()
    L.append(f"- clean files: mean MOS {clean_mos.mean():.2f} (n={len(clean_mos)}), "
             f"mean #words {clean['n_words'].mean():.0f}, terse-reply rate {clean['degenerate'].mean()*100:.0f}%\n")

    # ---- keyword prevalence vs SNR ----
    L.append("## Noise/quality keyword prevalence by SNR bucket (% of genuine noisy descriptions)\n")
    track = ["noise", "background", "distortion", "muffled", "clear", "clean", "smooth", "natural", "effort"]
    L.append("| keyword | " + " | ".join(order) + " | clean |")
    L.append("|---|" + "---|" * (len(order) + 1))
    for kw in track:
        cells = []
        for b in order:
            sub = gen_noisy[gen_noisy.bucket == b]
            cells.append(f"{100*sub['description'].apply(lambda t: has(t, kw)).mean():.0f}%" if len(sub) else "-")
        cl = f"{100*gen_clean['description'].apply(lambda t: has(t, kw)).mean():.0f}%" if len(gen_clean) else "-"
        L.append(f"| {kw} | " + " | ".join(cells) + f" | {cl} |")
    L.append("")

    # ---- verdict words ----
    L.append("## Quality verdict words (% of genuine descriptions containing)\n")
    L.append("| verdict | noisy | clean |")
    L.append("|---|---|---|")
    for v in VERDICTS:
        nf = gen_noisy["description"].apply(lambda t: has(t, v)).mean() if len(gen_noisy) else 0
        cf = gen_clean["description"].apply(lambda t: has(t, v)).mean() if len(gen_clean) else 0
        L.append(f"| {v} | {100*nf:.0f}% | {100*cf:.0f}% |")
    L.append("")

    # ---- distinguishing words: noisy vs clean (log-odds with +1 smoothing) ----
    def wordcounts(frame):
        c = Counter()
        for t in frame["description"]:
            c.update(set(tokenize(t)))
        return c

    cn, cc = wordcounts(gen_noisy), wordcounts(gen_clean)
    Nn, Nc = max(len(gen_noisy), 1), max(len(gen_clean), 1)
    vocab = {w for w, k in (cn + cc).items() if k >= max(5, 0.02 * (Nn + Nc))}
    scored = []
    for w in vocab:
        pn = (cn[w] + 1) / (Nn + 2)
        pc = (cc[w] + 1) / (Nc + 2)
        scored.append((math.log(pn / pc), w, 100 * cn[w] / Nn, 100 * cc[w] / Nc))
    scored.sort(reverse=True)
    L.append("## Words most over-represented in NOISY vs clean descriptions\n")
    L.append("| word | noisy % | clean % | log-odds |")
    L.append("|---|---|---|---|")
    for lo, w, pn, pc in scored[:15]:
        L.append(f"| {w} | {pn:.0f}% | {pc:.0f}% | {lo:+.2f} |")
    L.append("\n## Words most over-represented in CLEAN vs noisy descriptions\n")
    L.append("| word | clean % | noisy % | log-odds |")
    L.append("|---|---|---|---|")
    for lo, w, pn, pc in scored[-15:][::-1]:
        L.append(f"| {w} | {pc:.0f}% | {pn:.0f}% | {-lo:+.2f} |")
    L.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(L))
    print(f"Wrote {args.out}")

    # ---- plots ----
    _plots(noisy, clean, gen_noisy, order, track)


def _plots(noisy, clean, gen_noisy, order, track):
    # MOS vs SNR
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    s = noisy.dropna(subset=["snr_db", "mos"])
    ax[0].scatter(s["snr_db"], s["mos"], s=8, alpha=0.4)
    ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("MOS"); ax[0].set_title("MOS vs SNR (noisy)")
    data = [noisy[noisy.snr_db.apply(lambda x: snr_bucket(x) == b)]["mos"].dropna() for b in order]
    data = [d for d in data if len(d)]
    if data:
        ax[1].boxplot(data, labels=[b for b in order if len(noisy[noisy.snr_db.apply(lambda x: snr_bucket(x) == b)]["mos"].dropna())])
    ax[1].set_xlabel("SNR bucket (dB)"); ax[1].set_ylabel("MOS"); ax[1].set_title("MOS by SNR bucket")
    fig.tight_layout(); fig.savefig("experiments/results/mos_vs_snr.png", dpi=110); plt.close(fig)

    # keyword prevalence vs SNR
    fig, ax = plt.subplots(figsize=(9, 5))
    for kw in track:
        ys = []
        for b in order:
            sub = gen_noisy[gen_noisy.snr_db.apply(lambda x: snr_bucket(x) == b)]
            ys.append(100 * sub["description"].apply(lambda t: has(t, kw)).mean() if len(sub) else np.nan)
        ax.plot(order, ys, marker="o", label=kw)
    ax.set_xlabel("SNR bucket (dB)"); ax.set_ylabel("% of descriptions"); ax.set_title("Keyword prevalence vs SNR")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig("experiments/results/keywords_vs_snr.png", dpi=110); plt.close(fig)
    print("Wrote experiments/results/mos_vs_snr.png, keywords_vs_snr.png")


if __name__ == "__main__":
    main()
