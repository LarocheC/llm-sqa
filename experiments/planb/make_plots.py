"""
Regenerate every Plan B figure from the committed eval outputs.

No GPU, no datasets — reads only experiments/results/**. Run after any eval to refresh
the plots used by FINDINGS.md, PLAN_B_SUMMARY.md and the slide deck.

  uv run python -m experiments.planb.make_plots
"""

import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments import config as cfg  # noqa: E402

R = cfg.PLANB_DIR
RES = cfg.RESULTS_DIR
ORIG, V3, OPEN = "#8892a0", "#d29922", "#3fb950"


def jl(p):
    return [json.loads(l) for l in open(p)]


# ---------------------------------------------------------------- 1. degradation sweep
def degradation_sweep():
    rows = jl(R / "eval_compare_open.jsonl")
    fams = ["noise", "lowpass", "clip", "reverb"]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for a, fam in zip(ax, fams):
        for tag, col, lab in [("orig", ORIG, "original"), ("planb", OPEN, "open")]:
            sub = [r for r in rows if r["family"] == fam and r["tag"] == tag and r["mos"] is not None]
            if not sub:
                continue
            sev = sorted({r["severity"] for r in sub})
            means = [np.mean([r["mos"] for r in sub if r["severity"] == s]) for s in sev]
            rho = spearmanr([r["severity"] for r in sub], [r["mos"] for r in sub])[0]
            a.plot(sev, means, "o-", color=col, label=f"{lab} (ρ={rho:+.2f})", lw=2)
        a.set_title(fam); a.set_xlabel("severity →"); a.grid(alpha=.2)
        a.legend(fontsize=8)
    ax[0].set_ylabel("MOS")
    fig.suptitle("Degradation sweep — does the MOS fall as the degradation worsens?")
    fig.tight_layout(); fig.savefig(R / "degradation_sweep_open.png", dpi=120); plt.close(fig)


# ---------------------------------------------------------------- 2. MOS scale usage
def scale_usage():
    v3 = [r["mos"] for r in jl(R / "v3_voicebank.jsonl") if r.get("mos") is not None]
    op = [r["mos"] for r in jl(R / "open_voicebank.jsonl") if r.get("mos") is not None]
    orig = [r["mos"] for r in jl(RES / "voicebank_sqa.jsonl")
            if r["kind"] == "noisy" and r.get("mos") is not None]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
    for a, (vals, col, name) in zip(ax, [(orig, ORIG, "original"), (v3, V3, "v3"), (op, OPEN, "open")]):
        a.hist(vals, bins=np.arange(1, 5.15, 0.1), color=col, edgecolor="white")
        a.set_title(f"{name} — {len(set(vals))} distinct values")
        a.set_xlabel("MOS"); a.set_xlim(1, 5); a.grid(alpha=.2)
    ax[0].set_ylabel("# files")
    fig.suptitle("MOS scale usage on VoiceBank-DEMAND (824 noisy files)")
    fig.tight_layout(); fig.savefig(R / "mos_scale_usage_open.png", dpi=120); plt.close(fig)


# ---------------------------------------------------------------- 3. enhancement
def enhancement():
    op = jl(R / "open_enhancement.jsonl")
    by = {(r["id"], r["kind"]): r for r in op}
    ids = sorted({r["id"] for r in op})
    go, gv, gp = [], [], []
    for i in ids:
        n, e = by.get((i, "noisy")), by.get((i, "enhanced"))
        if n and e and None not in (n["mos_v3"], e["mos_v3"], n["mos_orig"], e["mos_orig"], n["pesq"], e["pesq"]):
            go.append(e["mos_orig"] - n["mos_orig"]); gv.append(e["mos_v3"] - n["mos_v3"])
            gp.append(e["pesq"] - n["pesq"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, (g, col, name) in zip(ax, [(go, ORIG, "original"), (gv, OPEN, "open")]):
        a.scatter(gp, g, s=26, color=col, alpha=.75)
        a.axhline(0, color="#888", lw=1, ls="--")
        a.set_xlabel("PESQ gain (noisy → enhanced)"); a.set_ylabel("MOS gain")
        a.set_title(f"{name}:  mean {np.mean(g):+.2f},  ρ={spearmanr(g, gp)[0]:+.2f}")
        a.grid(alpha=.2)
    fig.suptitle("Does the model notice that a denoiser improved the audio?")
    fig.tight_layout(); fig.savefig(R / "enhancement_open.png", dpi=120); plt.close(fig)


# ---------------------------------------------------------------- 4. THE reverb finding
def reverb_real_vs_synthetic():
    """Parse the two PESQ-referenced reverb tests out of the eval logs.

    This is the result that matters: v3 and open disagree about reverb *only because they
    were calibrated on different reverb distributions*. Synthetic exp-decay reverb carries
    an artificial direct path (high DRR at every RT60), which is exactly what v3's
    RT60-only map is tuned for.
    """
    log = R / "realreverb.log"
    real = {}
    if log.exists():
        txt = open(log).read()
        for m in re.finditer(r"^(v3|open)\s+([+-]?\d\.\d+)\s+([+-]?\d\.\d+)", txt, re.M):
            real[m.group(1)] = float(m.group(2))
    synth = {"v3": 0.610, "open": 0.391}   # from the PESQ-referenced sweep (see FINDINGS)
    if not real:
        print("  ! realreverb.log not found — skipping reverb figure")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(2); w = 0.35
    ax.bar(x - w/2, [synth["v3"], real["v3"]], w, color=V3, label="v3 (non-public RIRs)")
    ax.bar(x + w/2, [synth["open"], real["open"]], w, color=OPEN, label="open (public RIRs)")
    for i, (a_, b_) in enumerate([(synth["v3"], synth["open"]), (real["v3"], real["open"])]):
        ax.text(i - w/2, a_ + .015, f"{a_:.2f}", ha="center", fontsize=10)
        ax.text(i + w/2, b_ + .015, f"{b_:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["SYNTHETIC reverb\n(exp-decay: artificial direct path)",
                        "REAL reverb\n(held-out room impulse responses)"])
    ax.set_ylabel("ρ(model MOS, PESQ)")
    ax.set_title("Reverb: which model agrees with an independent quality metric?")
    ax.legend(loc="upper center", ncol=2, fontsize=9, framealpha=.95)
    ax.grid(alpha=.2, axis="y"); ax.set_ylim(0, .95)
    fig.tight_layout(); fig.savefig(R / "reverb_real_vs_synthetic.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    degradation_sweep(); print("  degradation_sweep_open.png")
    scale_usage(); print("  mos_scale_usage_open.png")
    enhancement(); print("  enhancement_open.png")
    reverb_real_vs_synthetic(); print("  reverb_real_vs_synthetic.png")
    print(f"-> {R}")
