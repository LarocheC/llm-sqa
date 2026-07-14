"""Analyze the enhancement experiment (experiments/results/enhancement.jsonl)."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# artifact / residual-issue vocabulary an enhancer can introduce
ARTIFACT = ["muffl", "robotic", "metallic", "watery", "musical", "artifact", "distort",
            "unnatural", "suppress", "over-suppress", "tinny", "hollow", "echo", "dull"]


def has_any(t, kws):
    t = t.lower()
    return any(k in t for k in kws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="experiments/results/enhancement.jsonl")
    ap.add_argument("--out", default="experiments/results/ENHANCEMENT.md")
    args = ap.parse_args()
    df = pd.DataFrame(json.loads(l) for l in open(args.inp))
    df["description"] = df["description"].fillna("")
    piv = {k: df[df.kind == k] for k in ["clean", "noisy", "enhanced"]}

    L = ["# Enhancement experiment — does SALMONN track ConvFSENet enhancement?\n"]
    L.append(f"- {df.id.nunique()} utterances x (clean, noisy, enhanced)\n")

    L.append("## Mean score by condition\n")
    L.append("| condition | SALMONN MOS | PESQ | DNSMOS | NISQA |")
    L.append("|---|---|---|---|---|")
    for k in ["noisy", "enhanced", "clean"]:
        d = piv[k]
        L.append(f"| {k} | {d.salmonn_mos.dropna().mean():.2f} | {d.pesq.dropna().mean():.2f} | "
                 f"{d.dnsmos_ovrl.dropna().mean():.2f} | {d.nisqa_mos.dropna().mean():.2f} |")
    L.append("")

    # per-file noisy -> enhanced deltas (paired by id)
    m = piv["noisy"].set_index("id")
    e = piv["enhanced"].set_index("id")
    ids = m.index.intersection(e.index)
    dmos = (e.loc[ids, "salmonn_mos"] - m.loc[ids, "salmonn_mos"]).dropna()
    dpesq = (e.loc[ids, "pesq"] - m.loc[ids, "pesq"]).dropna()
    dnis = (e.loc[ids, "nisqa_mos"] - m.loc[ids, "nisqa_mos"]).dropna()
    L.append("## Did SALMONN notice the enhancement? (per-file noisy -> enhanced)\n")
    L.append(f"- SALMONN MOS Δ: mean {dmos.mean():+.2f}; improved {100*(dmos>0).mean():.0f}%, "
             f"unchanged {100*(dmos==0).mean():.0f}%, worsened {100*(dmos<0).mean():.0f}%")
    L.append(f"- PESQ Δ: mean {dpesq.mean():+.2f} (improved {100*(dpesq>0).mean():.0f}%)")
    L.append(f"- NISQA Δ: mean {dnis.mean():+.2f} (improved {100*(dnis>0).mean():.0f}%)")
    common = dmos.index.intersection(dpesq.index)
    if len(common) > 4:
        L.append(f"- Spearman ρ(SALMONN MOS Δ, PESQ Δ) = {spearmanr(dmos.loc[common], dpesq.loc[common])[0]:+.2f}")
    L.append("")

    L.append("## Does SALMONN flag enhancement artifacts? (enhanced vs clean descriptions)\n")
    L.append("| vocabulary | enhanced | clean | noisy |")
    L.append("|---|---|---|---|")
    L.append(f"| any artifact term | {100*piv['enhanced'].description.apply(lambda t: has_any(t, ARTIFACT)).mean():.0f}% | "
             f"{100*piv['clean'].description.apply(lambda t: has_any(t, ARTIFACT)).mean():.0f}% | "
             f"{100*piv['noisy'].description.apply(lambda t: has_any(t, ARTIFACT)).mean():.0f}% |")
    for kw in ["muffl", "distort", "unnatural", "robotic", "artifact"]:
        L.append(f"| {kw} | {100*piv['enhanced'].description.apply(lambda t: kw in t.lower()).mean():.0f}% | "
                 f"{100*piv['clean'].description.apply(lambda t: kw in t.lower()).mean():.0f}% | "
                 f"{100*piv['noisy'].description.apply(lambda t: kw in t.lower()).mean():.0f}% |")
    L.append("")

    # example enhanced descriptions
    L.append("## Example enhanced descriptions\n")
    for _, r in piv["enhanced"].head(3).iterrows():
        L.append(f"- **{r['id']}** (MOS {r['salmonn_mos']}, PESQ {r['pesq'] and round(r['pesq'],2)}): {r['description'][:260]}")
    L.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(L))
    print(f"Wrote {args.out}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
