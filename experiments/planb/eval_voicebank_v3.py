"""
Re-run the original VoiceBank-DEMAND analyses (FINDINGS issues #1 description-vs-SNR
and #2 MOS-calibration) with the v3 checkpoint, and regenerate the plots — so we can
see whether the fine-tune fixed the "weak coarse rater / floored MOS / weak
calibration" findings of the original SALMONN-SQA model.

Reuses the cached objective + neural metrics (objective_metrics.csv, nisqa.csv) and
the per-file SNR from the original run (voicebank_sqa.jsonl). Only v3 inference on the
824 noisy files is new (cached in v3_voicebank.jsonl). The original model's numbers
come straight from voicebank_sqa.jsonl for a side-by-side.

Usage:
  python -m experiments.planb.eval_voicebank_v3 --infer     # run v3 on 824 noisy
  python -m experiments.planb.eval_voicebank_v3 --analyze   # tables + plots (after infer)
"""

import argparse
import glob
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments import config as cfg  # noqa: E402

# NOTE: salmonn_core (and eval_compare, which imports it) pull in the vendored SALMONN
# package, which is NOT part of a fresh clone. Import them lazily inside infer() so that
# --analyze runs on a bare clone with no model, no GPU and no datasets.

RES = str(cfg.RESULTS_DIR)
# which model's cached inference to analyze; both ship with the repo
OUT = f"{RES}/planb/open_voicebank.jsonl"
TMP = str(cfg.work_dir("v3_vb"))
# noise-environment vocabulary (issue #1: does it name the DEMAND noise type?)
ENV_WORDS = ["outdoor", "chatter", "music", "traffic", "people", "talking", "babble",
             "street", "car", "crowd", "wind", "cafe", "restaurant", "background noise",
             "background", "hiss", "hum", "rumble", "noisy", "noise"]


def find_parquet():
    return str(cfg.voicebank_parquet())


def infer(out, device="cuda:0"):
    import salmonn_core as sc
    from experiments.planb.eval_compare import load_with_ckpt, parse_dim_scores

    os.makedirs(TMP, exist_ok=True)
    os.makedirs(f"{RES}/planb", exist_ok=True)
    ckpt = str(cfg.ckpt_v3(stage=2))
    prompt = json.load(open("experiments/planb/train/test_prompt_planb.json"))["sqa_full"]
    print(f"v3 ckpt: {ckpt}")
    sqa = load_with_ckpt(ckpt, device)
    rows = pq.read_table(find_parquet()).to_pylist()
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out)}
    fh = open(out, "a")
    n = len(done)
    for r in rows:
        rid = r["id"]
        if rid in done:
            continue
        a, _ = sf.read(io.BytesIO(r["noisy"]["bytes"]))
        a = (a[:, 0] if a.ndim == 2 else a).astype(np.float32)
        path = f"{TMP}/{rid}.wav"
        sf.write(path, np.clip(a, -1, 1), 16000)
        raw = sc.generate_sqa(sqa, prompt=prompt, wav_path=path)
        text = sc.clean_output(raw)
        fh.write(json.dumps({"id": rid, "mos": sc.extract_mos(raw), "scores": parse_dim_scores(text),
                             "desc": text, "degenerate": sc.is_degenerate(raw)}) + "\n")
        fh.flush()
        n += 1
        if n % 100 == 0:
            print(f"  {n}/{len(rows)}")
    fh.close()
    print(f"done: {n} records -> {out}")


def desc_prose(text):
    import re
    return " ".join(l for l in text.split("\n")
                    if not re.search(r"(noise|reverberation|bandwidth|clipping|discontinuity|loudness)\s*:\s*[1-5]", l, re.I)
                    and not l.lower().strip().startswith("overall mos"))


def analyze(out, tag="open"):
    v3 = pd.DataFrame([json.loads(l) for l in open(out)])
    v3 = v3.rename(columns={"mos": "mos_v3", "desc": "desc_v3"})  # column name kept; row source is OUT
    snr = pd.DataFrame([{"id": r["id"], "snr_db": r["snr_db"], "mos_orig": r["mos"], "desc_orig": r["description"]}
                        for r in map(json.loads, open(f"{RES}/voicebank_sqa.jsonl")) if r["kind"] == "noisy"])
    obj = pd.read_csv(f"{RES}/objective_metrics.csv")
    nis = pd.read_csv(f"{RES}/nisqa.csv")
    df = v3.merge(snr, on="id").merge(obj, on="id").merge(nis, on="id")
    print(f"merged {len(df)} files\n")

    def rho(a, b):
        m = df[a].notna() & df[b].notna()
        return spearmanr(df[a][m], df[b][m])[0], int(m.sum())

    # ---- Issue #1: MOS vs SNR ----
    print("## Issue #1 — MOS vs SNR (v3 vs original)")
    bins = [(-99, 5), (5, 10), (10, 15), (15, 99)]
    print(f"  {'SNR band':10} | {'orig MOS':>8} | {'v3 MOS':>8} | n")
    for lo, hi in bins:
        s = df[(df.snr_db >= lo) & (df.snr_db < hi)]
        print(f"  {f'{lo}-{hi}':10} | {s.mos_orig.mean():>8.2f} | {s.mos_v3.mean():>8.2f} | {len(s)}")
    print(f"  rho(MOS,SNR): orig {rho('mos_orig','snr_db')[0]:+.3f}  v3 {rho('mos_v3','snr_db')[0]:+.3f}")
    # noise naming by SNR (prose only)
    df["names_env"] = df.desc_v3.apply(lambda t: any(w in desc_prose(t).lower() for w in ENV_WORDS))
    df["names_env_orig"] = df.desc_orig.apply(lambda t: any(w in (t or "").lower() for w in ENV_WORDS))
    lo = df[df.snr_db < 5]; hi = df[df.snr_db >= 15]
    print(f"  names noise (low SNR<5):  orig {lo.names_env_orig.mean():.0%}  v3 {lo.names_env.mean():.0%}")
    print(f"  names noise (high SNR>=15): orig {hi.names_env_orig.mean():.0%}  v3 {hi.names_env.mean():.0%}\n")

    # ---- Issue #2: calibration ----
    print("## Issue #2 — MOS calibration (v3 vs original)")
    preds = [("DNSMOS P.808", "dnsmos_p808"), ("PESQ", "pesq"), ("NISQA MOS", "nisqa_mos"),
             ("DNSMOS OVRL", "dnsmos_ovrl"), ("DNSMOS SIG", "dnsmos_sig")]
    print(f"  {'metric':14} | {'orig rho':>9} | {'v3 rho':>9}")
    for name, col in preds:
        ro = rho("mos_orig", col)[0]; rv = rho("mos_v3", col)[0]
        print(f"  {name:14} | {ro:>+9.3f} | {rv:>+9.3f}")
    print(f"  cross-agreement DNSMOS/NISQA/PESQ ~0.72-0.82 (unchanged baseline)")
    for label, col in [("orig", "mos_orig"), (tag, "mos_v3")]:
        vals = df[col].dropna()
        print(f"  {label} scale: mean {vals.mean():.2f} ± {vals.std():.2f}, range {vals.min():.2f}-{vals.max():.2f}, "
              f"{vals.nunique()} distinct values")
    print()

    # ---- plots ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for a, col, title in zip(ax, ["mos_orig", "mos_v3"], ["original SALMONN-SQA", "v3"]):
        jit = df[col] + np.random.uniform(-0.05, 0.05, len(df))
        a.scatter(df.snr_db, jit, s=7, alpha=0.3)
        a.set_xlabel("SNR (dB)"); a.set_ylabel("MOS"); a.set_ylim(1, 5.2)
        a.set_title(f"{title}  (rho={rho(col,'snr_db')[0]:+.2f})")
    fig.suptitle("MOS vs SNR — VoiceBank-DEMAND noisy (824 files)")
    fig.tight_layout(); fig.savefig(f"{RES}/planb/mos_vs_snr_{tag}.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for a, col, title in zip(ax, ["dnsmos_ovrl", "nisqa_mos", "pesq"], ["DNSMOS OVRL", "NISQA MOS", "PESQ"]):
        s = df.dropna(subset=[col, "mos_v3"])
        jit = s.mos_v3 + np.random.uniform(-0.05, 0.05, len(s))
        a.scatter(s[col], jit, s=8, alpha=0.3)
        a.set_xlabel(title); a.set_ylabel("v3 MOS")
        a.set_title(f"{title}  (rho={spearmanr(s[col], s.mos_v3)[0]:+.2f})")
    fig.suptitle("v3 MOS vs neural/objective metrics — VoiceBank-DEMAND noisy")
    fig.tight_layout(); fig.savefig(f"{RES}/planb/mos_vs_neural_{tag}.png", dpi=110); plt.close(fig)
    print(f"wrote {RES}/planb/mos_vs_snr_{tag}.png and mos_vs_neural_{tag}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", choices=["open", "v3"], default="open",
                    help="which cached results to analyze (both are committed)")
    args = ap.parse_args()
    OUT = f"{RES}/planb/{args.model}_voicebank.jsonl"
    if args.infer:
        infer(OUT, args.device)
    if args.analyze:
        analyze(OUT, args.model)
