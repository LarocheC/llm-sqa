"""
Calibration of SALMONN's descriptive-SQA MOS against objective + neural metrics.

Reuses the existing inference run (experiments/results/voicebank_sqa.jsonl): no
new SALMONN inference. For each VoiceBank-DEMAND test pair we compute, on the
exact parquet audio SALMONN saw:
  * intrusive metrics (need clean ref): PESQ (wb), STOI, SI-SDR, segmental SNR
  * reference-free neural MOS predictors: DNSMOS P.835 (sig/bak/ovrl/p808) and
    NISQA v2 (mos + noisiness/discontinuity/coloration/loudness)
then correlate them all with SALMONN's MOS.

DNSMOS and NISQA are the apt comparison: like SALMONN they are reference-free and
trained on human MOS, so "does SALMONN agree with them as well as they agree with
each other?" is the real calibration question.
"""

import glob
import io
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from pesq import pesq
from pystoi import stoi
from scipy.stats import pearsonr, spearmanr
from speechmos import dnsmos
from experiments import config as cfg  # noqa: E402

RESULTS = str(cfg.RESULTS_DIR)
METRICS_CSV = f"{RESULTS}/objective_metrics.csv"
NISQA_CSV = f"{RESULTS}/nisqa.csv"
NOISY_DIR = str(cfg.work_dir("vb_noisy"))
SR = 16000
PY = sys.executable


def find_test_parquet():
    return str(cfg.voicebank_parquet())


def decode(struct):
    a, sr = sf.read(io.BytesIO(struct["bytes"]))
    if a.ndim == 2:
        a = a[:, 0]
    return a.astype(np.float64), sr


def si_sdr(ref, est):
    ref, est = ref - ref.mean(), est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = a * ref
    return 10 * np.log10((np.sum(target**2) + 1e-12) / (np.sum((est - target) ** 2) + 1e-12))


def seg_snr(ref, est, frame=400):
    n = (len(ref) // frame) * frame
    r = ref[:n].reshape(-1, frame)
    d = (ref[:n] - est[:n]).reshape(-1, frame)
    rp, dp = np.sum(r**2, axis=1), np.sum(d**2, axis=1)
    mask = rp > 1e-7
    if mask.sum() == 0:
        return np.nan
    return float(np.clip(10 * np.log10((rp[mask] + 1e-10) / (dp[mask] + 1e-10)), -10, 35).mean())


def compute_metrics():
    """Intrusive metrics + DNSMOS, computed from the parquet. Also dumps noisy
    wavs to NOISY_DIR for NISQA. Cached in METRICS_CSV."""
    if os.path.exists(METRICS_CSV) and os.path.isdir(NOISY_DIR) and len(os.listdir(NOISY_DIR)) >= 800:
        print(f"Loading cached {METRICS_CSV}")
        return pd.read_csv(METRICS_CSV)
    os.makedirs(NOISY_DIR, exist_ok=True)
    rows = pq.read_table(find_test_parquet()).to_pylist()
    out = []
    for i, row in enumerate(rows):
        rid = row["id"]
        clean, _ = decode(row["clean"])
        noisy, _ = decode(row["noisy"])
        n = min(len(clean), len(noisy))
        clean, noisy = clean[:n], noisy[:n]
        sf.write(f"{NOISY_DIR}/{rid}.wav", noisy.astype(np.float32), SR)  # for NISQA
        rec = {"id": rid}
        try:
            rec["pesq"] = pesq(SR, clean.astype(np.float32), noisy.astype(np.float32), "wb")
        except Exception:
            rec["pesq"] = np.nan
        try:
            rec["stoi"] = stoi(clean, noisy, SR, extended=False)
        except Exception:
            rec["stoi"] = np.nan
        rec["si_sdr"] = si_sdr(clean, noisy)
        rec["seg_snr"] = seg_snr(clean, noisy)
        try:
            d = dnsmos.run(noisy.astype(np.float32), sr=SR)
            rec["dnsmos_ovrl"] = d["ovrl_mos"]
            rec["dnsmos_sig"] = d["sig_mos"]
            rec["dnsmos_bak"] = d["bak_mos"]
            rec["dnsmos_p808"] = d["p808_mos"]
        except Exception:
            rec.update(dnsmos_ovrl=np.nan, dnsmos_sig=np.nan, dnsmos_bak=np.nan, dnsmos_p808=np.nan)
        out.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  metrics {i+1}/{len(rows)}")
    df = pd.DataFrame(out)
    df.to_csv(METRICS_CSV, index=False)
    print(f"Wrote {METRICS_CSV}")
    return df


def compute_nisqa():
    """Run NISQA v2 over the extracted noisy wavs (predict_dir). Cached in NISQA_CSV."""
    if os.path.exists(NISQA_CSV):
        print(f"Loading cached {NISQA_CSV}")
        return pd.read_csv(NISQA_CSV)
    outdir = f"{RESULTS}/nisqa"
    os.makedirs(outdir, exist_ok=True)
    print("Running NISQA over noisy files ...")
    subprocess.run(
        [PY, "experiments/NISQA/run_predict.py", "--mode", "predict_dir",
         "--pretrained_model", "experiments/NISQA/weights/nisqa.tar",
         "--data_dir", NOISY_DIR, "--num_workers", "0", "--bs", "20", "--output_dir", outdir],
        check=True, capture_output=True, text=True)
    csv = sorted(glob.glob(f"{outdir}/*.csv"), key=os.path.getmtime)[-1]
    nis = pd.read_csv(csv)
    nis["id"] = nis["deg"].str.replace(".wav", "", regex=False)
    nis = nis.rename(columns={"mos_pred": "nisqa_mos", "noi_pred": "nisqa_noi",
                              "dis_pred": "nisqa_dis", "col_pred": "nisqa_col", "loud_pred": "nisqa_loud"})
    nis = nis[["id", "nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud"]]
    nis.to_csv(NISQA_CSV, index=False)
    print(f"Wrote {NISQA_CSV}")
    return nis


def load_mos():
    rows = [json.loads(l) for l in open(f"{RESULTS}/voicebank_sqa.jsonl")]
    return pd.DataFrame([
        {"id": r["id"], "mos": r["mos"], "snr_db": r["snr_db"]}
        for r in rows if r["kind"] == "noisy"
    ])


def corr(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 5:
        return None
    return pearsonr(x[m], y[m])[0], spearmanr(x[m], y[m])[0], int(m.sum())


def main():
    df = load_mos().merge(compute_metrics(), on="id", how="inner").merge(compute_nisqa(), on="id", how="inner")
    df = df[df["mos"].notna()].copy()

    L = ["# SALMONN MOS vs objective & neural metrics (VoiceBank-DEMAND-16k, noisy)\n"]
    L.append(f"- pairs with all metrics + parsed SALMONN MOS: {len(df)} of 824\n")

    metrics = [
        ("DNSMOS OVRL", "dnsmos_ovrl"), ("DNSMOS P.808", "dnsmos_p808"),
        ("DNSMOS SIG", "dnsmos_sig"), ("DNSMOS BAK", "dnsmos_bak"),
        ("NISQA MOS", "nisqa_mos"), ("NISQA noisiness", "nisqa_noi"),
        ("NISQA discontinuity", "nisqa_dis"), ("NISQA coloration", "nisqa_col"),
        ("PESQ (wb)", "pesq"), ("STOI", "stoi"), ("SI-SDR", "si_sdr"),
        ("segmental SNR", "seg_snr"), ("global SNR", "snr_db"),
    ]
    L.append("## Correlation of SALMONN MOS with each metric\n")
    L.append("| metric | Pearson r | Spearman ρ | n |")
    L.append("|---|---|---|---|")
    for name, col in metrics:
        c = corr(df[col], df["mos"])
        L.append(f"| {name} | {c[0]:+.3f} | {c[1]:+.3f} | {c[2]} |" if c else f"| {name} | n/a | n/a | 0 |")
    L.append("")

    # cross-correlation among the MOS predictors (Spearman) for context
    preds = [("SALMONN", "mos"), ("DNSMOS", "dnsmos_ovrl"), ("NISQA", "nisqa_mos"), ("PESQ", "pesq")]
    L.append("## Cross-agreement among MOS predictors (Spearman ρ)\n")
    L.append("| | " + " | ".join(n for n, _ in preds) + " |")
    L.append("|" + "---|" * (len(preds) + 1))
    for n1, c1 in preds:
        cells = []
        for n2, c2 in preds:
            c = corr(df[c1], df[c2])
            cells.append(f"{c[1]:+.2f}" if c else "-")
        L.append(f"| **{n1}** | " + " | ".join(cells) + " |")
    L.append("")
    L.append("> Reads: how well each pair of *quality predictors* rank the 824 files the same way. "
             "SALMONN's row shows whether it agrees with purpose-built MOS predictors as much as they agree with each other.\n")

    # scale usage
    L.append("## Scale usage (mean ± std)\n")
    for name, col in [("SALMONN MOS", "mos"), ("DNSMOS OVRL", "dnsmos_ovrl"), ("NISQA MOS", "nisqa_mos"), ("PESQ", "pesq")]:
        L.append(f"- {name}: {df[col].mean():.2f} ± {df[col].std():.2f}  (range {df[col].min():.2f}–{df[col].max():.2f})")
    L.append(f"- SALMONN distinct values: {sorted(df.mos.unique())}\n")

    with open(f"{RESULTS}/CALIBRATION.md", "w") as f:
        f.write("\n".join(L))
    print(f"Wrote {RESULTS}/CALIBRATION.md")
    print("\n".join(L))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, col, title in zip(ax, ["dnsmos_ovrl", "nisqa_mos", "pesq"], ["DNSMOS OVRL", "NISQA MOS", "PESQ"]):
        s = df.dropna(subset=[col, "mos"])
        jit = s["mos"] + np.random.uniform(-0.06, 0.06, len(s))
        a.scatter(s[col], jit, s=8, alpha=0.3)
        a.set_xlabel(title); a.set_ylabel("SALMONN MOS")
        a.set_title(f"{title}  (ρ={spearmanr(s[col], s['mos'])[0]:+.2f})")
    fig.tight_layout(); fig.savefig(f"{RESULTS}/mos_vs_neural.png", dpi=110); plt.close(fig)
    print(f"Wrote {RESULTS}/mos_vs_neural.png")


if __name__ == "__main__":
    main()
