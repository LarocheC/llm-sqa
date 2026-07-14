"""
Experiment #5 — does SALMONN's descriptive SQA track speech enhancement?

For a sample of VoiceBank-DEMAND test pairs, enhance the noisy file with Clement's
ConvFSENet (claroche1/sparse-nsnet2) and run SALMONN + DNSMOS + NISQA + PESQ on the
noisy / enhanced / clean triplet. Questions:
  * does SALMONN's MOS rise noisy -> enhanced (does it notice the improvement)?
  * does its MOS gain track the objective gain (PESQ/DNSMOS/NISQA)?
  * does it flag enhancement *artifacts* (over-suppression, muffling, musical noise)
    that distinguish enhanced from clean?
"""

import argparse
import glob
import io
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from pesq import pesq
from speechmos import dnsmos

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import config as cfg  # noqa: E402
import salmonn_core  # noqa: E402
from experiments.enhance_convfsenet import ConvFSENet  # noqa: E402

SR = 16000
RESULTS = str(cfg.RESULTS_DIR)
ENH_DIR = str(cfg.work_dir("vb_enh"))


def decode(struct):
    a, _ = sf.read(io.BytesIO(struct["bytes"]))
    return (a[:, 0] if a.ndim == 2 else a).astype(np.float64)


def run(args):
    os.makedirs(ENH_DIR, exist_ok=True)
    rows = pq.read_table(str(cfg.voicebank_parquet())).to_pylist()
    by_spk = {}
    for r in rows:
        by_spk.setdefault(r["id"].split("_")[0], []).append(r)
    sample = []
    for spk, rs in sorted(by_spk.items()):
        sample += rs[: args.per_speaker]
    print(f"{len(sample)} utterances")

    sqa = salmonn_core.load_model(device_name=args.device)
    enh = ConvFSENet()
    out = open(args.out, "w")
    n = 0
    for row in sample:
        rid = row["id"]
        clean = decode(row["clean"])
        noisy = decode(row["noisy"])
        enhanced = enh.enhance(noisy)
        L = min(len(clean), len(noisy), len(enhanced))
        ref = clean[:L].astype(np.float32)
        for kind, audio in [("clean", clean), ("noisy", noisy), ("enhanced", enhanced)]:
            wavp = f"{ENH_DIR}/{rid}__{kind}.wav"
            a = np.clip(audio[:L], -1, 1).astype(np.float32)
            sf.write(wavp, a, SR)
            raw = salmonn_core.generate_sqa(
                sqa, samples=salmonn_core.prepare_audio_sample(wavp, sqa.wav_processor, device=sqa.device))
            desc = salmonn_core.clean_output(raw)
            try:
                pq_s = float(pesq(SR, ref, a, "wb"))
            except Exception:
                pq_s = None
            try:
                d = dnsmos.run(a, sr=SR)
            except Exception:
                d = {}
            out.write(json.dumps({
                "id": rid, "kind": kind, "salmonn_mos": salmonn_core.extract_mos(raw),
                "description": desc, "n_words": len(desc.split()),
                "degenerate": salmonn_core.is_degenerate(raw),
                "pesq": pq_s, "dnsmos_ovrl": float(d.get("ovrl_mos")) if d else None,
                "dnsmos_p808": float(d.get("p808_mos")) if d else None,
                "wav": os.path.basename(wavp),
            }) + "\n")
            out.flush()
            n += 1
        print(f"  {rid} done")
    out.close()
    print(f"Generated {n} clips -> {args.out}")

    # NISQA over all clips
    outdir = f"{RESULTS}/nisqa_enh"
    os.makedirs(outdir, exist_ok=True)
    print("Running NISQA ...")
    subprocess.run(
        [sys.executable, "experiments/NISQA/run_predict.py", "--mode", "predict_dir",
         "--pretrained_model", "experiments/NISQA/weights/nisqa.tar", "--data_dir", ENH_DIR,
         "--num_workers", "0", "--bs", "20", "--output_dir", outdir],
        check=True, capture_output=True, text=True)
    import pandas as pd
    csv = sorted(glob.glob(f"{outdir}/*.csv"), key=os.path.getmtime)[-1]
    nis = pd.read_csv(csv).set_index("deg")["mos_pred"].to_dict()
    recs = [json.loads(l) for l in open(args.out)]
    for r in recs:
        r["nisqa_mos"] = nis.get(r["wav"])
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"Merged NISQA into {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{RESULTS}/enhancement.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-speaker", type=int, default=24)
    run(ap.parse_args())
