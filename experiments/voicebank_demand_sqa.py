"""
Run SALMONN descriptive SQA over the VoiceBank-DEMAND-16k test split and record,
for each file, the per-pair SNR and the model's natural-language description.

The dataset (JacobLinCool/VoiceBank-DEMAND-16k) stores 824 test rows as
{id, clean{bytes}, noisy{bytes}} where noisy = clean + noise (byte-aligned), so
SNR is computed directly: 10*log10(sum(clean^2) / sum((noisy-clean)^2)).

Results are appended to a JSONL (one line per inference) so a long run is durable
across interruption. Run the noisy files (vary in SNR) and, with --include-clean,
the matching clean files (for a clean-vs-noisy contrast).
"""

import argparse
import glob
import io
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import salmonn_core  # noqa: E402
from experiments import config as cfg  # noqa: E402


def find_test_parquet() -> str:
    return str(cfg.voicebank_parquet())


def decode(audio_struct) -> tuple[np.ndarray, int]:
    """Decode a HF audio struct {bytes, path} to a mono float array + sample rate."""
    audio, sr = sf.read(io.BytesIO(audio_struct["bytes"]))
    if audio.ndim == 2:
        audio = audio[:, 0]
    return audio.astype(np.float64), sr


def compute_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """SNR in dB from an aligned clean/noisy pair. noise = noisy - clean."""
    n = min(len(clean), len(noisy))
    clean, noisy = clean[:n], noisy[:n]
    noise = noisy - clean
    sig_p = float(np.sum(clean**2))
    noise_p = float(np.sum(noise**2))
    if noise_p <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(sig_p / noise_p)


def run(args):
    torch.manual_seed(args.seed)
    parquet_path = args.parquet or find_test_parquet()
    print(f"Reading {parquet_path}")
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if args.max_files:
        rows = rows[: args.max_files]
    print(f"{len(rows)} rows; loading SALMONN ...")

    sqa = salmonn_core.load_model(device_name=args.device)
    print("Model loaded. Running inference ...")

    # Resume support: skip (id, kind) already in the output file.
    done = set()
    if os.path.exists(args.out) and not args.overwrite:
        with open(args.out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["id"], r["kind"]))
                except Exception:
                    pass
        print(f"Resuming — {len(done)} results already present")

    kinds = ["noisy"] + (["clean"] if args.include_clean else [])
    out = open(args.out, "w" if args.overwrite else "a")
    t0 = time.time()
    n_done = 0

    for i, row in enumerate(rows):
        rid = row["id"]
        try:
            clean_a, sr = decode(row["clean"])
            noisy_a, _ = decode(row["noisy"])
            snr_db = compute_snr_db(clean_a, noisy_a)
        except Exception as e:
            print(f"[{rid}] decode failed: {e}")
            continue

        for kind in kinds:
            if (rid, kind) in done:
                continue
            audio = noisy_a if kind == "noisy" else clean_a
            samples = salmonn_core.prepare_audio_sample(
                io_to_tmp(audio, sr), sqa.wav_processor, device=sqa.device
            )
            ts = time.time()
            try:
                raw = salmonn_core.generate_sqa(sqa, samples=samples)
            except Exception as e:
                print(f"[{rid}/{kind}] inference failed: {e}")
                continue
            desc = salmonn_core.clean_output(raw)
            rec = {
                "id": rid,
                "kind": kind,
                "snr_db": None if kind == "clean" else round(snr_db, 3),
                "description": desc,
                "mos": salmonn_core.extract_mos(raw),
                "degenerate": salmonn_core.is_degenerate(raw),
                "n_words": len(desc.split()),
                "gen_time": round(time.time() - ts, 2),
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n_done += 1

        if (i + 1) % 10 == 0:
            rate = n_done / max(time.time() - t0, 1e-6)
            print(f"  {i+1}/{len(rows)} rows | {n_done} inferences | {rate:.2f}/s")

    out.close()
    print(f"Done: {n_done} inferences in {time.time()-t0:.0f}s -> {args.out}")


_TMP = str(cfg.work_dir("scratch") / "_vb_sqa_tmp.wav")


def io_to_tmp(audio: np.ndarray, sr: int) -> str:
    """prepare_audio_sample wants a path; write the decoded array to a temp wav."""
    sf.write(_TMP, audio.astype(np.float32), sr)
    return _TMP


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/results/voicebank_sqa.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-files", type=int, default=None, help="limit number of rows")
    ap.add_argument("--include-clean", action="store_true", help="also assess clean files")
    ap.add_argument("--parquet", default=None, help="override parquet path")
    ap.add_argument("--overwrite", action="store_true", help="ignore existing output")
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())
