"""
Package the SFT corpus as a Hugging Face dataset (parquet + embedded audio).

Turns Tier 3 from "download ~54 GB of source corpora and regenerate" into "download
~0.6 GB and train". The audio is a derivative of LibriTTS-R (CC BY 4.0) + MUSAN
(CC BY 4.0) + OpenSLR SLR28 simulated RIRs (Apache-2.0) — all of which permit
redistribution of adaptations with attribution.

  uv run python -m experiments.planb.build_hf_dataset --out /tmp/hf_ds
  uv run python -m experiments.planb.build_hf_dataset --out /tmp/hf_ds --push <repo_id>
"""

import argparse
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments import config as cfg  # noqa: E402

SHARD_MB = 400

# The HF `datasets` schema, embedded in the parquet metadata so the Hub renders the audio
# player and `load_dataset` returns a decoded Audio column. (We can't use the `datasets`
# library itself — it pins an old pyarrow that conflicts with the rest of the stack.)
FEATURES = {
    "id": {"_type": "Value", "dtype": "string"},
    "audio": {"_type": "Audio", "sampling_rate": 16000},
    "source_utterance": {"_type": "Value", "dtype": "string"},
    "prompt": {"_type": "Value", "dtype": "string"},
    "target_text": {"_type": "Value", "dtype": "string"},
    "mos": {"_type": "Value", "dtype": "float32"},
    "mos_source": {"_type": "Value", "dtype": "string"},
    "score_noise": {"_type": "Value", "dtype": "int32"},
    "score_reverberation": {"_type": "Value", "dtype": "int32"},
    "score_bandwidth": {"_type": "Value", "dtype": "int32"},
    "score_clipping": {"_type": "Value", "dtype": "int32"},
    "score_discontinuity": {"_type": "Value", "dtype": "int32"},
    "score_loudness": {"_type": "Value", "dtype": "int32"},
    "degradation_params": {"_type": "Value", "dtype": "string"},
    "metrics": {"_type": "Value", "dtype": "string"},
}

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("source_utterance", pa.string()),
    ("prompt", pa.string()),
    ("target_text", pa.string()),
    ("mos", pa.float32()),
    ("mos_source", pa.string()),
    ("score_noise", pa.int32()),
    ("score_reverberation", pa.int32()),
    ("score_bandwidth", pa.int32()),
    ("score_clipping", pa.int32()),
    ("score_discontinuity", pa.int32()),
    ("score_loudness", pa.int32()),
    ("degradation_params", pa.string()),
    ("metrics", pa.string()),
]).with_metadata({b"huggingface": json.dumps({"info": {"features": FEATURES}}).encode()})


def rows_for(jsonl, wav_dir):
    for line in open(jsonl):
        r = json.loads(line)
        wav = os.path.join(wav_dir, r["wav"])
        if not os.path.exists(wav):
            continue
        s = r["scores"]
        yield {
            "id": r["id"],
            "audio": {"bytes": open(wav, "rb").read(), "path": r["wav"]},
            "source_utterance": r.get("clean_path", ""),
            "prompt": r["prompt"],
            "target_text": r["target_text"],
            "mos": float(r["mos"]),
            "mos_source": r["mos_source"],
            "score_noise": int(s["noise"]),
            "score_reverberation": int(s["reverberation"]),
            "score_bandwidth": int(s["bandwidth"]),
            "score_clipping": int(s["clipping"]),
            "score_discontinuity": int(s["discontinuity"]),
            "score_loudness": int(s["loudness"]),
            "degradation_params": json.dumps(r["params"]),
            "metrics": json.dumps(r.get("metrics", {})),
        }


def write_split(name, jsonl, wav_dir, out):
    os.makedirs(out, exist_ok=True)
    batch, shard, nbytes, total = [], 0, 0, 0
    paths = []

    def flush():
        nonlocal batch, shard, nbytes
        if not batch:
            return
        p = os.path.join(out, f"{name}-{shard:05d}.parquet")
        pq.write_table(pa.Table.from_pylist(batch, schema=SCHEMA), p, compression="zstd")
        paths.append(p)
        print(f"    {os.path.basename(p)}  {len(batch)} rows  {os.path.getsize(p)/1e6:.0f} MB")
        batch, nbytes = [], 0
        shard += 1

    for row in rows_for(jsonl, wav_dir):
        batch.append(row)
        nbytes += len(row["audio"]["bytes"])
        total += 1
        if nbytes > SHARD_MB * 1e6:
            flush()
    flush()
    print(f"  {name}: {total} rows -> {len(paths)} shard(s)")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--push", default=None, help="HF dataset repo id to upload to")
    args = ap.parse_args()

    wav = cfg.CORPUS_WAV_DIR
    splits = {
        "train": (cfg.PLANB_DIR / "corpus_train_open.jsonl", wav / "wav_train_open"),
        "validation": (cfg.PLANB_DIR / "corpus_val_open.jsonl", wav / "wav_val_open"),
    }
    data_dir = os.path.join(args.out, "data")
    for name, (jl, wd) in splits.items():
        cfg.require(jl, f"the {name} corpus", "run experiments/planb/train/run_open.sh")
        cfg.require(wd, f"the {name} audio", "run experiments/planb/train/run_open.sh")
        write_split(name, str(jl), str(wd), data_dir)

    if args.push:
        from huggingface_hub import HfApi, create_repo
        api = HfApi()
        url = create_repo(args.push, repo_type="dataset", private=False, exist_ok=True)
        print(f"\nrepo: {url}")
        api.upload_folder(folder_path=args.out, repo_id=args.push, repo_type="dataset")
        print("uploaded")


if __name__ == "__main__":
    main()
