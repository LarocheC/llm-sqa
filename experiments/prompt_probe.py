"""
Prompt-steering probe — can a TARGETED prompt recover the degradations SALMONN
misses under the open-ended SQA prompt (reverb, bandwidth, clipping)?

For each utterance + blind-spot degradation, run on the SAME audio:
  1. the open-ended SQA prompt (baseline)
  2. a targeted yes/no detection prompt for that degradation
and also run the targeted prompt on the CLEAN file (false-positive control —
does the prompt just yes-bias the model regardless of the audio?).
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import config as cfg  # noqa: E402
import salmonn_core  # noqa: E402
from experiments.degradation_sweep import add_white_noise, clip_frac, lowpass, reverb  # noqa: E402

RESULTS = str(cfg.RESULTS_DIR)
TMP = str(cfg.work_dir("scratch") / "_probe.wav")

WORST = {  # family -> (apply_fn, value)
    "reverb": (reverb, 1.2),
    "lowpass": (lowpass, 2000),
    "clip": (clip_frac, 0.1),
}
TARGETED = {
    "reverb": ("Listen carefully to the room acoustics. Does this recording contain "
               "reverberation or echo — does it sound recorded in a room, hall, or from a "
               "distance? Begin your answer with YES or NO, then explain what you hear."),
    "lowpass": ("Listen carefully to the frequency content. Does this recording sound "
                "muffled, dull, or band-limited, as if high frequencies are missing (like "
                "telephone speech)? Begin your answer with YES or NO, then explain."),
    "clip": ("Listen carefully for distortion. Does this recording contain clipping or "
             "harsh, buzzy distortion? Begin your answer with YES or NO, then explain."),
}
RECOG = {
    "reverb": ["echo", "reverber", "reverb", "room", "hall", "distant", "cavern", "hollow"],
    "lowpass": ["muffl", "telephone", "dull", "bandwidth", "high frequenc", "high-frequenc",
                "lacks", "lacking", "tinny", "narrow", "filtered", "thin"],
    "clip": ["distort", "clipping", "clipped", "harsh", "crackl", "artifact", "robotic",
             "mechanical", "buzz", "rough", "grating"],
}


def mentions(text, fam):
    t = text.lower()
    return any(k in t for k in RECOG[fam])


def affirms(text):
    """True if the response affirms the degradation is present."""
    t = text.strip().lower()
    m = re.match(r"[^a-z]*\b(yes|no)\b", t)
    if m:
        return m.group(1) == "yes"
    return "yes" in t[:60]  # no clean lead-in; affirm if 'yes' appears early


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
        for fam, (fn, val) in WORST.items():
            deg = fn(clean, val)
            # 1. open-ended on degraded
            d_open = infer(sqa, deg, salmonn_core.DEFAULT_SQA_PROMPT)
            # 2. targeted on degraded
            d_tgt = infer(sqa, deg, TARGETED[fam])
            # 3. targeted on clean (false-positive control)
            c_tgt = infer(sqa, clean, TARGETED[fam])
            for cond, kind, text in [("open", "degraded", d_open),
                                     ("targeted", "degraded", d_tgt),
                                     ("targeted", "clean", c_tgt)]:
                out.write(json.dumps({
                    "id": rid, "family": fam, "cond": cond, "audio": kind,
                    "mentions": mentions(text, fam), "affirms_yes": affirms(text),
                    "text": text[:400],
                }) + "\n")
            out.flush()
        print(f"  {rid} done")
    out.close()

    # summary
    recs = [json.loads(l) for l in open(args.out)]
    print("\n=== prompt-steering recovery (% detecting the degradation) ===")
    print(f"{'family':8s} | open(deg) mentions | targeted(deg) affirms | targeted(clean) affirms = false-pos")
    for fam in WORST:
        od = [r for r in recs if r["family"] == fam and r["cond"] == "open" and r["audio"] == "degraded"]
        td = [r for r in recs if r["family"] == fam and r["cond"] == "targeted" and r["audio"] == "degraded"]
        tc = [r for r in recs if r["family"] == fam and r["cond"] == "targeted" and r["audio"] == "clean"]
        om = 100 * np.mean([r["mentions"] for r in od]) if od else 0
        ta = 100 * np.mean([r["affirms_yes"] for r in td]) if td else 0
        fp = 100 * np.mean([r["affirms_yes"] for r in tc]) if tc else 0
        print(f"{fam:8s} | {om:17.0f}% | {ta:20.0f}% | {fp:.0f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{RESULTS}/prompt_probe.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-speaker", type=int, default=6)
    run(ap.parse_args())
