"""Issue #4 re-run with v3: does its MOS track the ConvFSENet enhancement gain?
Reuses the enhanced/noisy audio written by enhancement_experiment.py plus the cached
PESQ in enhancement.jsonl; only the v3 MOS is new. Original: noisy->enhanced MOS +0.03,
rho(MOS gain, PESQ gain) ~ 0.00.

  --infer    run v3 on the audio (needs a GPU + the enhanced WAVs)
  (default)  analyze from the committed JSONL — no GPU, no audio needed
"""
import argparse, json, os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments import config as cfg  # noqa: E402

# salmonn_core / eval_compare pull in the vendored SALMONN package, which a fresh clone
# does not have. Imported lazily under --infer so the analysis path needs no model/GPU.

RES = str(cfg.RESULTS_DIR)
# The enhanced audio used to live in /tmp/vb_enh: written by enhancement_experiment.py
# and read back here. A /tmp clean silently produced an empty/stale eval instead of a
# clean failure, so it now lives under WORK_DIR and its absence is a hard error.
WAVD = cfg.WORK_DIR / "vb_enh"
OUT = f"{RES}/planb/v3_enhancement.jsonl"

ap = argparse.ArgumentParser()
ap.add_argument("--infer", action="store_true")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

base = [json.loads(l) for l in open(f"{RES}/enhancement.jsonl")]

if args.infer:
    import salmonn_core as sc
    from experiments.planb.eval_compare import load_with_ckpt

    cfg.require(
        WAVD, "the enhanced/noisy VoiceBank WAVs for the enhancement eval",
        "regenerate them with:  uv run python -m experiments.enhancement_experiment\n"
        "                 (needs the ConvFSENet ONNX; set SQA_CONVFSENET_ONNX)",
    )
    ckpt = cfg.ckpt_v3(stage=2)
    prompt = json.load(open("experiments/planb/train/test_prompt_planb.json"))["sqa_full"]
    sqa = load_with_ckpt(str(ckpt), args.device)
    with open(OUT, "w") as fh:
        for i, r in enumerate(base):
            p = str(WAVD / r["wav"])
            raw = sc.generate_sqa(sqa, prompt=prompt, wav_path=p)
            fh.write(json.dumps({"id": r["id"], "kind": r["kind"], "mos_v3": sc.extract_mos(raw),
                                 "pesq": r["pesq"], "mos_orig": r["salmonn_mos"]}) + "\n"); fh.flush()
            if (i + 1) % 48 == 0: print(f"  {i+1}/144", flush=True)
    print("infer done")

# analyze
rs = [json.loads(l) for l in open(OUT)]
by = {(r["id"], r["kind"]): r for r in rs}
ids = sorted({r["id"] for r in rs})
print(f"\n## Issue #4 enhancement — v3 vs original ({len(ids)} utts)")
print(f"{'condition':9} | {'orig MOS':>8} | {'v3 MOS':>8} | {'PESQ':>5}")
for k in ["noisy", "enhanced", "clean"]:
    sub = [r for r in rs if r["kind"] == k]
    mo = np.mean([r["mos_orig"] for r in sub if r["mos_orig"] is not None])
    mv = np.mean([r["mos_v3"] for r in sub if r["mos_v3"] is not None])
    pq = np.mean([r["pesq"] for r in sub if r["pesq"] is not None])
    print(f"{k:9} | {mo:>8.2f} | {mv:>8.2f} | {pq:>5.2f}")
# per-utt gains noisy->enhanced
gm_o, gm_v, gp = [], [], []
for i in ids:
    n, e = by.get((i, "noisy")), by.get((i, "enhanced"))
    if n and e and None not in (n["mos_v3"], e["mos_v3"], n["mos_orig"], e["mos_orig"], n["pesq"], e["pesq"]):
        gm_o.append(e["mos_orig"] - n["mos_orig"]); gm_v.append(e["mos_v3"] - n["mos_v3"]); gp.append(e["pesq"] - n["pesq"])
gm_o, gm_v, gp = map(np.array, (gm_o, gm_v, gp))
print(f"\nnoisy->enhanced: orig MOS gain {gm_o.mean():+.2f} (improved {np.mean(gm_o>0):.0%}), "
      f"v3 MOS gain {gm_v.mean():+.2f} (improved {np.mean(gm_v>0):.0%}); PESQ gain {gp.mean():+.2f}")
print(f"rho(MOS gain, PESQ gain): orig {spearmanr(gm_o,gp)[0]:+.2f}  v3 {spearmanr(gm_v,gp)[0]:+.2f}")
