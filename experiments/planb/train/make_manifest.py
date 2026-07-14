"""
Convert a Plan B corpus JSONL into the SALMONN training manifest format
({"annotation": [{"path", "text", "task"}, ...]}) for one of the two SFT stages.

Stage 1 (calibration): target = the per-dimension score block only; task = sqa_score.
Stage 2 (reasoning):    target = the full score block + description + Overall MOS;
                        task = sqa_full.

The task name selects which prompt list the harness samples from (see
prompts_planb.json), keeping prompt and target consistent within a stage.

Usage:
  python -m experiments.planb.train.make_manifest \
      --jsonl experiments/results/planb/corpus_train.jsonl \
      --wav-dir "$SQA_CORPUS_WAV_DIR/wav_train_v3" \
      --stage 1 --out experiments/results/planb/manifests/train_stage1.json
"""

import argparse
import json
import os


def build(jsonl, wav_dir, stage):
    ann, missing = [], 0
    for line in open(jsonl):
        r = json.loads(line)
        path = os.path.abspath(os.path.join(wav_dir, r["wav"]))
        if not os.path.exists(path):
            missing += 1
            continue
        if stage == 1:
            text, task = r["target_text"].split("\n")[0], "sqa_score"
        else:
            text, task = r["target_text"], "sqa_full"
        ann.append({"path": path, "text": text, "task": task})
    return ann, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ann, missing = build(args.jsonl, args.wav_dir, args.stage)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"annotation": ann}, open(args.out, "w"), indent=1)
    print(f"stage {args.stage}: wrote {len(ann)} samples -> {args.out}"
          + (f"  ({missing} skipped: wav missing)" if missing else ""))
    if ann:
        ex = ann[0]
        print(f"  task={ex['task']}  path={ex['path']}")
        print(f"  text={ex['text']!r}")


if __name__ == "__main__":
    main()
