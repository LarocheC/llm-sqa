#!/usr/bin/env bash
# Description-quality iteration: retrain ONLY Stage 2 on the v2 targets (richer,
# more diverse, all-degraded-axes descriptions; same scores + MOS), continuing from
# the existing Stage 1 checkpoint. Stage 1 (the score block) is unchanged, so we
# reuse it rather than retraining it.
set -euo pipefail

export SQA_ROOT="${SQA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="$SQA_ROOT/.venv/bin/python"          # the single uv env (uv sync --extra experiments)
CFG=$SQA_ROOT/experiments/planb/train/sqa_finetune.yaml
MAN=$SQA_ROOT/experiments/results/planb/manifests
RES=$SQA_ROOT/experiments/results/planb
CORPUS_WAV="${SQA_CORPUS_WAV_DIR:-${SQA_DATA_ROOT:-$HOME/data}/planb_corpus}"
WAV_TRAIN="$CORPUS_WAV/wav_train"
WAV_VAL="$CORPUS_WAV/wav_val"

echo "== building v2 Stage 2 manifests =="
$PY -m experiments.planb.train.make_manifest --jsonl "$RES/corpus_train_v2.jsonl" --wav-dir "$WAV_TRAIN" --stage 2 --out "$MAN/train_stage2_v2.json"
$PY -m experiments.planb.train.make_manifest --jsonl "$RES/corpus_val_v2.jsonl"   --wav-dir "$WAV_VAL"   --stage 2 --out "$MAN/val_stage2_v2.json"

S1=$(ls -t "$RES/ckpt_stage1"/*/checkpoint_best.pth 2>/dev/null | head -1)
[ -z "$S1" ] && { echo "ERROR: no Stage 1 checkpoint found"; exit 1; }
echo "== continuing from Stage 1 checkpoint: $S1 =="

cd "$SQA_ROOT/salmonn_sqa/SALMONN"
$PY train.py --cfg-path "$CFG" --options \
  model.ckpt="$S1" \
  datasets.train_ann_path="$MAN/train_stage2_v2.json" \
  datasets.valid_ann_path="$MAN/val_stage2_v2.json" \
  datasets.test_ann_path="$MAN/val_stage2_v2.json" \
  run.output_dir="$RES/ckpt_stage2_v2" \
  run.optims.max_epoch=4

S2=$(ls -t "$RES/ckpt_stage2_v2"/*/checkpoint_best.pth 2>/dev/null | head -1)
echo "== DONE v2. Final ckpt: $S2 =="
