#!/usr/bin/env bash
# Plan B two-stage LoRA-SFT for SALMONN-SQA.
#   Stage 1 (calibration): learn the per-dimension score block  (task sqa_score)
#   Stage 2 (reasoning):   add the grounded description + Overall MOS, continuing
#                          from the Stage 1 checkpoint            (task sqa_full)
# Encoders stay frozen; Q-Former + projection + Vicuna-LoRA train.
#
# Usage:
#   bash run_finetune.sh            # full two-stage run
#   bash run_finetune.sh --smoke    # 4 iters x 1 epoch each, to prove it trains
set -euo pipefail

export SQA_ROOT="${SQA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="$SQA_ROOT/.venv/bin/python"          # the single uv env (uv sync --extra experiments)
CFG=$SQA_ROOT/experiments/planb/train/sqa_finetune.yaml
MAN=$SQA_ROOT/experiments/results/planb/manifests
RES=$SQA_ROOT/experiments/results/planb
CORPUS=$RES/corpus_train.jsonl
VAL=$RES/corpus_val.jsonl
CORPUS_WAV="${SQA_CORPUS_WAV_DIR:-${SQA_DATA_ROOT:-$HOME/data}/planb_corpus}"
WAV_TRAIN="$CORPUS_WAV/wav_train"
WAV_VAL="$CORPUS_WAV/wav_val"
SMOKE="${1:-}"

echo "== building manifests =="
for st in 1 2; do
  $PY -m experiments.planb.train.make_manifest --jsonl "$CORPUS" --wav-dir "$WAV_TRAIN" --stage $st --out "$MAN/train_stage$st.json"
  $PY -m experiments.planb.train.make_manifest --jsonl "$VAL"    --wav-dir "$WAV_VAL"   --stage $st --out "$MAN/val_stage$st.json"
done

SMOKE_OPTS=""
if [ "$SMOKE" = "--smoke" ]; then
  SMOKE_OPTS="run.epoch_based=False run.iters_per_epoch=4 run.optims.max_epoch=1 run.optims.warmup_steps=1 run.log_freq=1 run.accum_grad_iters=1"
  echo "== SMOKE mode: 4 iters x 1 epoch per stage =="
fi

cd "$SQA_ROOT/salmonn_sqa/SALMONN"   # train.py uses relative imports

echo "== STAGE 1: calibration (score block) =="
$PY train.py --cfg-path "$CFG" --options \
  datasets.train_ann_path="$MAN/train_stage1.json" \
  datasets.valid_ann_path="$MAN/val_stage1.json" \
  datasets.test_ann_path="$MAN/val_stage1.json" \
  run.output_dir="$RES/ckpt_stage1" \
  run.optims.max_epoch=2 $SMOKE_OPTS

# the harness saves under output_dir/<job_timestamp>/checkpoint_*.pth — recurse,
# prefer checkpoint_best, fall back to the newest epoch checkpoint.
S1=$(ls -t "$RES/ckpt_stage1"/*/checkpoint_best.pth 2>/dev/null | head -1)
[ -z "$S1" ] && S1=$(ls -t "$RES/ckpt_stage1"/*/checkpoint_*.pth 2>/dev/null | head -1)
echo "== Stage 1 checkpoint: $S1 =="
[ -z "$S1" ] && { echo "ERROR: no Stage 1 checkpoint found; aborting before Stage 2"; exit 1; }

echo "== STAGE 2: reasoning (description + MOS), continuing from Stage 1 =="
$PY train.py --cfg-path "$CFG" --options \
  model.ckpt="$S1" \
  datasets.train_ann_path="$MAN/train_stage2.json" \
  datasets.valid_ann_path="$MAN/val_stage2.json" \
  datasets.test_ann_path="$MAN/val_stage2.json" \
  run.output_dir="$RES/ckpt_stage2" \
  run.optims.max_epoch=4 $SMOKE_OPTS

S2=$(ls -t "$RES/ckpt_stage2"/*/checkpoint_best.pth 2>/dev/null | head -1)
[ -z "$S2" ] && S2=$(ls -t "$RES/ckpt_stage2"/*/checkpoint_*.pth 2>/dev/null | head -1)
echo "== DONE. Final Plan B checkpoint: $S2 =="
echo "Evaluate with: point salmonn_core / inference_config model.ckpt at $S2 and rerun the eval suite."
