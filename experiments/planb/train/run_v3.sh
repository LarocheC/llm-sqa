#!/usr/bin/env bash
# Plan B v3 — full pipeline: rebuild the corpus with synthetic degradation types,
# LLM-paraphrase the descriptions (Opus 4.8), then full Stage 1 + Stage 2 retrain.
# The synthetic types change the audio, so both stages are retrained from the
# released SQA checkpoint. ~3 hours end to end. Run detached (setsid nohup).
set -euo pipefail

# Paths come from the environment (see experiments/config.py + REPRODUCING.md); the
# defaults below match the documented layout. Override any SQA_* var to relocate.
export SQA_ROOT="${SQA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="$SQA_ROOT/.venv/bin/python"          # the single uv env (uv sync --extra experiments)
CFG=$SQA_ROOT/experiments/planb/train/sqa_finetune.yaml
RES=$SQA_ROOT/experiments/results/planb
MAN=$RES/manifests

DATA_ROOT="${SQA_DATA_ROOT:-$HOME/data}"
LIBRITTS="${SQA_LIBRITTS_ROOT:-$DATA_ROOT/libritts_r/LibriTTS_R}"
CORPUS_WAV="${SQA_CORPUS_WAV_DIR:-$DATA_ROOT/planb_corpus}"
CLEAN_TRAIN="$LIBRITTS/train-clean-100"
CLEAN_VAL="$LIBRITTS/dev-clean"
WAV_TRAIN="$CORPUS_WAV/wav_train_v3"
WAV_VAL="$CORPUS_WAV/wav_val_v3"
MODEL="${1:-claude-opus-4-8}"

[ -x "$PY" ] || { echo "ERROR: $PY not found. Run:  uv sync --extra experiments"; exit 1; }
[ -d "$CLEAN_TRAIN" ] || { echo "ERROR: LibriTTS-R not at $CLEAN_TRAIN (set SQA_LIBRITTS_ROOT)"; exit 1; }

cd "$SQA_ROOT"

echo "== [0/5] preflight =="
# The paraphrase pool cache ships with the repo, so a rerun over the same degradation
# taxonomy makes ZERO API calls. Only verify the key if one is actually set; step [2/5]
# will fail with a clear message if new profiles turn up and no key is available.
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  $PY - <<PYEOF
import anthropic
anthropic.Anthropic().messages.create(model="${MODEL}", max_tokens=4,
    messages=[{"role":"user","content":"ping"}])
print("  API key OK (${MODEL} reachable)")
PYEOF
else
  echo "  no ANTHROPIC_API_KEY — fine as long as the shipped paraphrase cache covers every profile"
fi

echo "== [1/5] regenerate corpus with synthetic degradation types =="
$PY -m experiments.planb.generate_corpus --clean-dir "$CLEAN_TRAIN" --n 4000 --seed 1 \
    --out "$RES/corpus_train_v3pre.jsonl" --wav-dir "$WAV_TRAIN"
$PY -m experiments.planb.generate_corpus --clean-dir "$CLEAN_VAL" --n 300 --seed 7 \
    --out "$RES/corpus_val_v3pre.jsonl" --wav-dir "$WAV_VAL"

echo "== [2/5] LLM-paraphrase descriptions ($MODEL, shared profile cache) =="
$PY -m experiments.planb.paraphrase --in "$RES/corpus_train_v3pre.jsonl" \
    --out "$RES/corpus_train_v3.jsonl" --cache "$RES/paraphrase_pool.json" --model "$MODEL"
$PY -m experiments.planb.paraphrase --in "$RES/corpus_val_v3pre.jsonl" \
    --out "$RES/corpus_val_v3.jsonl" --cache "$RES/paraphrase_pool.json" --model "$MODEL"

echo "== [3/5] build v3 manifests =="
for st in 1 2; do
  $PY -m experiments.planb.train.make_manifest --jsonl "$RES/corpus_train_v3.jsonl" --wav-dir "$WAV_TRAIN" --stage $st --out "$MAN/train_stage${st}_v3.json"
  $PY -m experiments.planb.train.make_manifest --jsonl "$RES/corpus_val_v3.jsonl"   --wav-dir "$WAV_VAL"   --stage $st --out "$MAN/val_stage${st}_v3.json"
done

cd "$SQA_ROOT/salmonn_sqa/SALMONN"   # train.py uses relative imports

echo "== [4/5] STAGE 1 (score block) from the released SQA checkpoint =="
$PY train.py --cfg-path "$CFG" --options \
  datasets.train_ann_path="$MAN/train_stage1_v3.json" \
  datasets.valid_ann_path="$MAN/val_stage1_v3.json" \
  datasets.test_ann_path="$MAN/val_stage1_v3.json" \
  run.output_dir="$RES/ckpt_stage1_v3" run.optims.max_epoch=2

S1=$(ls -t "$RES/ckpt_stage1_v3"/*/checkpoint_best.pth 2>/dev/null | head -1)
[ -z "$S1" ] && { echo "ERROR: no Stage 1 v3 checkpoint"; exit 1; }
echo "== Stage 1 v3 checkpoint: $S1 =="

echo "== [5/5] STAGE 2 (description + MOS) continuing from Stage 1 v3 =="
$PY train.py --cfg-path "$CFG" --options \
  model.ckpt="$S1" \
  datasets.train_ann_path="$MAN/train_stage2_v3.json" \
  datasets.valid_ann_path="$MAN/val_stage2_v3.json" \
  datasets.test_ann_path="$MAN/val_stage2_v3.json" \
  run.output_dir="$RES/ckpt_stage2_v3" run.optims.max_epoch=4

S2=$(ls -t "$RES/ckpt_stage2_v3"/*/checkpoint_best.pth 2>/dev/null | head -1)
echo "== DONE v3. Final ckpt: $S2 =="
echo "Eval: python -m experiments.planb.eval_compare --planb-ckpt $S2 --n-clips 6"
