#!/usr/bin/env bash
# The openly reproducible build: every input is public.
#
#   clean speech  LibriTTS-R          (CC BY 4.0, openslr.org/141)
#   noise         MUSAN               (CC BY 4.0, openslr.org/17)
#   reverb        OpenSLR SLR28 RIRs  (Apache-2.0, scripts/fetch_rirs.sh)
#   descriptions  paraphrase pool cache (ships with the repo — no API key needed)
#
# Five steps, ~4 h on a single 24 GB GPU:
#   corpus -> paraphrase -> manifests -> Stage 1 (calibration) -> Stage 2 (reasoning)
#
# RT60/DRR are measured from each impulse response, so this works with ANY RIR corpus:
# just point SQA_RIR_ROOT elsewhere.
set -euo pipefail

export SQA_ROOT="${SQA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="$SQA_ROOT/.venv/bin/python"
CFG=$SQA_ROOT/experiments/planb/train/sqa_finetune.yaml
RES=$SQA_ROOT/experiments/results/planb
MAN=$RES/manifests

DATA_ROOT="${SQA_DATA_ROOT:-$HOME/data}"
LIBRITTS="${SQA_LIBRITTS_ROOT:-$DATA_ROOT/libritts_r/LibriTTS_R}"
WAV="${SQA_CORPUS_WAV_DIR:-$DATA_ROOT/planb_corpus}"

[ -x "$PY" ] || { echo "ERROR: $PY missing. Run:  uv sync --extra experiments"; exit 1; }
[ -d "$LIBRITTS/train-clean-100" ] || { echo "ERROR: LibriTTS-R not found (set SQA_LIBRITTS_ROOT)"; exit 1; }

cd "$SQA_ROOT"

echo "== [1/5] generate corpus (known degradation parameters) =="
$PY -m experiments.planb.generate_corpus --clean-dir "$LIBRITTS/train-clean-100" \
    --n 4000 --seed 1 --out "$RES/corpus_train_open_pre.jsonl" --wav-dir "$WAV/wav_train_open"
$PY -m experiments.planb.generate_corpus --clean-dir "$LIBRITTS/dev-clean" \
    --n 300 --seed 7 --out "$RES/corpus_val_open_pre.jsonl" --wav-dir "$WAV/wav_val_open"

echo "== [2/5] paraphrase descriptions =="
# The pool cache ships with the repo and covers the degradation taxonomy, so this is a
# pure cache hit and makes ZERO API calls. A key is only needed if new profiles appear.
$PY -m experiments.planb.paraphrase --in "$RES/corpus_train_open_pre.jsonl" \
    --out "$RES/corpus_train_open.jsonl" --cache "$RES/paraphrase_pool.json"
$PY -m experiments.planb.paraphrase --in "$RES/corpus_val_open_pre.jsonl" \
    --out "$RES/corpus_val_open.jsonl" --cache "$RES/paraphrase_pool.json"

echo "== [3/5] manifests =="
for st in 1 2; do
  $PY -m experiments.planb.train.make_manifest --jsonl "$RES/corpus_train_open.jsonl" \
      --wav-dir "$WAV/wav_train_open" --stage $st --out "$MAN/train_stage${st}_open.json"
  $PY -m experiments.planb.train.make_manifest --jsonl "$RES/corpus_val_open.jsonl" \
      --wav-dir "$WAV/wav_val_open"   --stage $st --out "$MAN/val_stage${st}_open.json"
done

cd "$SQA_ROOT/salmonn_sqa/SALMONN"   # train.py uses relative imports

echo "== [4/5] STAGE 1 — calibration: target = the per-dimension score block only =="
$PY train.py --cfg-path "$CFG" --options \
  datasets.train_ann_path="$MAN/train_stage1_open.json" \
  datasets.valid_ann_path="$MAN/val_stage1_open.json" \
  datasets.test_ann_path="$MAN/val_stage1_open.json" \
  run.output_dir="$RES/ckpt_stage1_open" run.optims.max_epoch=2

S1=$(ls -t "$RES/ckpt_stage1_open"/*/checkpoint_best.pth 2>/dev/null | head -1)
[ -z "$S1" ] && { echo "ERROR: no Stage 1 checkpoint"; exit 1; }

echo "== [5/5] STAGE 2 — reasoning: + description + Overall MOS, from Stage 1 =="
$PY train.py --cfg-path "$CFG" --options \
  model.ckpt="$S1" \
  datasets.train_ann_path="$MAN/train_stage2_open.json" \
  datasets.valid_ann_path="$MAN/val_stage2_open.json" \
  datasets.test_ann_path="$MAN/val_stage2_open.json" \
  run.output_dir="$RES/ckpt_stage2_open" run.optims.max_epoch=4

S2=$(ls -t "$RES/ckpt_stage2_open"/*/checkpoint_best.pth 2>/dev/null | head -1)
echo "== DONE. Final checkpoint: $S2 =="
