#!/usr/bin/env bash
#
# Fetch the SALMONN model code + weights that this repo deliberately does NOT
# commit. Clones bytedance/SALMONN into salmonn_sqa/SALMONN, applies our local
# patches, and downloads the model assets. Safe to re-run (idempotent).
#
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root
ROOT="$PWD"

SALMONN_DIR="$ROOT/salmonn_sqa/SALMONN"
MODELS_DIR="$ROOT/salmonn_sqa/models"
CKPT_DIR="$ROOT/salmonn_sqa/ckpt"
PATCH="$ROOT/salmonn_sqa/patches/local-salmonn.patch"

WHISPER_DIR="$MODELS_DIR/whisper-large-v2"
VICUNA_DIR="$MODELS_DIR/vicuna-7b-v1_5"
BEATS_PT="$MODELS_DIR/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
SQA_CKPT="$CKPT_DIR/finetuned_SALMONN_7B_2.pth"

mkdir -p "$MODELS_DIR" "$CKPT_DIR"

# ---- 1. Clone SALMONN code (inference 'salmonn' branch) ----
if [[ ! -d "$SALMONN_DIR/.git" ]]; then
  echo ">> Cloning bytedance/SALMONN ..."
  git clone https://github.com/bytedance/SALMONN.git "$SALMONN_DIR"
fi
git -C "$SALMONN_DIR" fetch --all --quiet
git -C "$SALMONN_DIR" checkout salmonn

# ---- 2. Apply our local patches (EOFError handler + dependency bumps) ----
if [[ -f "$PATCH" ]]; then
  if git -C "$SALMONN_DIR" apply --check "$PATCH" 2>/dev/null; then
    echo ">> Applying local SALMONN patch ..."
    git -C "$SALMONN_DIR" apply "$PATCH"
  else
    echo ">> Local SALMONN patch already applied (or conflicts) — skipping."
  fi
fi

# ---- 3. Download model assets ----
# SALMONN-7B SQA-finetuned LoRA checkpoint (~363 MB)
if [[ ! -f "$SQA_CKPT" ]]; then
  echo ">> Downloading finetuned SALMONN-7B SQA checkpoint ..."
  python - "$CKPT_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="tsinghua-ee/Speech_Quality_Assessment",
                filename="finetuned_SALMONN_7B_2.pth", local_dir=sys.argv[1])
PY
fi

# Whisper Large V2
if [[ ! -d "$WHISPER_DIR" ]]; then
  echo ">> Downloading openai/whisper-large-v2 ..."
  python - "$WHISPER_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("openai/whisper-large-v2", local_dir=sys.argv[1],
                  ignore_patterns=["*.msgpack", "*.h5"])
PY
fi

# Vicuna 7B v1.5 (requires HF access to lmsys/vicuna-7b-v1.5)
if [[ ! -d "$VICUNA_DIR" ]]; then
  echo ">> Downloading lmsys/vicuna-7b-v1.5 ..."
  python - "$VICUNA_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("lmsys/vicuna-7b-v1.5", local_dir=sys.argv[1])
PY
fi

# BEATs iter3+ (AS2M) cpt2 encoder — manual download (OneDrive)
if [[ ! -f "$BEATS_PT" ]]; then
  cat <<EOF

======================================================================
MANUAL DOWNLOAD REQUIRED: BEATs checkpoint
  1. Open: https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea
  2. Download 'BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
  3. Place it at: $BEATS_PT
  4. Re-run this script.
======================================================================
EOF
  exit 2
fi

# NISQA — the neural MOS predictor used to build the corpus targets and to calibrate
# the model. Previously an undocumented, manual clone; fetch it here so the research
# pipeline is actually reproducible.
NISQA_DIR="${SQA_NISQA_DIR:-$ROOT/experiments/NISQA}"
if [[ ! -f "$NISQA_DIR/weights/nisqa.tar" ]]; then
  echo ">> fetching NISQA -> $NISQA_DIR"
  if [[ ! -d "$NISQA_DIR/.git" ]]; then
    git clone --depth 1 https://github.com/gabrielmittag/NISQA.git "$NISQA_DIR"
  fi
  if [[ ! -f "$NISQA_DIR/weights/nisqa.tar" ]]; then
    echo "!! NISQA cloned but weights/nisqa.tar is missing (upstream ships it in-repo)."
    echo "   Check https://github.com/gabrielmittag/NISQA"
    exit 2
  fi
else
  echo ">> NISQA present ($NISQA_DIR)"
fi

echo ">> SALMONN setup complete. Model paths resolve via \$SQA_ROOT ($ROOT)."
echo ">> Next:  uv sync --extra experiments && uv run python scripts/check_env.py"
