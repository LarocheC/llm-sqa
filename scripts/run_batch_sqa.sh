#!/usr/bin/env bash
#
# Batch descriptive SQA over a directory of audio files, logged to one MLflow
# run. Audio is resampled to 16 kHz automatically, so any WAV directory works.
#
#   scripts/run_batch_sqa.sh [AUDIO_DIR] [DEVICE]
#
set -euo pipefail
cd "$(dirname "$0")/.."

AUDIO_DIR="${1:-noisy_testset_wav}"
DEVICE="${2:-cuda:0}"

echo "Batch SQA: dir=$AUDIO_DIR device=$DEVICE"
uv run python batch_process_sqa.py "$AUDIO_DIR" \
    --device "$DEVICE" \
    --experiment "SALMONN_Batch_SQA"

echo "View results: mlflow ui"
