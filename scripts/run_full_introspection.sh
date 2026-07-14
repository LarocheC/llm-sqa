#!/usr/bin/env bash
#
# Run the full embedding-introspection harness on a single audio file and log
# activations, decoded embeddings and visualizations to MLflow.
#
#   scripts/run_full_introspection.sh [AUDIO_FILE] [DEVICE]
#
set -euo pipefail
cd "$(dirname "$0")/.."

AUDIO_FILE="${1:-test_audio_samples/003_noisy.wav}"
DEVICE="${2:-cuda:0}"

if [[ ! -f "$AUDIO_FILE" ]]; then
  echo "Audio file not found: $AUDIO_FILE"
  echo "Pass a path: scripts/run_full_introspection.sh /path/to/audio.wav"
  exit 1
fi

python introspect_one.py "$AUDIO_FILE" \
    --config salmonn_sqa/inference_config.yaml \
    --device "$DEVICE"

echo "View results: mlflow ui"
