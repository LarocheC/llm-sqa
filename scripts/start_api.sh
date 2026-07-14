#!/usr/bin/env bash
#
# Start the SALMONN SQA FastAPI server.
#
#   scripts/start_api.sh [HOST] [PORT] [DEVICE]
#
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -d salmonn_sqa/models || ! -d salmonn_sqa/ckpt ]]; then
  echo "ERROR: model weights not found."
  echo "Run: scripts/setup_salmonn.sh"
  exit 1
fi

HOST="${1:-0.0.0.0}"
PORT="${2:-8000}"
DEVICE="${3:-cuda:0}"

echo "Starting API on http://$HOST:$PORT  (docs at /docs)"
python3 api_inference.py \
  --host "$HOST" --port "$PORT" --device "$DEVICE" \
  --config salmonn_sqa/inference_config.yaml
