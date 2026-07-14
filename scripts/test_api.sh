#!/usr/bin/env bash
#
# Smoke-test a running SQA API: hits /health, /, and /assess.
#
#   scripts/test_api.sh [BASE_URL] [AUDIO_FILE]
#
set -euo pipefail
cd "$(dirname "$0")/.."

URL="${1:-http://localhost:8000}"
AUDIO_FILE="${2:-test_audio_samples/003_noisy.wav}"

echo "== GET /health =="; curl -s "$URL/health"; echo
echo "== GET / =="; curl -s "$URL/"; echo

if [[ -f "$AUDIO_FILE" ]]; then
  echo "== POST /assess ($AUDIO_FILE) =="
  curl -s -X POST "$URL/assess" -F "file=@$AUDIO_FILE"; echo
else
  echo "No audio at $AUDIO_FILE — skipping /assess"
fi
