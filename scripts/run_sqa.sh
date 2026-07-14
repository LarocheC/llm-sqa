#!/usr/bin/env bash
#
# Single-file descriptive SQA via the SALMONN interactive CLI. Feeds the audio
# path + the canonical SQA prompt (pulled from salmonn_core, the single source
# of truth) into the upstream cli_inference.py.
#
set -euo pipefail
cd "$(dirname "$0")/.."
export SQA_ROOT="$PWD"

AUDIO="${1:-}"
if [[ -z "$AUDIO" || ! -f "$AUDIO" ]]; then
  echo "Usage: $0 /path/to/audio.wav"
  exit 1
fi

# Ensure the model code + weights are present.
[[ -d salmonn_sqa/SALMONN ]] || scripts/setup_salmonn.sh

PROMPT="$(python -c 'from salmonn_core import DEFAULT_SQA_PROMPT; print(DEFAULT_SQA_PROMPT)')"

python salmonn_sqa/SALMONN/cli_inference.py --cfg-path salmonn_sqa/inference_config.yaml <<EOF
$AUDIO
$PROMPT
EOF
