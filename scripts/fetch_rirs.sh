#!/usr/bin/env bash
# Fetch the room-impulse-response bank used for the reverberation axis.
#
# OpenSLR SLR28 (RIRS_NOISES) — Apache-2.0, 16 kHz, ~1.3 GB. Contains simulated RIRs
# (small/medium/large rooms) plus real RIRs from RWCP, the REVERB challenge and the
# Aachen AIR database.
#
# RT60 and DRR are MEASURED from each response at corpus-build time
# (experiments/planb/degradations.py: measure_rt60 / _drr_db), so no filename convention
# is assumed — point SQA_RIR_ROOT at any RIR corpus and the pipeline works.
set -euo pipefail

ROOT="${SQA_RIR_ROOT:-${SQA_DATA_ROOT:-$HOME/data}/rirs_open}"
URL="https://www.openslr.org/resources/28/rirs_noises.zip"

mkdir -p "$ROOT"
cd "$ROOT"

if [[ -d RIRS_NOISES/simulated_rirs ]]; then
  echo ">> RIRs already present at $ROOT/RIRS_NOISES"
else
  if [[ ! -f rirs_noises.zip ]]; then
    echo ">> downloading SLR28 (~1.3 GB) -> $ROOT"
    curl -L --progress-bar -o rirs_noises.zip "$URL"
  fi
  echo ">> unzipping"
  unzip -q rirs_noises.zip
fi

N_SIM=$(find RIRS_NOISES/simulated_rirs -name '*.wav' | wc -l)
N_REAL=$(find RIRS_NOISES/real_rirs_isotropic_noises -name '*.wav' ! -name '*noise*' | wc -l)
echo ">> ready: $N_SIM simulated + $N_REAL real RIRs"
echo ">> export SQA_RIR_ROOT=$ROOT"
