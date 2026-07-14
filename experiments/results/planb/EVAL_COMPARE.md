# Plan B before/after — controlled degradation sweep

- 6 clean clips x 4 families x graded levels (held-out: synthetic degradations on VoiceBank-DEMAND clean; training used real RIRs/MUSAN/codec on LibriTTS — disjoint).
- orig prompt = default SQA (prose + JSON MOS); planb prompt = sqa_full (structured).

## rho(MOS, severity) — does overall MOS track the degradation?

| family | orig | Plan B |
|---|---|---|
| noise | -0.90 | -0.93 |
| lowpass | -0.37 | -0.81 |
| clip | -0.48 | -0.83 |
| reverb | -0.27 | -0.95 |

## rho(dimension score, severity) — Plan B per-dimension scores

| family -> dim | Plan B |
|---|---|
| noise -> noise | -0.82 |
| lowpass -> bandwidth | -0.88 |
| clip -> clipping | -0.85 |
| reverb -> reverberation | -0.95 |

## naming rate — does the natural-language *description* name the degradation?
(Plan B's score block is excluded so this measures prose, not the dimension labels.)

| family | orig | Plan B |
|---|---|---|
| noise | 97% | 100% |
| lowpass | 0% | 71% |
| clip | 79% | 75% |
| reverb | 8% | 100% |

## output robustness

- orig: 20/108 degenerate, 6/108 unparsed MOS
- Plan B: 0/108 degenerate, 0/108 unparsed MOS

## reading

- **MOS now ranks every axis**, including the former blind spots reverb and bandwidth (and clipping), while noise stays strong — the de-specialization worked.
- **Per-dimension scores track severity strongly** across all four axes: the calibration heads learned the right monotonic mapping (held-out, synthetic).
- **Descriptions are mixed**: reverb/bandwidth prose naming rises sharply, but noise/clip prose naming can fall vs the original — Plan B encodes degradation evidence mainly in the *scores* now; the free-text descriptions are terser and less reliable than the numbers. Strengthening Stage-2 description supervision is the natural next iteration.
