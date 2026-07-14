# Plan B — Stage 1: synthetic degradation + target generator

Builds the parameter-labelled training corpus that de-specializes SALMONN-SQA from
a "lenient noise specialist" into a multi-dimensional rater that can perceive and
name **reverberation, bandwidth limiting, clipping and discontinuity** — its
documented blind spots (see `../FINDINGS.md`). Stage 0 already proved this info
survives in the frozen encoder features (`../stage0_encoder_probe.py`), so the fix
is data + targets, not architecture.

## What it produces

For each clip, one JSONL record whose `target_text` is what the model is trained
to emit, in the scoping doc's **calibrate-then-describe** order:

```
noise:5 reverberation:2 bandwidth:5 clipping:5 discontinuity:5 loudness:5.
This clip has strong reverberation suggesting a large, echoey room.
Overall MOS: 2.65
```

1. **Per-dimension 1–5 scores** (`noise reverberation bandwidth clipping
   discontinuity loudness`) derived from the **exact** degradation parameters —
   noise-free supervision. This is the direct fix for "the quality prior overrides
   the named dimension."
2. **A 1–3 sentence grounded description**, worst axis first. Every clause is
   backed by an applied degradation — it never names an axis that was left clean.
3. **`Overall MOS: X.XX`** last, de-compressed to two decimals.

## Pipeline

```
clean (LibriTTS-R, 16 kHz)
  └─ render(): reverb → noise → clip → bandwidth → codec → loudness → packet-loss
       reverb : real measured RIRs (RT60 + measured DRR)
       noise  : real MUSAN noise/music at known SNR (white-noise fallback)
       codec  : real Opus/MP3 encode-decode, scored into bandwidth via measured rolloff
       rest   : DSP (low-pass, hard-clip, frame-drop, regain) — every param recorded
  └─ metrics: PESQ (intrusive) + DNSMOS + NISQA
  └─ targets: param→score anchors, grounded description, param-anchored overall MOS
```

- `degradations.py` — the graded, parameter-known degradation chain.
- `targets.py` — param→1-5 anchors, MOS fusion, description templating, prompt.
- `generate_corpus.py` — orchestrator (balanced recipe sampling, metrics, JSONL).

## Run

```bash
uv run python -m experiments.planb.generate_corpus \
    --clean-dir "$SQA_LIBRITTS_ROOT/dev-clean" \
    --n 400 --out experiments/results/planb/corpus_dev.jsonl \
    --wav-dir "$SQA_CORPUS_WAV_DIR/wav_dev"
```

Dataset roots come from the environment (`SQA_LIBRITTS_ROOT`, `SQA_MUSAN_ROOT`, `SQA_RIR_ROOT`, …)
— see [`experiments/config.py`](../config.py) and [REPRODUCING.md](../../REPRODUCING.md).
The **full v3 pipeline** (corpus → paraphrase → two-stage train) is `train/run_v3.sh`.

`--no-metrics` skips PESQ/DNSMOS/NISQA for a fast schema dry-run (MOS falls back to
the param anchor only).

## Two design decisions worth knowing

**1. Any applied RIR floors reverb at 4, never 5.** The measured RIRs are *measured*
room responses at a distance; even RT60 0.15 imparts audible coloration (DRR
≈ −10 dB). Scoring those as "dry" mislabels clearly-reverberant clips as clean and
makes the description hallucinate "clean." Only un-convolved audio is `reverb:5`.

**2. Overall MOS is param-anchored, not pure metric fusion.** The original scoping
plan was to fuse PESQ + NISQA + DNSMOS. Tested and rejected: **PESQ floors at ~1.0
for *any* reverb** (mild and severe alike) and **DNSMOS is reverb-blind**, so a
fused-metric MOS cannot rank the very axes Plan B exists to fix — mild and severe
reverb both landed at MOS ≈ 1.4. Instead the MOS is anchored on the exact
per-dimension scores (worst-axis-dominated: `0.55·min + 0.45·mean`) and the metric
fusion is folded in at 30 % for perceptual realism and 2-decimal de-compression.
Result: MOS now ranks reverb severity monotonically (reverb 1→4 ⇒ MOS 2.1→3.7)
while staying continuous. See `targets.overall_mos`.

## Validation (dev-clean, n=400) — generator confirmed correct

First validated slice (`corpus_dev.jsonl`, LibriTTS-R dev-clean, commit 0fa0fbc):

- **Coverage:** every dimension populated across all 5 score bins; reverb (the
  priority axis) best-balanced at 95/60/68/65/112 over scores 1–5.
- **De-compression:** Overall MOS spans 1.70–4.83 with **186 distinct** 2-decimal
  values across 400 clips — no banding.
- **Monotonic severity:** single-axis reverb MOS rises 2.17 (extreme) → 3.74 (mild)
  → 4.7+ (clean); worst-axis-dominated MOS forms a clean ladder by `min(scores)`.
- **Grounding holds:** 0 / 400 descriptions name an axis that was left clean (the
  "never describe a degradation that wasn't applied" rule).

Caught and fixed two mislabel bugs during validation (the RIR-floor and
MOS-anchoring decisions above). dev-clean is the **dev/QA slice**; the training
corpus scales onto train-clean-100/360.

> Infra note: the venv's `datasets` lib is wedged (pyarrow 24 vs datasets 2.14,
> can't upgrade without breaking the transformers 4.35 pin). Read parquet directly
> with pyarrow; do **not** `import datasets`. Audio sources are plain wav dirs.

## Scaling

dev-clean (5,736 wavs) is the dev/QA slice. The training corpus draws from
**LibriTTS-R train-clean-100** (33,232 wavs, 247 speakers) with the MUSAN
(noise+music, ~1,590 files) and codec axes active:

```bash
uv run python -m experiments.planb.generate_corpus \
    --clean-dir "$SQA_LIBRITTS_ROOT/train-clean-100" \
    --n 4000 --seed 1 --out experiments/results/planb/corpus_train_v3pre.jsonl \
    --wav-dir "$SQA_CORPUS_WAV_DIR/wav_train_v3"
```

## Leakage

LibriTTS-R has no VCTK speakers, so it cannot overlap the VoiceBank-DEMAND test
speakers (p232/p257). No DEMAND noise is used; MUSAN is a disjoint noise family. The held-out eval suite (sweep /
calibration / enhancement / forced-choice d′) stays untouched as the success metric.
