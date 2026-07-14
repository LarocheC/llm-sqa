# Plan B — Summary: de-specializing SALMONN-SQA

Standalone summary of Plan B: fine-tuning the SQA-finetuned SALMONN audio LLM from a
"lenient noise specialist" into a calibrated, multi-dimensional speech-quality rater.
Full experiment log: [`FINDINGS.md`](FINDINGS.md). Code: [`planb/`](planb/).

**Final model:** `experiments/results/planb/ckpt_stage2_v3/202606141722/checkpoint_best.pth`

---

## 1. The problem

The original SQA checkpoint (from the five baseline experiments in `FINDINGS.md`) was a
**lenient noise specialist**: its descriptions were good for additive noise, but as a
quality *meter* it was weak —

- MOS was a **coarse 5-value scale floored at 2.5**, agreeing with PESQ/NISQA/DNSMOS
  only at ρ 0.40–0.49 (those metrics agree with *each other* at 0.72–0.82 → it was an
  outlier).
- **Blind to reverberation** (ρ(MOS,severity) −0.11), and weak on bandwidth/clipping.
- **Blind to enhancement** (MOS moved +0.03 while every objective metric improved).
- Sometimes collapsed into a made-up JSON schema under out-of-distribution degradation.

Goal: keep the descriptive ability, but make the model **perceive and name every common
degradation** and emit a **de-compressed, calibrated MOS**.

---

## 2. Stage 0 — de-risk: is the degradation visible out of the frozen encoders?

**Why first.** Whisper is ASR-trained (deliberately reverb-*invariant*) and the
window-level Q-Former compresses ~17:1, so reverb/bandwidth might be destroyed in the
**frozen** front-end. If the information isn't there, no amount of Q-Former/LoRA training
can recover it and we'd have to unfreeze the encoders (much heavier). A cheap probe
decides the whole approach.

**How** (`planb/stage0_encoder_probe.py`). Apply graded degradations with *known*
parameters — reverb (measured RIRs → RT60), bandwidth (low-pass → cutoff), clipping (→
fraction) — then extract features at four tap points of the **frozen** model and fit a
**linear probe** (standardize → PCA → ridge), scored by **R² + Spearman, held out by
utterance**. A *linear* probe is the point: it tests whether the info is linearly
**decodable** (present and accessible), not buried.

| tap point | what it is |
|---|---|
| `whisper` | Whisper encoder output (1280-d, frozen) |
| `beats` | BEATs encoder output (768-d, frozen) |
| `concat` | `[whisper \| beats]` (2048-d) — what feeds the Q-Former |
| `qformer` | Q-Former + projection output — the trained bottleneck that reaches the LLM |

**Result — PASSED.** Reverb / bandwidth / clipping were strongly decodable at *every*
tap, including the Q-Former output (R² 0.82–0.95, Spearman 0.90–0.98). The degradation
signal survives the frozen front-end and reaches the LLM → the blindness is in the LLM's
**learned mapping**, not the architecture.

**Decision:** encoders stay frozen (the original recipe); the fix is **data + targets**.

Plots (`planb/stage0_r2_bars.png`, `planb/stage0_scatter.png`): the bar chart shows
held-out R² per target × tap (all well above the 0.5 decodable line); the scatter
shows predicted-vs-true RT60 / cutoff / clip-fraction sitting tight on the y=x line
at every tap — including `qformer` (the bottleneck that feeds the LLM). Regenerate
with `python experiments/stage0_encoder_probe.py`.

---

## 3. v3 — the final training (what data, where, how)

Pipeline: `planb/train/run_v3.sh` → generate corpus → Opus paraphrase → two-stage
LoRA-SFT. Composition below is verified against `corpus_train_v3.jsonl`.

### Clean speech (leakage-safe)
- **Train:** LibriTTS-R **train-clean-100** — 4,000 clips (seed 1).
- **Val:** LibriTTS-R **dev-clean** — 300 clips (seed 7).
- LibriTTS-R has no VCTK speakers → disjoint from the VoiceBank-DEMAND eval speakers
  (p232/p257). No DEMAND noise is used in training.

### Degradations (applied on-the-fly, known params; `planb/degradations.py`)
A blind-spot-weighted sampler gives each clip 0–3 axes. Verified counts (4,000 train):

| axis | how | v3 corpus |
|---|---|---|
| **reverb** | real measured RIRs (RT60+DRR) **+** synthetic exp-decay | 2,783 (1,934 real / 849 synthetic) |
| **noise** | real **MUSAN** (noise+music) **+** synthetic white/pink/brown, at known SNR | 512 (266 real / 246 synthetic) |
| **codec** | real Opus/MP3 (torchaudio), scored into bandwidth via measured rolloff | 509 |
| **bandwidth** | DSP Butterworth low-pass (cutoff) | (DSP) |
| **clipping** | hard-clip (clipped fraction) | (DSP) |
| **discontinuity** | frame-drop (loss rate) | (DSP) |
| **loudness** | re-gain (dB) | (DSP) |
| **clean** | no degradation | 373 |

The real **and** synthetic noise/reverb mix was the v3 addition — to close the
train/eval gap (the held-out sweep uses *synthetic* degradations the earlier corpus
never trained on).

### Targets (`planb/targets.py`)
- **Per-dimension 1–5 scores** ← exact degradation params (noise-free anchors). The
  direct fix for "quality prior overrides the named dimension".
- **Overall MOS** ← `0.55·min + 0.45·mean` of the dimension scores, blended **70/30**
  with a fused **PESQ + NISQA + DNSMOS** metric MOS, kept to 2 decimals. (Pure metric
  fusion was tried and rejected — PESQ floors on any reverb and DNSMOS is reverb-blind,
  so it couldn't rank the very axes we needed.)
- **Description** ← templated, then **paraphrased by Claude Opus 4.8** (`planb/paraphrase.py`):
  deduped to ~740 unique degradation profiles, one API call each → a pool of variants,
  **grounding-verified** (must name every degraded axis, no clean axis), sampled per
  clip. All 4,300 descriptions paraphrased, **0 template fallbacks**.

### Two-stage LoRA-SFT (`planb/train/run_v3.sh`)
Trainable: Q-Former + speech→LLaMA projection + Vicuna-LoRA (r=8, α=28). **Frozen:**
Whisper, BEATs, Vicuna base (Stage 0 said the encoders can stay frozen).

| stage | target text | task | start from | epochs |
|---|---|---|---|---|
| **1 — calibration** | per-dimension **score block only** | `sqa_score` | released SQA ckpt | 2 |
| **2 — reasoning** | score block **+ paraphrased description + `Overall MOS`** | `sqa_full` | Stage 1 `checkpoint_best` (`strict=False`) | 4 |

So the model first learns the calibrated numbers, then learns to describe and score on
top of that (calibrate-then-describe).

---

## 4. Results — v3 vs. the original (all five findings re-measured)

Held-out: synthetic degradation sweep on VoiceBank-DEMAND clean (training used real
RIRs/MUSAN/codec on LibriTTS — disjoint speakers). Scripts: `planb/eval_compare.py`,
`planb/eval_voicebank_v3.py`, `planb/eval_enhancement_v3.py`, `planb/eval_realnoise.py`.

| # | Original finding | **v3** |
|---|---|---|
| 1 | MOS↔SNR ρ 0.37, lenient | ρ **0.50**; penalizes low SNR; names noise 99% (low SNR) |
| 2 | Calibration ρ 0.40–0.49 (outlier); **5 values floored at 2.5** | ρ **0.69–0.75** (inside the metrics' 0.72–0.82 band); **81 distinct values** (2.10–4.89) |
| 3 | reverb **−0.11**, bw −0.39, clip −0.33 | reverb **−0.95**, bw −0.81, clip −0.83, noise −0.93 |
| 4 | Enhancement: MOS +0.03, ρ(gain,PESQ)≈0 | MOS gain **+0.68** (76% of files), ρ **+0.22** |
| 5 | Acquiescence — leading yes/no → always "YES" | emits calibrated discriminative scores; failure mode N/A |

Per-dimension score sensitivity ρ(score,severity): noise −0.82, bandwidth −0.88,
clipping −0.85, reverb −0.95. Output robustness: **0/108 degenerate** (original 20 + 6
unparsed). Real-noise control (held-out MUSAN): prose naming climbs v1 30% → v2 72% →
**v3 90%**, approaching the original's 92% while v3 *also* de-blinds the other axes.

Plots (`planb/`): `mos_vs_snr_v3.png`, `mos_vs_neural_v3.png`, `degradation_sweep_v3.png`.

---

## 5. Bottom line

v3 turns the original "lenient noise specialist with a weak 5-value MOS" into a
**calibrated, multi-dimensional rater**: the MOS is de-compressed (5 → 81 values) and
agrees with PESQ/NISQA/DNSMOS as well as they agree with each other; it perceives and
ranks every degradation type (including the former reverb blind spot); it tracks
enhancement gain (usable as an automatic enhancer evaluator — it wasn't before); and it
names degradations in natural, grounded prose. **The per-dimension structured scores are
the most reliable signal.**

**Evaluate:** point `salmonn_sqa/inference_config.yaml` `model.ckpt` at the v3 checkpoint
and prompt with the `sqa_full` instruction (`planb/train/test_prompt_planb.json`).

### Optional further work
- Scale the clean pool to **train-clean-360** for more speaker diversity.
- More degradation *types* and real-world recordings to harden far-OOD robustness.
- A small scalar regression head on the LLM hidden state if any MOS banding remains.
