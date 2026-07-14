# SALMONN as a descriptive speech-quality assessor — findings

A study of the SQA-finetuned SALMONN audio LLM on the **VoiceBank-DEMAND-16k**
test set (824 clean/noisy pairs), characterizing what it can and cannot do as a
speech-quality model. Five experiments, run on an RTX 4090.

## TL;DR

SALMONN-SQA is a **lenient noise specialist**. Its natural-language *description*
is the real asset — for additive noise it perceives, time-localizes, and even
names the noise environment. But as a quality *meter* it is weak: the **MOS is a
coarse, optimistic, 5-value scale** that agrees with purpose-built predictors
only about half as well as they agree with each other, is **largely insensitive
to non-noise degradations** (clipping, bandwidth, reverb) and **to enhancement**,
and is **suggestible** — leading yes/no prompts just elicit "YES". Use the prose
for noise characterization; do **not** use the number as a calibrated MOS or as an
automatic enhancer evaluator.

## Setup

- **Model**: SALMONN-7B (Whisper + BEATs → window-level Q-Former → Vicuna-7B/LoRA),
  SQA-finetuned checkpoint. Greedy beam search, fp16; the canonical prompt in
  `salmonn_core.DEFAULT_SQA_PROMPT`.
- **Data**: `JacobLinCool/VoiceBank-DEMAND-16k` test split. Per-pair SNR computed
  directly from `noise = noisy − clean`.
- **Reference metrics**: PESQ (wb), STOI, SI-SDR, segmental SNR (intrusive);
  DNSMOS P.835 and NISQA v2 (reference-free neural MOS predictors).
- **Reproduce**: see "Scripts" at the bottom. Results land in `results/` (gitignored).

## Architecture

Two parallel encoders take **different features** of the same audio — the log-mel
spectrogram feeds Whisper, the raw waveform feeds BEATs — then fuse and pass
through a Q-Former bottleneck into Vicuna. Only the Q-Former + projection + LoRA
are trained (the SQA checkpoint); Whisper, BEATs and the Vicuna base are frozen.

```
                       16 kHz mono waveform  (1–30 s)
                                      │
              ┌───────────────────────┴───────────────────────┐
              │ feature = log-mel spectrogram                  │ feature = raw waveform
              │   WhisperFeatureExtractor                      │   (must stay float64 —
              │   → [80 mel-bins × ~3000 frames]               │    float32+fp16 → NaN)
              ▼                                                ▼
   ┌──────────────────────────┐                    ┌──────────────────────────┐
   │  WHISPER-Large-v2 encoder │                    │  BEATs encoder            │
   │       (FROZEN)            │                    │      (FROZEN)             │
   │  → [~1500 × 1280]         │                    │  → [~1500 × 768]          │
   │  "what is said" / speech  │                    │  "how it sounds" / audio  │
   └────────────┬─────────────┘                    └────────────┬─────────────┘
                │ LayerNorm                                       │ LayerNorm
                └─────────────────────┬───────────────────────────┘
                                      ▼
                       concat on feature axis  → [~1500 frames × 2048]
                                      │
                                      ▼
                  ┌──────────────────────────────────────────┐
                  │  WINDOW-LEVEL Q-FORMER     (TRAINED)       │
                  │  slices into ~0.33 s windows (~17 frames), │
                  │  1 query token per window                  │
                  │  → ~88 "speech tokens" × 768               │
                  └────────────────────┬───────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │  Linear projection  (TRAINED)  768 → 4096  │
                  │  → ~88 tokens × 4096                       │
                  └────────────────────┬───────────────────────┘
                                       │  audio embeddings (LLM space)
  SQA prompt text                      │
  "USER: <Speech><SpeechHere></Speech> ▼
   {describe quality…}\nASSISTANT:" ──► splice audio tokens AT <SpeechHere>
        │ (Vicuna tokenizer→embeds)     → [ text-emb | 88 audio-emb | text-emb ]
        └───────────────────────────────────────┬──────────────
                                                 ▼
                  ┌──────────────────────────────────────────┐
                  │  VICUNA-7B v1.5  +  LoRA (r=8, α=28)       │
                  │  base FROZEN, only LoRA TRAINED            │
                  │  beam search (4), fp16, ≤ 400 new tokens   │
                  └────────────────────┬───────────────────────┘
                                       ▼
            "The audio has outdoor chatter from 0–3 s … MOS of 4.0"
                                       │
                          ┌────────────┴────────────┐
                          ▼                          ▼
                  description (the asset)     extract_mos() → 4.0
```

This explains the capability profile below: acoustic-degradation cues are what
**BEATs** captures, so it's a noise specialist; reverb/bandwidth are subtle
spectro-temporal effects that survive the ~17:1 Q-Former compression poorly.

**Evaluation harness** (how the experiments wrap the model):

```
  VoiceBank-DEMAND pair ─┬─ clean ───────────────────────────► reference (PESQ/STOI/SI-SDR)
                         └─ noisy ─┬───────────────────────────► SALMONN ─► desc + MOS
                                   └─ ConvFSENet (FP32 ONNX) ─► enhanced ─► SALMONN
        degradation sweep:  clean ─► {+noise │ lowpass │ clip │ reverb} ─► SALMONN
  every clip ─► PESQ·STOI·SI-SDR (vs clean) + DNSMOS·NISQA (ref-free) ─► correlated vs SALMONN MOS
```

---

## 1. Description vs SNR — `voicebank_demand_sqa.py` + `analyze_voicebank.py`
*824 noisy + 824 clean.*

- **MOS tracks SNR** (Spearman 0.37): <5 dB → 3.60, 5–10 → 3.91, 10–15 → 4.08,
  ≥15 → 4.36, clean → 4.43.
- **It names the noise environment**: "outdoor chatter", "background music",
  "people talking", "traffic" appear in **41%** of low-SNR descriptions, falling
  to 25% at high SNR and **~1% on clean** — it's effectively identifying the
  DEMAND noise type, and time-localizing it ("from 0 to 3.8 seconds").
- "Noise is absent/minimal" language rises 11% → 44% with SNR.
- Clean vs noisy distinguishing words: noisy → *intrusive, outdoor, chatter,
  issues*; clean → *effortlessly, exceptional, uninterrupted, no strain*.
- 95/1648 replies were terse (a bare score, mostly on clean).

## 2. MOS calibration — `calibration.py`
*696 noisy files with a parsed MOS, vs objective + neural metrics.*

| SALMONN MOS vs | Spearman ρ |
|---|---|
| DNSMOS P.808 (holistic ACR) | **0.49** |
| PESQ | 0.48 |
| NISQA MOS | 0.44 |
| DNSMOS OVRL | 0.40 |
| DNSMOS SIG (signal distortion) | 0.26 |

**Cross-agreement (the key context):** DNSMOS/NISQA/PESQ agree with *each other*
at ρ = 0.72–0.82; SALMONN agrees with all of them only at **0.40–0.48**. It is an
outlier in quality-predictor space — capturing ~half the shared quality signal.

**Scale usage:** only 5 discrete values {2.5, 3.5, 4.0, 4.5, 5.0}, mean 3.98,
**floored at 2.5** — the worst-PESQ quartile (1.03–1.30, genuinely bad) still
averages 3.50. Best alignment with holistic P.808, worst with fine signal
distortion (SIG) — an "overall impression" rater, not a distortion analyzer.

## 3. Controlled degradation sweep — `degradation_sweep.py` + `analyze_degradation.py`
*16 utterances × {noise, bandwidth, clipping, reverb}, one variable at a time.*

| degradation | MOS sensitivity ρ(severity) | names it? | vs references |
|---|---|---|---|
| additive **noise** | **−0.78** | 88 → 100% | DNSMOS/NISQA/PESQ −0.93…−0.95 |
| **clipping** | −0.33 | up to 100% | names it, barely drops MOS; refs −0.66…−0.92 |
| **bandwidth** (lowpass) | −0.39 | ≤ 50% | mostly misses; DNSMOS also weak (−0.16), NISQA/PESQ catch it |
| **reverb** | **−0.11** | 0 – 31% | **blind**; refs −0.70…−0.75 |

A noise specialist: partially deaf to clipping, largely deaf to bandwidth, and
**effectively blind to reverberation** (degradations outside its training data).
Failure mode: under extreme out-of-distribution degradation it sometimes
**collapses into a made-up JSON schema** instead of an assessment.

## 4. Enhancement — `enhance_convfsenet.py` + `enhancement_experiment.py` + `analyze_enhancement.py`
*48 utterances × (noisy → ConvFSENet-enhanced → clean). Enhancer = Clement's
`claroche1/sparse-nsnet2` ConvFSENet, FP32 streaming ONNX.*

| condition | SALMONN MOS | PESQ | DNSMOS | NISQA |
|---|---|---|---|---|
| noisy | 4.20 | 2.23 | 2.86 | 3.24 |
| enhanced | 4.23 | 3.09 | 3.23 | 4.47 |
| clean | 4.61 | 4.64 | 3.37 | 4.57 |

Enhancement improves **every objective metric on 100% of files** (PESQ +0.86,
NISQA +1.23), but SALMONN's MOS moves **+0.03** (improved 27%, *worsened 21%*),
with **ρ(MOS gain, PESQ gain) ≈ 0.00**. It does not flag enhancement-specific
artifacts (artifact vocabulary on enhanced ≈ clean). → **SALMONN is blind to the
enhancement gain and unusable as an automatic enhancer evaluator** — leniency
leaves no headroom above noisy.

## 5. Prompt steering — `prompt_probe.py`
*Can a targeted prompt recover the blind spots? Open vs. leading yes/no prompt on
the same clip, with a clean false-positive control.*

| degradation | open-ended detects | targeted (degraded) | targeted (**clean**) |
|---|---|---|---|
| reverb | 33% | 100% "YES" | **100% "YES"** |
| bandwidth | 58% | 100% "YES" | **100% "YES"** |
| clipping | 100% | 100% "YES" | **100% "YES"** |

The model replies literally **"YES"** to every leading detection question — on
clean audio too. Targeted prompting has **zero discriminative power** (acquiescence
bias), not recovered detection.

### Follow-up: neutral forced-choice & rubric — `forced_choice_probe.py`
*Removes the yes-bias: balanced 2AFC labels (DRY/REVERBERANT, FULLBAND/MUFFLED,
CLEAN/DISTORTED) run in both option orders, scored with signal-detection d′; plus
a 1–5 rubric. 12 utterances, 252 clips.*

| degradation | 2AFC d′ | flips with option order | rubric ρ(severity) |
|---|---|---|---|
| reverb | +0.20 | **92%** (picks 1st word 96%) | **−0.30** |
| bandwidth | +0.77 (detects 8%) | 8% | **−0.75** |
| clipping | **+1.28** | 21% | −0.25 |

The model *follows* the format (one-word / number answers), but neutral framing
**does not recover the blind spots** — it exposes deeper failures: reverb is pure
**position bias** (answer flips with option order 92% of the time), bandwidth is
**near-blind** (says "MUFFLED" 8% of the time on a 2 kHz-limited clip), and the
**rubrics invert** — all ρ negative because the model reports overall *quality*
("higher = better") instead of the named attribute. **Clipping** shows weak-but-real
discrimination (d′ +1.28). → Reverb/bandwidth perception is **genuinely absent**,
not un-elicited; prompt steering can't fix it (motivates Plan B, retraining).

---

## What this means in practice

- **Use the description, not the number.** For noisy speech the prose is rich,
  accurate, and noise-aware; the MOS is a coarse 5-bucket proxy at best.
- **Don't use it to evaluate an enhancer.** It doesn't reward denoising and
  doesn't catch enhancement artifacts — PESQ/DNSMOS/NISQA do.
- **Don't trust leading yes/no probes.** It is suggestible.
- **Scope it to additive noise.** It is largely blind to reverb, bandwidth, and
  (in score) clipping.

## Caveats & next steps

- Modest N for exps 3–5 (16/48/12 utterances; only speakers p232/p257).
- Reference-free MOS predictors and PESQ are themselves proxies for human opinion
  — the real validity test is a **human-MOS corpus** (NISQA / ITU-T P.808).
- Prompt steering was tested in two framings — leading yes/no AND neutral
  forced-choice/rubric (`forced_choice_probe.py`). Neither recovers reverb or
  bandwidth; the limitation is in the model's perception, not the prompt. The
  open path is **Plan B (broader-degradation fine-tuning)**.
- Reverb/bandwidth blindness suggests the SQA fine-tune saw mostly additive noise;
  fine-tuning on a broader degradation set is the path to a general SQA describer.

## Scripts (run with `uv run python -m experiments.<name>`)

| script | produces |
|---|---|
| `voicebank_demand_sqa.py` | `results/voicebank_sqa.jsonl` (descriptions + SNR + MOS) |
| `analyze_voicebank.py` | `results/REPORT.md` (exp 1) |
| `calibration.py` | `results/CALIBRATION.md` (exp 2; PESQ/STOI/DNSMOS/NISQA) |
| `degradation_sweep.py` / `analyze_degradation.py` | `results/DEGRADATION.md` (exp 3) |
| `enhance_convfsenet.py` | ConvFSENet enhancer (importable) |
| `enhancement_experiment.py` / `analyze_enhancement.py` | `results/ENHANCEMENT.md` (exp 4) |
| `prompt_probe.py` | `results/prompt_probe.jsonl` + summary (exp 5) |

All deps are declared: `uv sync --extra experiments`. NISQA is fetched by
`scripts/setup_salmonn.sh` into `experiments/NISQA/` (gitignored). The ConvFSENet ONNX comes from a
separate repo — set `SQA_CONVFSENET_ONNX` (its analysis is reproducible from the committed JSONL
without it; see [REPRODUCING.md](../REPRODUCING.md)).

---

# Plan B outcome — fine-tuning closed the blind spots (held-out sweep)

After scoping (`PLAN-B-SCOPING.md`), Stage 0 de-risk, a synthetic param-labelled
corpus (`planb/`), and a two-stage LoRA-SFT (`planb/train/`), the fine-tuned model
was compared against the original SQA checkpoint on the **same controlled sweep**
(6 VoiceBank-DEMAND clean clips × noise/lowpass/clip/reverb × graded levels). This
is held-out: the eval degradations are *synthetic*, while training used real measured RIRs
RIRs / MUSAN / codec on LibriTTS (disjoint speakers). `planb/eval_compare.py`.

**ρ(MOS, severity) — does overall MOS fall as the degradation worsens?**

| family | orig | Plan B |
|---|---|---|
| noise | −0.90 | −0.88 |
| lowpass (bandwidth) | −0.37 | **−0.87** |
| clip | −0.48 | **−0.92** |
| reverb | −0.27 | **−0.95** |

**ρ(per-dimension score, severity), Plan B:** noise −0.77, bandwidth −0.82,
clipping −0.78, reverb −0.95 — every calibration head learned the right monotonic
mapping.

**Output robustness:** orig 20/108 degenerate + 6 unparsed MOS; Plan B **0/108
degenerate, 0 unparsed** — the structured format is far more reliable.

**Honest caveat — descriptions, not just scores:** prose *naming* of the
degradation (score block excluded) improved sharply for reverb (8%→100%) and
bandwidth (0%→58%) but *fell* for noise (97%→23%) and clip (79%→62%). Plan B now
carries degradation evidence chiefly in the **scores**; the free-text descriptions
are terser and less reliable than the numbers. Strengthening Stage-2 description
supervision (more/diverse grounded prose, keep the score block) is the next
iteration. **Net: the core goal — perceive & rank reverb/bandwidth/clipping and
de-compress MOS — is achieved.**

## Plan B v2 — description-quality iteration (richer prose, retrain Stage 2 only)

v1 produced correct scores but drifting descriptions (prose naming 23% noise /
62% clip). Root cause: ~2 phrasings/axis → template memorization. Fix: 5 phrasings
× 8 sentence frames per axis, describe ALL degraded axes (`targets.describe` v2);
regenerate only the descriptions (`rebuild_targets.py`, scores/MOS reused) and
retrain only Stage 2 from the existing Stage 1 ckpt (`train/run_stage2_v2.sh`).

3-way on the same held-out sweep (orig / v1 / v2):

| metric (family) | orig | v1 | v2 |
|---|---|---|---|
| ρ(MOS,sev) noise / lowpass / clip / reverb | −.90/−.37/−.48/−.27 | −.88/−.87/−.92/−.95 | −.88/−.85/−.86/−.95 |
| ρ(dim,sev) noise/bw/clip/rev | — | −.77/−.82/−.78/−.95 | −.73/**−.88**/**−.84**/−.95 |
| prose naming noise / lowpass / clip / reverb | 97/0/79/8% | 23/58/62/100% | **50**/**67**/50/100% |

**Read:** scores/MOS held strong (the reliable deliverable; bandwidth & clipping
dim-scores even improved). Prose naming improved for noise (23→50%) and bandwidth
(58→67%), held reverb (100%), but clip slipped (62→50%). The residual gap is a
genuine degradation-**type** discrimination limit on the *synthetic* sweep (loud
white noise → "clipped", strong clipping → "muffled") — and partly a synthetic-vs-
real train/eval mismatch (training noise = real MUSAN). Both v1 and v2: 0/108
degenerate vs orig 20+6 — the structured format is far more robust.

**Takeaways / next levers:** (1) use the structured *scores* as the primary signal;
(2) close the train/eval gap by adding synthetic degradation *types* to the corpus
(or eval on real MUSAN noise to confirm the mismatch hypothesis); (3) real
LLM-paraphrased descriptions (needs an API key) for deeper diversity; (4) scale to
train-clean-360.

## Real-noise control — held-out MUSAN noise (v1/v2/v3, CORRECTED metric)

The main sweep used *synthetic white noise*; training used *real MUSAN*. Re-ran a
noise-only sweep with **held-out MUSAN noise** (684 files never used in training)
on VoiceBank-DEMAND clean clips (`planb/eval_realnoise.py`).

> **Metric correction:** the first pass reported v1/v2 prose naming as 100% — that
> was a bug. `eval_realnoise` reused `eval_compare.description_only`, which only
> strips the score block for `tag=="planb"`; with tags `v1/v2/v3` it kept the
> score line, so `"noise:3"` trivially matched the "noise" keyword. Fixed with a
> local `prose()` that strips for every fine-tuned model. Corrected numbers:

| model | ρ(MOS,sev) | ρ(noise-dim,sev) | noise naming (prose) |
|---|---|---|---|
| orig | −0.59 | — | 92% |
| v1 | −0.72 | −0.67 | 30% |
| v2 | −0.82 | −0.73 | 72% |
| v3 | −0.79 | −0.65 | **90%** |

(vs synthetic white noise prose naming: orig ~92%, v1 23%, v2 50%, v3 **100%**.)

**Conclusion:** on real noise, prose naming climbs steadily v1 (30%) → v2 (72%) →
**v3 (90%)**, approaching the original's 92% — while v3 *also* de-blinds reverb /
bandwidth / clipping, which the original cannot name. So the description iteration
+ synthetic-type training were genuine wins, but the earlier "both name it 100%,
it was purely an eval artifact" claim was overstated: v1's real-noise prose was
weak (30%), and the synthetic sweep was *harder* but not *fake*. MOS ranking is
strong for all fine-tuned models (−0.72…−0.82 vs orig −0.59); the original's
"noise specialist" reputation was still partly an easy-white-noise artifact
(−0.90 synthetic vs −0.59 real). **v3 is best overall.**

**Plan B overall:** SUCCESS. v3 perceives & ranks reverb / bandwidth / clipping
(former blind spots) and noise, with a de-compressed MOS, robust structured output
(0 degenerate), and natural grounded descriptions that name the right axes ~75–100%
of the time on both synthetic and real degradations. Recommended checkpoint:
`ckpt_stage2_v3/202606141722/checkpoint_best.pth`; the structured per-dimension
scores remain the most reliable signal.

## Plan B v3 — synthetic degradation types + Opus-paraphrased descriptions

Two follow-ups, in one rebuild: (1) mix **synthetic noise (white/pink/brown) and
synthetic reverb** into the corpus alongside the real MUSAN/RIRs (close the
train/eval distribution gap), and (2) **LLM-paraphrase every description with Opus
4.8** (grounding-verified, deduped to ~740 profiles), replacing the templates.
Full Stage 1 + Stage 2 retrain (`train/run_v3.sh`). 4,300 descriptions paraphrased,
0 template fallbacks.

4-way on the same held-out synthetic sweep (orig / v1 / v2 / v3):

| metric (family) | orig | v1 | v2 | v3 |
|---|---|---|---|---|
| ρ(MOS,sev) noise/lowpass/clip/reverb | −.90/−.37/−.48/−.27 | −.88/−.87/−.92/−.95 | −.88/−.85/−.86/−.95 | **−.93**/−.81/−.83/−.95 |
| ρ(dim,sev) noise/bw/clip/rev | — | −.77/−.82/−.78/−.95 | −.73/−.88/−.84/−.95 | **−.82/−.88/−.85/−.95** |
| prose naming noise/lowpass/clip/reverb | 100/0/88/8% | 23/58/62/100% | 50/67/50/100% | **100/71/75/100%** |

**Read:** v3 closes the held-out weakness. **Noise prose naming 50%→100%** and
**clip 50%→75%** — the synthetic-type training fixed the OOD type-confusion that
made v2 misread synthetic white noise as "clipped". Per-dimension scores are the
best of any version (noise −0.82, all axes ≥ v2). MOS ranking stays strong (reverb
−0.95). v3 0/108 degenerate vs orig 20+6. The Opus paraphrases read naturally while
staying grounded (e.g. "a thick layer of background noise … paired with pronounced
reverberation that smears the words"). **v3 is the recommended checkpoint:**
`ckpt_stage2_v3/202606141722/checkpoint_best.pth`.

## Plan B v3 — re-evaluated on the original five findings (VoiceBank-DEMAND)

Re-ran the original analyses with the v3 checkpoint (`planb/eval_voicebank_v3.py`,
reusing cached PESQ/DNSMOS/NISQA/SNR). Plots: `planb/{mos_vs_snr_v3,mos_vs_neural_v3,
degradation_sweep_v3}.png`.

**#1 Description vs SNR:** ρ(MOS,SNR) **0.37 → 0.50**; v3 MOS by band 3.04/3.28/3.52/
3.72 (vs orig 3.60/3.91/4.08/4.36) — no longer floored/lenient at low SNR. Names the
noise at low SNR 90% → **99%**.

**#2 Calibration — FIXED:** v3 MOS vs references rises into the metrics' own
cross-agreement band (0.72–0.82): PESQ 0.48→**0.72**, NISQA 0.44→**0.75**, DNSMOS
P.808 0.49→**0.71**, OVRL 0.40→**0.69**. Scale de-compressed: orig **5 distinct
values floored at 2.50** → v3 **81 distinct values, range 2.10–4.89**. The original's
"outlier coarse rater" verdict no longer holds.

**#3 Degradation sweep — FIXED:** ρ(MOS,severity) noise −0.78→−0.93, clipping
−0.33→−0.83, bandwidth −0.39→−0.81, reverb **−0.11→−0.95** (the blind spot). Plus
per-dimension scores track each axis (−0.82…−0.95). No JSON-schema collapse (0/108
degenerate).

**#4 Enhancement — FIXED (re-run, `planb/eval_enhancement_v3.py`, 48 utts):**
noisy→enhanced MOS gain **+0.68** (76% of files improved) vs orig **+0.03** (27%);
ρ(MOS gain, PESQ gain) **+0.03 → +0.22** (modest — the per-utt magnitude isn't finely
ranked at n=48, but the gain is consistently positive). v3 condition means climb
3.53→4.17→4.55 (noisy/enhanced/clean) where orig was flat-and-floored 4.20→4.23→4.61.
v3 is usable as an enhancer evaluator; orig was not.

**#5 Prompt steering / acquiescence — resolved by design:** the original failure was
leading yes/no prompts always returning "YES" (no discriminative power). v3 doesn't
answer yes/no; it emits calibrated per-dimension scores that *do* discriminate
(reverb dim ρ −0.95, etc.), so the acquiescence failure mode is moot.

---

# The openly reproducible model (`open`) — public data end to end

The v3 result depended on a **non-public RIR set**, so nobody else could regenerate the corpus.
`open` rebuilds the reverberation axis on **OpenSLR SLR28** (`RIRS_NOISES`, Apache-2.0), making every
input public: LibriTTS-R (CC BY 4.0) + MUSAN (CC BY 4.0) + SLR28 + VoiceBank-DEMAND.

Published as the headline model: https://huggingface.co/claroche1/salmonn-sqa-planb-v3

## It was not a path swap — the severity map was wrong for real rooms

Two things broke when the RIR corpus changed, and both were load-bearing:

1. **RT60 was parsed from the filename.** Replaced with a **measurement** (Schroeder backward
   integration, `degradations.measure_rt60`), so the pipeline now works with any RIR corpus.
2. **The reverb severity map was RT60-primary with a `DRR < −15 dB` correction.** That rule was tuned
   to a set of RIRs measured at a roughly *constant* (distant) mic position. On a corpus where mic
   distance varies it is simply wrong:

   | | ρ with PESQ |
   |---|---|
   | RT60 | **−0.27** |
   | DRR  | **+0.67** |

   DRR predicts perceived degradation *far* better than RT60 — a long-RT60 room still sounds fairly
   dry when the mic is close. The old `DRR < −15` rule fired **0/240 times** on SLR28 (dead code).

   `score_reverb` now takes the **worse of the RT60 band and the DRR band**. Against PESQ, the label
   itself improved from ρ ≈ **+0.13…+0.24 → +0.709**.

   The "any applied RIR floors reverb at 4, never 5" rule was **re-verified, not assumed**: convolving
   clean speech with every RIR and measuring PESQ, *none* came out transparent — even the most benign
   (RT60 0.05 s, DRR +18 dB) scores PESQ 3.75. So it stands.

## Results: `open` vs `v3` — an honest trade, not a clean win

| | orig | v3 | **open** |
|---|---|---|---|
| MOS↔SNR ρ | +0.37 | **+0.50** | +0.46 |
| Calibration ρ (PESQ/NISQA/DNSMOS) | 0.40–0.49 | **0.69–0.75** | 0.65–0.71 |
| MOS scale (distinct values) | 5 | **81** | 63 |
| Enhancement: MOS gain / ρ | +0.03 / +0.03 | +0.68 / +0.22 | **+1.05 / +0.32** (91% of files) |
| Degradation sweep: bandwidth | −0.37 | −0.81 | **−0.88** |
| Degradation sweep: reverb | −0.27 | **−0.95** | −0.80 |
| Naming: bandwidth / clipping | 0% / 79% | 71% / 75% | **83% / 92%** |
| Degenerate outputs | 20/108 | **0/108** | **0/108** |

### The reverb result is a distribution story, and the synthetic sweep is misleading

Scored against **PESQ** — an independent reference that knows nothing about either model's severity
map — the two models disagree depending on *which kind of reverb* you test:

| reverb type | v3 | **open** |
|---|---|---|
| **synthetic** exp-decay (our generator; fixed artificial direct path) | **+0.610** | +0.391 |
| **SLR28 simulated** (image-method, varying DRR) — *open trained on these* | +0.328 | **+0.632** |
| **REAL measured** (RWCP / Aachen AIR / REVERB) — held out from **both** | +0.787 | **+0.827** |

`synth_reverb` injects an explicit direct path, so synthetic reverb has an artificially **high DRR at
every RT60**. v3, trained on uniformly-distant RIRs, keys almost entirely on RT60 — exactly how that
sweep grades severity — so it scores well there and `open` does not.

**On genuinely measured rooms the two models are close, and `open` leads only modestly (+0.83 vs
+0.79).** The large gap is on SLR28's *simulated* RIRs, which `open` trained on — that comparison
largely measures an in-distribution advantage, not generalization. v3 is *not* bad at reverb on real
rooms. Both models do markedly better on real RIRs than on either synthetic kind.

(An earlier version of this control sampled the RIR pool uniformly. SLR28 is ~60k simulated vs ~320
measured responses, so it drew only simulated RIRs while calling them "real", and reported a ~2×
gap. `eval_realreverb.py` now filters to measured RIRs explicitly and asserts on it.)

**Conclusion:** `open` is modestly better on real rooms, clearly better on enhancement, bandwidth and
naming, and is the only variant reproducible from public data. v3 remains better on the *synthetic* sweep, on
clipping, and is slightly better calibrated on VoiceBank-DEMAND. Both are published; the trade is
documented rather than hidden.

Caveat: the measured-RIR control is 96 clips × 12 held-out RIRs. Enough to show `open` is not worse
on real rooms; not enough to call a +0.04 difference decisive.
