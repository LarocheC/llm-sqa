# Plan B — data & target scoping for broader-degradation SQA fine-tuning

Goal: de-specialize SALMONN-SQA so it perceives reverb / bandwidth / clipping /
codec (not just additive noise) and emits a de-compressed, multi-dimensional MOS.
This document scopes **data** and **targets** before any training code.

> Citations below name the relevant works; treat specific arXiv IDs/dates as
> "to verify" — they came from web research and should be sanity-checked.

---

## Executive recommendation

A **hybrid, two-stage** recipe (the pattern recent literature converged on —
"LibriAugmented"-style synthetic pretrain → human-MOS finetune):

1. **Stage 0 (de-risk, cheap):** before any training, probe whether the **frozen**
   Whisper+BEATs features even *encode* reverb/bandwidth (linear probe → RT60 /
   cutoff). If they don't, LoRA+Q-Former training can't fix it and we'd need to
   unfreeze encoders. This is a half-day check that could save the whole effort.
2. **Stage 1 — synthetic, param-labeled corpus (the backbone):** degrade
   leakage-free clean speech with **known parameters**; supervise per-dimension
   1–5 scores directly from those params + a continuous fused MOS. This is what
   directly cures the blind spots and gives noise-free dimension labels.
3. **Stage 2 — human calibration:** fine-tune on a human-rated slice
   (**QualiSpeech** + **NISQA**) so the scale and descriptions match real
   perception.

Keep the same trainable parts as the original checkpoint (Q-Former + projection +
Vicuna-LoRA; encoders frozen) — unless Stage 0 says otherwise. Fits the 4090.

## Stage 0 result — PASSED ✅ (encoders can stay frozen)

A linear probe (PCA + ridge, held out by utterance, N=20) on the **frozen**
features shows reverb (RT60), bandwidth (cutoff) and clipping are **strongly
decodable at every tap point — including the trained Q-Former output that feeds
the LLM**:

| target | whisper R² | beats R² | concat R² | qformer-out R² |
|---|---|---|---|---|
| reverb (RT60) | 0.87 | 0.95 | 0.95 | **0.94** |
| bandwidth (cutoff) | 0.94 | 0.97 | 0.94 | **0.87** |
| clipping (fraction) | 0.80 | 0.88 | 0.82 | **0.82** |

(Spearman 0.90–0.98 throughout.) The degradation info is present in the audio
representation and survives to the LLM input. So the blindness is **not** in the
frozen front-end or the Q-Former bottleneck — it's in the LLM's learned *mapping*
(Vicuna+LoRA was only ever trained to verbalize noise). **Plan B is viable as
scoped with encoders frozen**: the fix is data/targets, not architecture.
See `experiments/stage0_encoder_probe.py`.

## Stage 1 status — generator BUILT ✅ (`experiments/planb/`)

The synthetic degradation + target generator is implemented and validated
end-to-end on leakage-safe **LibriTTS-R dev-clean** (5,736 clips, no VCTK):

- `planb/degradations.py` — graded, parameter-known chain: real **measured RIRs** for
  reverb (RT60 + measured DRR), DSP low-pass (bandwidth), hard-clip (clipping),
  white noise (kept modest), frame-drop (discontinuity), regain (loudness).
- `planb/targets.py` — param→1-5 anchors per dimension, grounded description
  templating, and the overall-MOS model.
- `planb/generate_corpus.py` — balanced recipe sampling (blind-spot-weighted),
  PESQ/DNSMOS/NISQA, JSONL output. See `planb/README.md`.

**Two findings that changed the plan during the build:**

1. *Any applied RIR floors reverb at 4, never 5.* The measured RIRs are measured room
   responses at a distance (DRR ≈ −10 dB even at RT60 0.15), so they are never
   "dry." Scoring them 5 mislabelled clearly-reverberant clips as clean.
2. *The overall MOS is param-anchored, not pure metric fusion (deviation from the
   original plan above).* Pure PESQ+NISQA+DNSMOS fusion was **tested and rejected**:
   PESQ floors at ~1.0 for *any* reverb and DNSMOS is reverb-blind, so the fused
   MOS could not rank reverb severity — the core Plan B axis. MOS is now anchored on
   the exact per-dimension scores (`0.55·min + 0.45·mean`) with metrics folded in at
   30 % for realism + 2-decimal de-compression. It now ranks reverb monotonically
   (reverb 1→4 ⇒ MOS ≈ 2.1→3.7) and uses ~all distinct 2-decimal values.

**Scale-up DONE:** clean pool extended to LibriTTS-R **train-clean-100** (33,232
wavs / 247 speakers); **MUSAN** real noise/music (~1,590 files, exact SNR) replaces
white noise; **codec** (real Opus/MP3 via torchaudio AudioEffector) added and folded
into the bandwidth dimension via measured spectral rolloff (codec floored at 4 —
never transparent). First scaled corpus = 4,000 clips (`corpus_train.jsonl`).

**Next unbuilt step:** the two-stage LoRA SFT trainer (Stage 1 calibration on the
score block, Stage 2 + grounded description & MOS), evaluated with the existing
suite (sweep / calibration / enhancement / forced-choice d′). Optional: extend to
train-clean-360 for more speakers.

---

## DATA

### Clean speech (leakage-safe)
The hard constraint: **the eval set (VoiceBank-DEMAND-16k test) is VCTK speakers
p232 & p257.** Any VCTK-derived training data must exclude them (and ideally the
whole VBD train speaker set).

| source | role | why | leakage |
|---|---|---|---|
| **LibriTTS-R** (OpenSLR SLR141, CC BY 4.0) | **primary clean pool** | large, clean, audiobook domain → forces degradation sensitivity, not memorized timbre | none (no VCTK) |
| VCTK minus {p232,p257,+VBD-train spk} (CC BY 4.0) | optional in-distribution timbre | matches VBD recording style | **needs speaker exclusion** |
| local `denoiser/.../VCTK_concat` (60×120 s, 16 kHz) | tempting (on disk) | already 16 kHz mono | **BLOCKED**: anonymized concat may contain p232/p257 — verify or skip |

→ **Use LibriTTS-R as the bulk.** Resample 24→16 kHz mono. Skip the local VCTK-concat unless we verify p232/p257 are absent.

### Degradations (on-the-fly, graded severity, **known params**)
Bias the curriculum toward the blind spots; keep noise modest (already strong).

| axis | source / method | severity label | status |
|---|---|---|---|
| **reverb** (priority) | local **measured RIRs** `~/data/RIRs` (RT60-labelled, ~84 GB, 16 kHz) + **OpenSLR SLR28** (real+sim, 16 kHz, Apache-2.0) + **pyroomacoustics** (sweep RT60/DRR) — **mix real+sim** | RT60 / DRR bucket | measured RIRs local; SLR28 download; pyroomacoustics = 1 pip |
| **bandwidth** | DSP low-pass + resample to 8/4 kHz (codec-independent) | cutoff / eff. Nyquist | trivial, reliable |
| **codec** | `torchaudio.io` bundled ffmpeg (Opus 6–24 kbps, MP3) — **probe `get_audio_encoders()` first** | bitrate | needs runtime check |
| **clipping** | scale + hard-clip | clip fraction | trivial (have it) |
| **packet loss** | frame dropping | loss rate | trivial |
| **noise** (keep modest) | **MUSAN** (SLR17, 16 kHz) + WHAM! — **exclude DEMAND** (eval's noise) | SNR | download |

### Held-out eval (do not train on)
VoiceBank-DEMAND-16k test (existing) + our degradation sweep + optionally
**TCD-VoIP** (real clip/echo MOS) and NISQA's LIVETALK as extra probes.

### Leakage rules (hard)
- Exclude VCTK p232/p257 (and VBD-train speakers) from any VCTK-derived data.
- **No DEMAND noise** in training (eval's noise family).
- **No DNS-Challenge *clean*** (contains VCTK) — DNS noise/RIR only if used.
- Screen QualiSpeech/NISQA for any VBD-test clip/speaker after download.

---

## TARGETS

### Per-clip target schema (structured, parseable, calibrate-then-describe)
Emit verbatim text the model learns; parse with regex:
```
noise:4 reverberation:2 bandwidth:3 clipping:5 discontinuity:5 loudness:4.
The speech is noticeably reverberant, as if in a large room, and slightly
band-limited; little background noise.
Overall MOS: 2.73
```
- **Per-dimension 1–5 scores FIRST** (the calibration scaffold), covering the
  blind-spot axes: noise, reverberation, bandwidth/coloration, clipping/distortion,
  discontinuity, loudness.
- **1–3 sentence grounded description** naming the dominant degradation(s).
- **`Overall MOS: X.XX` LAST** (chain-of-thought: scoring after describing lowers error).

### How to generate targets (leakage-free, mostly on disk)
1. **Dimension scores ← known degradation params** (param→1–5 anchor with fixed
   criteria). Noise-free supervision; this is the **direct fix for "quality prior
   overrides the named dimension."**
2. **De-compress the overall MOS:** fuse multiple references (PESQ + NISQA + DNSMOS
   mapped to a common 1–5 axis), **keep 2 decimals**, never copy a single 5-level
   teacher. Optional: a small scalar regression head on the LLM hidden state if
   eval still shows banding.
3. **Descriptions: template → LLM-paraphrase → verify.** Paraphrase for diversity
   (shown to materially cut error), but **ground every claim in a true param**
   (never describe a degradation that wasn't applied); randomize phrasing/order.

### Training
- **Stage 1 (calibration SFT):** emit the per-dimension score block; **balanced
  sampling per dimension-bin** so the model can't exploit the marginal (this is
  the lever against "all reverb=3").
- **Stage 2 (reasoning SFT):** add the grounded description + final MOS; keep the
  numeric scores in the output ("concise-with-num" beats free text).
- Optional later: GRPO with **dimension-wise** rewards (per-axis accuracy + desc
  similarity) using our known params as cheap programmatic reward — **not** a single
  "overall quality" reward (that's what lets the quality prior swamp the axes).

---

## The key decision — where does the MOS/label signal come from?

| | A. Synthetic only | B. Human corpora only | **C. Hybrid (recommended)** |
|---|---|---|---|
| MOS target | fused PESQ/NISQA/DNSMOS | NISQA / QualiSpeech human MOS | synthetic backbone → human calibrate |
| dimension labels | exact (from params) | human (noisy, 1 rater) | exact + human |
| blind-spot coverage | full (we control it) | partial (corpus-dependent) | full |
| realism of MOS | metric proxy | real opinion | real after Stage 2 |
| effort | low (no downloads/licensing) | medium (downloads, scale calibration, leakage screen) | medium |
| risk | teacher-metric bias (e.g. DNSMOS is reverb-blind → fuse, don't single-source) | scale heterogeneity, leakage, no parametric severity | most robust; most moving parts |

**Recommendation: C.** The synthetic backbone is what actually injects the missing
vocabulary and gives exact per-axis labels (cures reverb/bandwidth/clip); the human
slice (QualiSpeech for descriptive targets + NISQA for multi-dim MOS) calibrates to
real perception and de-compresses the scale. **QualiSpeech is a major find** — a
ready-made human descriptive-SQA dataset (per-dimension scores + descriptions +
reasoning) in almost exactly our target format.

## Critical risk (flag now)

The reverb/bandwidth blindness may live in the **frozen** front-end: Whisper is
ASR-trained (reverb-*invariant* by design — bad for us), and the ~17:1 window-level
Q-Former compresses long reverb tails. Recent work attributes a real chunk of
per-dimension accuracy to making the **audio encoder trainable** — which our
"match the original recipe" constraint forbids. **Hence Stage 0:** a linear probe
on the frozen encoder features for RT60/cutoff. If reverb isn't decodable there,
the plan changes (unfreeze encoders = heavier), and we'd rather know on day one.
**→ RESOLVED (see "Stage 0 result" above): all three are strongly decodable
(R² 0.82–0.95) including at the Q-Former output — encoders stay frozen; the fix is
data/targets, not the architecture.**

## Effort / feasibility
- Fits a single 4090 (LoRA-SFT of 7B; same as the existing checkpoint).
- Downloads: LibriTTS-R (subset), SLR28, MUSAN, NISQA (~free), QualiSpeech (HF, CC-BY-NC). measured RIRs already local. Tens–low-hundreds of GB if we take subsets.
- Reuse: PESQ/STOI/DNSMOS/NISQA already wired; the **eval suite (sweep + calibration + enhancement + forced-choice d′) is the success metric** — success = reverb/bandwidth sweep ρ go strongly negative, calibration ρ vs PESQ/NISQA rises toward ~0.7, enhancement MOS tracks the gain, forced-choice d′ > 0, MOS spreads beyond 5 values.

## Decision points for you
1. **MOS-target approach:** A / B / **C (rec)**.
2. **Stage 0 encoder probe first?** (cheap; strongly recommended).
3. **Encoders frozen** (match original) vs allow unfreezing if Stage 0 demands it.
4. **License posture:** several human corpora are CC-BY-NC (research-only) — fine for a study, matters if productized.
