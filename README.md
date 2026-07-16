# SQA — Descriptive Speech Quality Assessment with an Audio LLM

Turn the **SALMONN** audio LLM into a speech-quality rater that *describes* what is wrong with a
recording — and, after the fine-tune in this repo, ranks degradation severity instead of being
blind to it.

> **Scope of the claim.** Calibration here is **proxy-calibration**: the training target is
> parameter-anchored (70%) blended with PESQ/NISQA/DNSMOS (30%), and the reported calibration
> gains are measured against those same three metrics. No human MOS is used anywhere in this
> pipeline, so none of the numbers below are validated against human perception. The
> metric-independent results are the severity ordering against known synthesis parameters, the
> forced-choice probe, and the encoder probe. See *Known limitations*.

```
noise:2 reverberation:1 bandwidth:4 clipping:5 discontinuity:5 loudness:5.
The recording suffers from heavy reverberation and persistent background noise, and the
limited frequency range gives the voice a somewhat boxed-in, muffled character.
Overall MOS: 1.87
```

## The result (Plan B)

The released SQA checkpoint was a **lenient noise specialist**: good at describing additive noise,
but as a *meter* it was weak — a coarse 5-value MOS floored at 2.5, effectively **blind to
reverberation** (ρ(MOS,severity) = −0.11), and unmoved when a denoiser improved the audio (+0.03).

A Stage-0 linear probe showed the degradation signal *was* present in the frozen encoders
(R² 0.82–0.95 at every tap, including the Q-Former output that reaches the LLM). The blindness was
in the LLM's learned **read-out**, not the architecture — so the fix was **data + targets**, with
the encoders left frozen. Two-stage LoRA-SFT on a 4,000-clip corpus with *known* degradation
parameters ("calibrate, then describe"):

| # | Original | **published model** |
|---|---|---|
| 1 | MOS↔SNR ρ 0.37 | **ρ 0.46**; names noise 98% at low SNR |
| 2 | Proxy-calibration ρ 0.40–0.49; **5 distinct MOS values** | **ρ 0.65–0.71**; **63 distinct values** |
| 3 | reverb −0.27, bandwidth −0.37, clipping −0.48 | reverb −0.80, **bandwidth −0.88**, clipping −0.81 |
| 4 | Enhancement: MOS **+0.03**, ρ ≈ 0 | MOS gain **+1.05** (91% of files), ρ **+0.32** |
| 5 | Degenerate JSON under OOD input | **0/108 degenerate** |

Real reverberation (held-out **measured** RIRs): ρ(MOS, PESQ) **+0.83** (vs +0.79 for the older
model). The *rooms* are held out, not PESQ — PESQ is one of the three metrics in the training
target, so this is proxy agreement on unseen rooms, not external validation.
**Every training input is public** — LibriTTS-R + MUSAN + OpenSLR SLR28 RIRs — so the whole pipeline
reproduces end to end. See [FINDINGS.md](experiments/FINDINGS.md) for the honest `open` vs `v3`
trade-off (v3 scores higher on the *synthetic* reverb sweep; on *measured* rooms the two are close,
with `open` modestly ahead).

📄 [PLAN_B_SUMMARY.md](experiments/PLAN_B_SUMMARY.md) · [FINDINGS.md](experiments/FINDINGS.md) (full log) ·
🤗 [model weights](https://huggingface.co/claroche1/salmonn-sqa-planb-v3) ·
📚 [SFT dataset](https://huggingface.co/datasets/claroche1/sqa-degraded) ·
🔁 [REPRODUCING.md](REPRODUCING.md)

## Reproduce it

Every figure and number above regenerates from a bare clone — **no GPU, no datasets** (the corpora,
eval outputs, and metrics are committed):

```bash
git clone https://github.com/LarocheC/llm-sqa && cd llm-sqa
uv sync --extra experiments
uv run python -m experiments.planb.eval_voicebank_v3 --analyze   # findings #1 + #2
uv run python scripts/check_env.py                               # what else could I run?
```

To **evaluate the model** on your own audio (needs a GPU):

```bash
bash scripts/setup_salmonn.sh                # SALMONN + Whisper/Vicuna/BEATs + NISQA
uv run python scripts/fetch_checkpoints.py   # published weights (~121 MB) from the HF Hub
uv run python -m experiments.planb.eval_compare --n-clips 6
```

To **regenerate the corpus and retrain** (~3 h on a 24 GB GPU), see
[REPRODUCING.md](REPRODUCING.md) — including the datasets you need to supply.

## Repository map

| Path | What it is |
|------|------------|
| `salmonn_core.py` | **Shared inference core** — model loading, the canonical SQA prompt, audio prep, prompt formatting, MOS extraction, inference context. Single source of truth. |
| `api_inference.py` | FastAPI server (`/assess`, `/assess-batch`, `/health`). |
| `batch_process_sqa.py` | Batch runner over a directory, logged to one MLflow run. |
| `client_example.py` | Python client + CLI for the API. |
| `model_introspection.py`, `visualization_utils.py` | Introspection library (peek inside the SALMONN pipeline) + plots. |
| `introspect_one.py` | End-to-end introspection harness for one file. |
| `resample_to_16k.py` | Offline 16 kHz resampler (optional — the pipeline now resamples on the fly). |
| `salmonn_sqa/inference_config.yaml` | Model + decoding config (paths via `$SQA_ROOT`). |
| `salmonn_sqa/SALMONN/` | Vendored upstream model — **not committed**; fetched by `scripts/setup_salmonn.sh`. |
| `scripts/` | `setup_salmonn.sh`, `start_api.sh`, `run_batch_sqa.sh`, `run_sqa.sh`, `run_full_introspection.sh`, `test_api.sh`, `check_env.py`. |
| `docs/` | [deployment](docs/deployment.md) · [mlflow](docs/mlflow.md) · [introspection](docs/introspection.md) |
| `Dockerfile`, `docker-compose.yml`, `k8s/` | Containerized GPU deployment. |

## Prerequisites

- Python **3.12**, a CUDA GPU (~16 GB+ for SALMONN-7B), `git`, [`uv`](https://github.com/astral-sh/uv).
- HF access to `lmsys/vicuna-7b-v1.5`; one BEATs checkpoint is a manual download (the setup script prints the link).

## Setup

```bash
uv sync --extra experiments          # research / eval  (pyproject.toml is the source of truth)
# uv sync --extra api --extra tracking --extra viz   # the serving stack instead

scripts/setup_salmonn.sh             # clone SALMONN, apply patches, download weights, fetch NISQA
uv run python scripts/check_env.py   # reports, per tier, what's present and what's missing
```

> `mlflow` pins `pyarrow<15`, which the research code can't use — and nothing under `experiments/`
> imports it. So it lives in a `tracking` extra declared **conflicting** with `experiments`:
> install one or the other, not both in one env. Everything runs in the single `.venv`; there is
> no second interpreter.

Model/checkpoint paths in `inference_config.yaml` resolve from the `SQA_ROOT`
environment variable (defaults to the repo root; set it to `/app` in Docker).
Weights, datasets, and the `mlruns/` store are **gitignored** — never committed.

## Quickstart

**Batch a directory** (the heart of the project):
```bash
scripts/run_batch_sqa.sh /path/to/wavs cuda:0
mlflow ui     # browse descriptions, MOS, timings
```

**Serve the API:**
```bash
scripts/start_api.sh                # http://localhost:8000/docs
curl -s -X POST localhost:8000/assess -F "file=@sample.wav"
```

**One file via the CLI:**
```bash
scripts/run_sqa.sh sample.wav
```

See [docs/deployment.md](docs/deployment.md) for Docker / Kubernetes / Azure.

## Experiments & findings

The five baseline experiments characterizing the *original* SALMONN-SQA (SNR response, MOS
calibration vs PESQ/DNSMOS/NISQA, controlled degradation sweep, enhancement evaluation, prompt
steering) and the Plan B fine-tune that fixed them are logged in
**[experiments/FINDINGS.md](experiments/FINDINGS.md)**, with a standalone narrative in
**[experiments/PLAN_B_SUMMARY.md](experiments/PLAN_B_SUMMARY.md)** and a slide deck at
`experiments/results/planb_deck.pdf`.

| | |
|---|---|
| `experiments/config.py` | **every path** the experiments use, env-var driven — no hardcoded paths |
| `experiments/planb/` | corpus generation, the severity map + targets, the Opus paraphrase, evals |
| `experiments/planb/train/` | the two-stage LoRA-SFT (`run_open.sh` is the one that produced the published model) |
| `experiments/results/` | committed artifacts: plots, v3 corpora, paraphrase cache, eval outputs |

## How the assessment is prompted

The canonical prompt lives once in `salmonn_core.DEFAULT_SQA_PROMPT` (imported by
the API, batch runner, and CLI). It is **instruction-only** and contains no worked
example: an earlier prompt ended with `Example: ... {"MOS": 3.9}` and SALMONN
parroted it verbatim in ~63% of outputs, yielding no real description and a
near-constant MOS. Decoding is deterministic 4-beam search (`num_beams: 4`,
`do_sample: False`, `repetition_penalty: 1.0` — see `inference_config.yaml`);
removing the worked example, not the decoding strategy, is what stopped the
template collapse. Sampling was tried and abandoned: this checkpoint emits
occasional inf/nan logits that crash multinomial sampling.
`batch_process_sqa.py` flags degenerate
(echoed/too-short) outputs and excludes them from the aggregate MOS.

## Credits — what this work builds on

This repo does **not** train a speech-quality rater from scratch. It continues training an
existing, published SQA checkpoint, and that checkpoint's provenance materially affects how
the results here should be read.

- **The base checkpoint** (`finetuned_SALMONN_7B_2.pth`, from
  [`tsinghua-ee/Speech_Quality_Assessment`](https://huggingface.co/tsinghua-ee/Speech_Quality_Assessment))
  is the model from **Wang et al., "Enabling Auditory Large Language Models for Automatic
  Speech Quality Evaluation", ICASSP 2025** ([arXiv:2409.16644](https://arxiv.org/abs/2409.16644),
  DOI [10.1109/ICASSP49660.2025.10890590](https://doi.org/10.1109/ICASSP49660.2025.10890590)).
  It is SFT'd on **human MOS ratings** — BVCC, NISQA and SOMOS (plus VoxSim for speaker
  similarity) — and reports LCC 0.861 / SRCC 0.859 averaged over the NISQA
  LIVETALK/FOR/P501 test sets. **If you use this repo's model, cite that paper too.**
- **The architecture** is SALMONN: **Tang et al., ICLR 2024**
  ([arXiv:2310.13289](https://arxiv.org/abs/2310.13289)).

Two consequences worth stating plainly:

1. **This pipeline is not "human-label-free".** Its foundation was trained on ~100k human
   ratings. What is label-free is only the *fine-tuning* stage in this repo.
2. **The fine-tune replaced human-anchored calibration with proxy-anchored calibration.**
   Because no human-labeled test set is evaluated here, it cannot be ruled out that the base
   model is better aligned with human perception on in-domain material while this one is
   better aligned with PESQ. The reverb blindness this repo repairs is evidenced most
   strongly by the forced-choice probe and the parameter-severity ordering — not by the
   proxy-metric correlations.

> **If you evaluate this model against the base model on human labels, choose the test set
> carefully.** BVCC, NISQA and SOMOS are the base checkpoint's *training* domains, so
> NISQA_TEST_FOR/P501/LIVETALK are in-domain for it and out-of-domain for this fine-tune —
> a head-to-head there is biased toward the base model for reasons unrelated to perception.
> Note also that the QualiSpeech test split *incorporates* NISQA_TEST_FOR and P501, and is
> ~39% BVCC, so it inherits the same problem. Enhancement-focused sets outside those domains
> (TMHINT-QI, VoiceMOS noisy/enhanced tracks, URGENT) are fairer ground; NISQA-test is worth
> running as a deliberately hostile control, labelled as such.

## Known limitations / next steps

**Validity of the headline claims** (be aware of these before citing any number above):

- **No human ground truth anywhere.** Every reported correlation is against PESQ/NISQA/DNSMOS or
  against known synthesis parameters. The pipeline has never been compared to a human rating.
- **The calibration evaluation is partly circular.** PESQ/NISQA/DNSMOS supply 30% of the training
  target and are then used to score the result. The gain from ρ 0.40–0.49 to ρ 0.65–0.75 is
  therefore partly built into the supervision, and cannot be read as "better matches perception".
- **The controlled sweeps are small and single-speaker.** The degradation sweeps default to 6–8
  clips, all from VoiceBank-DEMAND speaker `p232` (`pick_ids` sorts speaker-first, so a small
  `--n-clips` draws from one speaker only). The whole VoiceBank eval has 2 speakers.
- **Point estimates only.** Correlations are reported without confidence intervals or significance
  tests, and training is single-seed. Deltas between models on n≈6 are not distinguishable from
  noise as reported. Pass `--bootstrap` to the analyses to get CIs.
- **Prompt/format confound in orig-vs-new comparisons.** The two models are evaluated under their
  own prompts, and unparseable original-model rows are dropped — so "0/108 vs 20/108 degenerate" is
  partly a format artifact and the correlations are not over identical clip sets.
- **In-distribution by construction.** The eval prompt is byte-identical to a training prompt, and
  the sweep's degradation DSP is the same code family as the training corpus generator.
- **Label distribution is narrow.** Training MOS spans 1.69–4.85 with ~51% in [2,3) and no clip
  above 4.85, so the model cannot have learned to emit MOS below ~1.7. Every axis except
  reverberation is ≥68% clean (noise: 3,494/4,000 at score 5).
- **Single-axis eval only**, although ~45% of training clips carry more than one degradation.

**Engineering:**

- The extracted MOS is best-effort (parsed from text); the descriptive output is the primary signal.
- The API has no auth / rate limiting / upload-size cap yet — fine for local use, not for public exposure (see deployment doc).
- `Dockerfile` still targets Python 3.10 / CUDA 11.8 and a `requests` healthcheck — needs alignment with this pyproject (tracked as follow-up).
