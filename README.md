# SQA — Descriptive Speech Quality Assessment with an Audio LLM

Turn the **SALMONN** audio LLM into a speech-quality rater that *describes* what is wrong with a
recording — and, after the fine-tune in this repo, actually **perceives and calibrates** it.

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
| 2 | Calibration ρ 0.40–0.49; **5 distinct MOS values** | **ρ 0.65–0.71**; **63 distinct values** |
| 3 | reverb −0.27, bandwidth −0.37, clipping −0.48 | reverb −0.80, **bandwidth −0.88**, clipping −0.81 |
| 4 | Enhancement: MOS **+0.03**, ρ ≈ 0 | MOS gain **+1.05** (91% of files), ρ **+0.32** |
| 5 | Degenerate JSON under OOD input | **0/108 degenerate** |

Real reverberation (held-out **measured** RIRs, PESQ as an independent reference): ρ(MOS, PESQ)
**+0.83** (vs +0.79 for the older model).
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
near-constant MOS. Decoding uses light sampling (`do_sample=True`, low
temperature, a repetition penalty) instead of greedy beam search so outputs vary
and don't collapse onto a template. `batch_process_sqa.py` flags degenerate
(echoed/too-short) outputs and excludes them from the aggregate MOS.

## Known limitations / next steps

- The extracted MOS is best-effort (parsed from text); the descriptive output is the primary signal.
- The API has no auth / rate limiting / upload-size cap yet — fine for local use, not for public exposure (see deployment doc).
- `Dockerfile` still targets Python 3.10 / CUDA 11.8 and a `requests` healthcheck — needs alignment with this pyproject (tracked as follow-up).
