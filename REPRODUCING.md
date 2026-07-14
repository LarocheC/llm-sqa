# Reproducing Plan B

Three tiers, cheapest first. **Tier 1 needs no GPU and no datasets** — everything it uses is
committed to this repo. Only go further if you need to.

| tier | what you can do | needs | time |
|---|---|---|---|
| **1** | re-run every analysis, regenerate every figure/table | nothing but the clone | ~1 min |
| **2** | evaluate the published model on your own audio | GPU (~16 GB) + base models | ~30 min setup |
| **3** | retrain (download the corpus, or regenerate it) | 24 GB GPU (+54 GB datasets only if regenerating) | ~4 h train |

Check where you stand at any point:

```bash
uv run python scripts/check_env.py
```

It prints, per tier, exactly what is present and the command that fixes anything missing.

---

## Install (all tiers)

```bash
git clone https://github.com/LarocheC/llm-sqa && cd llm-sqa
uv sync --extra experiments
```

Everything runs in that single `.venv`. There is no second interpreter.

> **A note on extras.** `mlflow` (2.9.2) pins `pyarrow<15`, which the research code cannot use.
> Nothing under `experiments/` imports mlflow, so it lives in a separate `tracking` extra that is
> declared *conflicting* with `experiments`. Install `--extra experiments` for research, or
> `--extra api --extra tracking` for the serving stack — not both in one env.

---

## Tier 1 — analysis & figures (no GPU, no datasets)

The corpora, the eval outputs, the objective metrics, and the LLM paraphrase cache are all
committed (~10 MB). So every number and plot in [`FINDINGS.md`](experiments/FINDINGS.md) and
[`PLAN_B_SUMMARY.md`](experiments/PLAN_B_SUMMARY.md) can be regenerated from a bare clone:

```bash
# Findings #1 + #2: MOS-vs-SNR and calibration vs PESQ/NISQA/DNSMOS
uv run python -m experiments.planb.eval_voicebank_v3 --analyze              # published model
uv run python -m experiments.planb.eval_voicebank_v3 --analyze --model v3   # the older one

# Finding #4: does the MOS track a denoiser's gain?
uv run python -m experiments.planb.eval_enhancement_v3

# every figure, from the committed data alone
uv run python -m experiments.planb.make_plots
```

Expected for the published model (`--model open`):

```
rho(MOS,SNR):  orig +0.372            open +0.461
calibration:   orig +0.40..0.49       open +0.65..0.71   (the metrics agree with each other at 0.72-0.82)
MOS scale:     orig 5 values (2.50-5.00)   open 63 values (1.90-4.89)
enhancement:   orig +0.03 MOS (rho +0.03)  open +1.05 MOS (rho +0.32)
```

Rebuild the slide deck (`--extra deck`): `uv run python experiments/build_deck.py`

## Tier 2 — evaluate the published model

```bash
bash scripts/setup_salmonn.sh              # SALMONN code + Whisper + Vicuna + NISQA
                                           #   (BEATs is a manual OneDrive download — the script tells you)
uv run python scripts/fetch_checkpoints.py # the published weights, ~121 MB, from the HF Hub
uv run python -m experiments.planb.eval_compare --n-clips 6
```

Weights: **https://huggingface.co/claroche1/salmonn-sqa-planb-v3** (Q-Former + projection + LoRA
only; the base models come from `setup_salmonn.sh`).

Score one file of your own:

```python
import salmonn_core as sc, json
from experiments.planb.eval_compare import load_with_ckpt
from experiments import config as cfg

sqa = load_with_ckpt(str(cfg.ckpt_v3(stage=2)), "cuda:0")
prompt = json.load(open("experiments/planb/train/test_prompt_planb.json"))["sqa_full"]
print(sc.clean_output(sc.generate_sqa(sqa, prompt=prompt, wav_path="your.wav")))
```

## Tier 3 — retrain

### Shortcut: download the corpus instead of regenerating it

The SFT corpus is published, so you can skip the 54 GB of source datasets entirely:

```python
from datasets import load_dataset
ds = load_dataset("claroche1/sqa-degraded")     # ~570 MB, 4,000 train / 300 val
```

**https://huggingface.co/datasets/claroche1/sqa-degraded** — degraded audio + per-dimension scores +
`target_text` + the exact degradation parameters. CC BY 4.0 (derivative of LibriTTS-R + MUSAN +
SLR28). Only regenerate from scratch (below) if you want to *change* the degradation taxonomy or the
severity map.

### Datasets you must supply (only if regenerating the corpus)

**Every input is public and freely downloadable.**

| dataset | used for | size | licence | where |
|---|---|---|---|---|
| LibriTTS-R (`train-clean-100`, `dev-clean`) | the clean speech the corpus is built from | ~30 GB | CC BY 4.0 | https://www.openslr.org/141/ |
| MUSAN | real noise (noise + music) | ~23 GB | CC BY 4.0 | https://www.openslr.org/17/ |
| OpenSLR SLR28 (`RIRS_NOISES`) | reverberation (RT60 + DRR) | ~1.3 GB | Apache-2.0 | `bash scripts/fetch_rirs.sh` |
| VoiceBank-DEMAND-16k | the held-out eval set | ~1 GB | CC BY 4.0 | `huggingface-cli download JacobLinCool/VoiceBank-DEMAND-16k --repo-type dataset` |

Point the code at them (defaults assume `~/data/<name>`):

```bash
export SQA_DATA_ROOT=/path/to/data          # the simple case: everything under one root
# or override individually:
export SQA_LIBRITTS_ROOT=/path/to/LibriTTS_R
export SQA_MUSAN_ROOT=/path/to/musan
export SQA_RIR_ROOT=/path/to/rirs_open
```

> **Any RIR corpus works.** RT60 and DRR are *measured* from each impulse response
> (Schroeder backward integration), not parsed from filenames, so the pipeline is not tied
> to SLR28 — point `SQA_RIR_ROOT` at any set of RIR wavs.

Every path in the pipeline resolves through [`experiments/config.py`](experiments/config.py) —
there are no hardcoded paths. The full list of `SQA_*` variables is documented at the top of that file.

### Run it

```bash
bash scripts/fetch_rirs.sh                   # the RIR bank (Apache-2.0, ~1.3 GB)
bash experiments/planb/train/run_open.sh     # corpus -> paraphrase -> two-stage LoRA-SFT
```

Five steps, ~4 h on a single 24 GB GPU: generate corpus → paraphrase descriptions → build
manifests → Stage 1 (calibration, 2 epochs) → Stage 2 (reasoning, 4 epochs).

> **You probably don't need an Anthropic API key.** The descriptions are LLM-paraphrased *per
> degradation profile*, and that cache (`experiments/results/planb/paraphrase_pool.json`, 1,029
> profiles / 4,532 paraphrases) **is committed**. A rerun with the same degradation taxonomy is a
> pure cache hit and makes **zero API calls** — verified with the key unset. You only need a key if
> you change the taxonomy or the severity map, so that new profiles appear.

### Optional: the Stage-0 probe

The experiment that justified leaving the encoders frozen — a linear probe showing the degradation
is already linearly decodable from the frozen features:

```bash
uv run python experiments/stage0_encoder_probe.py
```

---

## What is *not* reproducible from this repo

Being honest about the edges:

- **The ConvFSENet enhancer** (finding #4) is an ONNX export from a separate private repo. The
  *analysis* is fully reproducible (the enhanced-audio PESQ and both models' MOS are committed in
  `experiments/results/enhancement.jsonl` + `planb/v3_enhancement.jsonl`), but regenerating the
  enhanced audio from scratch needs that model. Set `SQA_CONVFSENET_ONNX` if you have it; any other
  enhancer would work for a qualitatively similar experiment.
- **Exact bitwise retraining.** Corpus generation is seeded (`--seed 1` / `--seed 7`) and the
  degradation parameters are deterministic, but GPU nondeterminism means a retrain will land close
  to, not identical to, the published checkpoint.
- **The v1/v2 corpora and checkpoints** are superseded and not shipped. `run_finetune.sh` (v1),
  `run_stage2_v2.sh` (v2) and `run_v3.sh` are kept for the record; **`run_open.sh`** produced the
  published model.
