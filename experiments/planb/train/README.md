# Plan B — two-stage LoRA-SFT trainer

Fine-tunes the released SALMONN-SQA checkpoint on the Stage 1 synthetic corpus to
de-specialize it from a "lenient noise specialist" into a multi-dimensional rater
that perceives reverb / bandwidth / clipping / discontinuity and emits a
de-compressed MOS. Reuses the vendored SALMONN training harness (`train.py` +
`runner.py` + `dataset.py`) — this directory only adds the adapters.

## What trains, what's frozen

Stage 0 proved the degradation signal already survives in the frozen encoder
features, so we keep the original recipe: **Whisper + BEATs frozen; Q-Former +
speech→LLaMA projection + Vicuna-LoRA (r=8, α=28) trainable.** Training *continues
from* the released SQA checkpoint (`finetuned_SALMONN_7B_2.pth`, loaded
`strict=False`) rather than from scratch, preserving its descriptive ability.

## Two stages (calibrate-then-describe)

| | target text | task / prompt | why |
|---|---|---|---|
| **Stage 1** | the per-dimension score block only (`noise:5 reverberation:2 …`) | `sqa_score` | learn the calibration scaffold first; cheap, balanced per dimension-bin |
| **Stage 2** | full block + grounded description + `Overall MOS: X.XX`, continuing from Stage 1 | `sqa_full` | add reasoning + scale; "concise-with-num" beats free text |

The harness masks the prompt/speech tokens to `-100` and computes cross-entropy
only on the target text (`salmonn.py` forward).

## Files

- `make_manifest.py` — Plan B corpus JSONL → SALMONN `{"annotation":[…]}` manifest,
  one per stage (selects score-block vs full target + task name).
- `prompts_planb.json` / `test_prompt_planb.json` — `sqa_score` / `sqa_full` prompt
  lists with `<SpeechHere>`; `multi_prompt` samples a paraphrase per clip.
- `sqa_finetune.yaml` — single config (model block = the SQA model; single-GPU run
  block, AMP, batch 2 × accum 8 ≈ eff. batch 16, AdamW + warmup-cosine, lr 1e-5).
- **`run_v3.sh` — the pipeline that produced the published model.** Five steps: regenerate the
  corpus (with the synthetic degradation types) → LLM-paraphrase the descriptions → build
  manifests → Stage 1 (calibration, 2 epochs) → Stage 2 (reasoning, 4 epochs). ~3 h.
- `run_finetune.sh` (v1) / `run_stage2_v2.sh` (v2) — superseded, kept for the record.
  `run_finetune.sh --smoke` does 4 iters × 1 epoch per stage to prove it trains.

## Run

```bash
bash experiments/planb/train/run_v3.sh                  # the real thing (~3 h, 24 GB GPU)
bash experiments/planb/train/run_finetune.sh --smoke    # quick sanity that training works
```

Dataset roots and `SQA_ROOT` are resolved from the environment (see
[`experiments/config.py`](../../config.py) and [REPRODUCING.md](../../../REPRODUCING.md)).
**No Anthropic API key is needed** for a standard rerun: the paraphrase pool ships with the repo
and covers every profile, so step 2 is a pure cache hit.

Outputs land under a per-run timestamp dir: `…/ckpt_stage1_v3/<YYYYMMDDHHMM>/checkpoint_best.pth`
(trainable params only — LoRA + Q-Former + projection + LayerNorm: 30.2 M params; ~346 MB on disk
because the file also carries optimizer state. The *published* checkpoints are stripped to 121 MB).
Everything runs in the single uv env (`uv sync --extra experiments`); `tensorboardX` is declared
there. Single 24 GB GPU; smoke peak was ~20 GB at batch 2 — drop `batch_size_train`
to 1 if OOM. Validated end-to-end: Stage 1 trains from the SQA ckpt and Stage 2
continues from the Stage 1 `checkpoint_best.pth` (`strict=False` merge).

## Evaluate

Point `salmonn_sqa/inference_config.yaml` `model.ckpt` (or `salmonn_core.load_model`)
at the Stage 2 `checkpoint_*.pth` and rerun the existing eval suite — success =
reverb/bandwidth sweep ρ go strongly negative, calibration ρ vs PESQ/NISQA rises
toward ~0.7, enhancement MOS tracks the gain, forced-choice d′ > 0, MOS spreads
beyond 5 discrete values.

## Data lineage

- Train: `corpus_train_v3.jsonl` (4,000 clips, LibriTTS-R **train-clean-100**),
  wavs under `$SQA_CORPUS_WAV_DIR/wav_train_v3`.
- Val: `corpus_val_v3.jsonl` (300 clips, LibriTTS-R **dev-clean** — disjoint speakers),
  wavs under `$SQA_CORPUS_WAV_DIR/wav_val_v3`.
- Both leakage-safe vs the VoiceBank-DEMAND eval set (no VCTK speakers).
