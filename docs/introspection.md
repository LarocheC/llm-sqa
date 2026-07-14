# SALMONN Model Introspection

This document describes the model-introspection subsystem for the SQA project. It
lets you peek inside SALMONN's audio pipeline, capturing intermediate activations
at every stage, decoding LLaMA embeddings back to their nearest vocabulary tokens,
and rendering visualizations. Everything can be logged to MLflow for comparison
across runs.

Paths below use `$SQA_ROOT` to refer to the project root (the directory containing
`model_introspection.py`).

---

## 1. The SALMONN pipeline

SALMONN is a multi-modal LLM that turns audio into a text quality assessment
through the following stages:

```
1. Audio input
   ├──> mel spectrogram ──> Whisper encoder ──> speech embeddings   (B, 1500, 1280)
   └──> raw waveform     ──> BEATs encoder   ──> audio embeddings    (B, T, 768)

2. LayerNorm + concatenation
   LayerNorm(speech) ++ LayerNorm(audio)  ──> combined embeddings    (B, 1500, 2048)

3. Window-based Q-Former
   split into ~0.33s windows, attend with learned query tokens
   ──> query outputs                                                 (B*num_windows, num_queries, 768)

4. Linear projection
   project Q-Former outputs into LLaMA's embedding space
   ──> projected audio embeddings                                    (B, num_windows, 4096)

5. Prompt wrapping
   embed the text prompt tokens, then concatenate
   [prompt_before] ++ [audio_embeddings] ++ [prompt_after]

6. LLaMA generation
   autoregressive decoding ──> assessment text
```

Stage notes:

- **Whisper** is trained for speech recognition and captures linguistic /
  phonetic content from the mel spectrogram. Output `(B, 1500, 1280)`:
  1500 time steps (~30s at 50 Hz) by 1280 dims.
- **BEATs** is trained for general audio understanding and captures acoustic
  properties (noise, distortion, fidelity) from the raw waveform. Output
  `(B, T, 768)`, where `T` varies with audio length.
- **LayerNorm** (`ln_speech`, `ln_audio`) centers and standardizes each encoder
  output before they are concatenated along the feature dimension
  (1280 + 768 = 2048).
- **Q-Former** compresses the long combined sequence into a small fixed-size set
  of query embeddings per window, making the sequence tractable for the LLM.
- **Linear projection** maps the 768-dim Q-Former outputs to LLaMA's 4096-dim
  embedding space so audio sits in the same space as text tokens.
- **Prompt wrapping**: for a prompt like
  `"<Speech><SpeechHere></Speech> Assess the quality"`, the `<SpeechHere>`
  placeholder is replaced by the projected audio embeddings, producing a single
  mixed sequence of text-token and audio embeddings that LLaMA decodes over.

Audio "tokens" are not discrete vocabulary items — they are continuous 4096-dim
vectors. The embedding decoder (below) finds which real tokens they sit closest
to, which is a useful interpretability lens but not a literal decoding.

---

## 2. The introspection library

Two modules implement the subsystem:

- `$SQA_ROOT/model_introspection.py` — capture, analysis, token/embedding decoding.
- `$SQA_ROOT/visualization_utils.py` — plotting.

### ModelIntrospector

Registers forward hooks on the SALMONN sub-modules, runs generation, and returns
both the generated text and a dictionary of per-stage statistics.

```python
from model_introspection import ModelIntrospector

introspector = ModelIntrospector(model)

# Runs the model, captures activations, then removes hooks.
outputs, introspection_data = introspector.generate_with_introspection(
    samples, generate_cfg, prompts=[prompt]
)

# Last activation captured at each stage:
encoder_outputs = introspector.get_encoder_outputs()   # {'whisper': ..., 'beats': ...}
qformer_out     = introspector.get_qformer_outputs()   # tensor or None
projection_out  = introspector.get_projection_output() # tensor or None
llama_embeds    = introspector.get_llama_embeddings()  # list of tensors
```

Methods:

- `generate_with_introspection(samples, generate_cfg, prompts=None)` — registers
  hooks, calls `model.generate(...)`, extracts data, removes hooks. Returns
  `(generated_text_list, introspection_data)`.
- `extract_introspection_data()` — formats captured activations into a dict keyed
  by stage name, each with `num_calls` and a list of per-call activation stats.
- `get_encoder_outputs()` — dict with `whisper` and/or `beats` (the last captured
  activation for each, raw form).
- `get_qformer_outputs()`, `get_projection_output()` — last captured activation,
  or `None` if the stage wasn't hooked/called.
- `get_llama_embeddings()` — list of every embedding-layer activation captured
  during generation (one per forward call).
- `remove_hooks()` — detach all registered hooks (called automatically by
  `generate_with_introspection`).

### What the hooks capture

`_register_hooks()` attaches an `ActivationCapture` forward hook to each
sub-module that exists on the model. Hooks are conditional, so missing components
are simply skipped:

| Capture key         | Hooked module                       | Condition |
|---------------------|-------------------------------------|-----------|
| `whisper_encoder`   | `model.speech_encoder`              | `hasattr(model, 'speech_encoder')` |
| `beats_encoder`     | `model.beats`                       | `hasattr(model, 'beats')` and `model.beats_path` |
| `ln_speech`         | `model.ln_speech`                   | present |
| `ln_audio`          | `model.ln_audio`                    | present |
| `qformer`           | `model.speech_Qformer.bert`         | present |
| `linear_projection` | `model.speech_llama_proj`           | present |
| `llama_embeddings`  | LLaMA `embed_tokens` layer          | present |

The LLaMA embedding layer path depends on whether LoRA is active:

- LoRA on:  `model.llama_model.model.model.embed_tokens`
- LoRA off: `model.llama_model.model.embed_tokens`

`ActivationCapture` stores each hook's input and output, detached and moved to
CPU to save GPU memory. Transformer-style outputs (those exposing
`last_hidden_state`) are stored as a dict with `last_hidden_state` plus optional
`hidden_states` / `attentions`; tuples and plain tensors are stored as-is.

`_analyze_activation()` reduces each captured tensor to JSON-friendly stats:
`shape`, `dtype`, `mean`, `std`, `min`, `max`, and `norm` (L2). Dicts and tuples
are summarized element-wise.

### EmbeddingDecoder

Interprets continuous embedding vectors by finding the nearest tokens in the
LLaMA vocabulary (cosine similarity by default; Euclidean also supported). It
caches the full vocab embedding matrix on construction.

```python
from model_introspection import EmbeddingDecoder

# LoRA models nest one level deeper (see hook table above).
embedding_layer = model.llama_model.model.model.embed_tokens  # or .model.embed_tokens
decoder = EmbeddingDecoder(embedding_layer, model.llama_tokenizer)

llama_embeddings = introspector.get_llama_embeddings()[-1]   # (B, seq_len, 4096)
decoded = decoder.decode_embedding_sequence(llama_embeddings[0], top_k=5)
print(decoder.decode_and_format(llama_embeddings[0], top_k=5))
```

Methods:

- `find_nearest_tokens(embedding, top_k=5, metric='cosine')` — for one embedding
  vector, returns a ranked list of dicts with `rank`, `token_id`, `token_text`,
  and either `cosine_similarity` or `euclidean_distance`.
- `decode_embedding_sequence(embeddings, top_k=3, metric='cosine')` — applies the
  above across a `(seq_len, hidden)` or `(batch, seq_len, hidden)` sequence
  (first batch only). Returns a list (per position) of top-k token lists.
- `decode_and_format(embeddings, top_k=3, metric='cosine')` — same as above but
  returns a human-readable, multi-line string.

Sample formatted output:

```
Position 0:
  1. 'The' (id=450, sim=0.8523)
  2. 'This' (id=851, sim=0.7891)
  3. 'A' (id=32, sim=0.7234)
Position 1:
  1. 'speech' (id=12456, sim=0.9123)
  2. 'audio' (id=8932, sim=0.8567)
```

### TokenAnalyzer

Helper for decoding and inspecting generated token sequences.

```python
from model_introspection import TokenAnalyzer

analyzer = TokenAnalyzer(model.llama_tokenizer)
tokens = analyzer.decode_tokens([450, 12032, 756, 1781, 7477])
info   = analyzer.analyze_generation(output_ids)   # num_tokens, tokens, token_ids, decoded_text
```

Methods: `decode_tokens(token_ids)`, `analyze_generation(output_ids)`,
`get_token_embeddings(token_ids, embedding_layer)`.

### Module-level helpers

- `create_introspection_summary(introspection_data)` — condenses the raw data
  into an ordered `pipeline_stages` list (shape + mean/std/norm per stage). Used
  to produce `introspection_summary.json`.
- `save_introspection_data(introspection_data, output_path, include_tensors=False)`
  — writes `summary.json` and `metadata.json` under `output_path`.
- `visualize_embedding_statistics(embeddings, title=...)` — returns a dict of
  embedding statistics computed across the dimension and sequence axes (no
  plotting; data only). Requires a 3D `(batch, seq_len, hidden)` tensor.

---

## 3. Visualizations produced

All plotting lives in `$SQA_ROOT/visualization_utils.py` and uses the
non-interactive `Agg` matplotlib backend. Each `plot_*` function returns a
matplotlib `Figure` and optionally saves a PNG when given `output_path`.

| Function | Produces | Reads |
|----------|----------|-------|
| `plot_activation_statistics(introspection_data, ...)` | 3-panel bar chart of mean / std / L2-norm across all captured stages | introspection data dict |
| `plot_pipeline_flow(introspection_data, ...)` | flow diagram of stage names and tensor shapes | introspection data dict |
| `plot_embedding_heatmap(embedding, ...)` | seq-position × embedding-dim heatmap (truncated to `max_seq_len=100`, `max_dim=256`) | a tensor |
| `plot_embedding_distribution(embedding, ...)` | histogram + box plot of activation values, with mean/std/min/max | a tensor |
| `plot_sequence_norms(embedding, ...)` | L2 norm per sequence position (line plot) + norm histogram | a tensor |

`create_introspection_visualizations(introspection_data, encoder_outputs, output_dir)`
is the convenience entry point. It writes, into `output_dir`:

- `activation_statistics.png`
- `pipeline_flow.png`
- per encoder (`whisper`, `beats`): `<encoder>_heatmap.png`,
  `<encoder>_distribution.png`, `<encoder>_norms.png`

It returns the list of saved paths and tolerates per-plot failures (logs and
continues).

Reading them:

- **Activation statistics** — watch for sudden spikes or stages that collapse to
  zero; norms should stay in a sane range across the pipeline.
- **Pipeline flow** — confirms shapes transform as expected (sequence compression
  at the Q-Former, 768→4096 expansion at the projection).
- **Heatmaps** — temporal (x) and per-dimension (y) activation patterns; red is
  positive, blue negative.
- **Distributions** — healthy encoder activations are roughly centered and
  bell-shaped; large outliers are suspect.
- **Sequence norms** — peaks tend to mark high-information moments, valleys
  silence.

---

## 4. How to run it

### End-to-end harness: `introspect_one.py`

`introspect_one.py` (formerly `test_full_introspection.py`) loads the model, runs
a single audio file through full introspection, decodes the LLaMA embeddings,
builds every visualization, and logs all of it to MLflow.

```bash
# Minimal
python introspect_one.py $SQA_ROOT/test_audio_samples/sample1_noisy.wav

# With options
python introspect_one.py \
    <audio_file> \
    --config salmonn_sqa/inference_config.yaml \
    --device cuda:0 \
    --prompt "Assess the speech quality in detail" \
    --embedding-top-k 10 \
    --max-decode-positions 100 \
    --experiment "My_Introspection_Test"
```

Arguments:

- `audio_file` (required) — path to the audio file to analyze.
- `--config` — model config (default `salmonn_sqa/inference_config.yaml`).
- `--device` — `cuda:0`, `cpu`, etc. (default `cuda:0`). CPU is much slower but
  avoids GPU OOM.
- `--prompt` — custom assessment prompt (optional).
- `--embedding-top-k` — nearest tokens to show per position (default 5).
- `--max-decode-positions` — max sequence positions to decode (default 50).
- `--experiment` — MLflow experiment name
  (default `SALMONN_Introspection_Debug`).

MLflow artifacts written per run:

- `assessment_output.txt` — generated quality assessment
- `embedding_decoding.json` — decoded embedding tokens
- `embeddings_decoded.txt` — human-readable decoded sequence
- `introspection_summary.json` — per-stage summary
- `introspection_detailed.json` — full per-call activation stats
- `input_audio/` — the input file
- `introspection_visualizations/` — the PNGs from section 3

Plus timing metrics (inference / embedding-decode / visualization / total) and
parameters (audio file, device, prompt, top-k, etc.).

View results:

```bash
mlflow ui   # then open http://localhost:5000
```

### Per-request via the API

The same introspection runs through `api_inference.py` when you pass
`?enable_introspection=true`:

```bash
python api_inference.py --device cuda:0

curl -X POST "http://localhost:8000/assess?enable_introspection=true" \
  -F "file=@audio.wav" \
  -F "prompt=Assess the speech quality"
```

When enabled, the endpoint logs `introspection_summary.json` and
`introspection_detailed.json` to MLflow alongside the assessment. If the
introspection dependencies are unavailable the request still succeeds — it just
skips the extra capture.

### Programmatic use

```python
from model_introspection import ModelIntrospector, EmbeddingDecoder
from visualization_utils import plot_embedding_heatmap

introspector = ModelIntrospector(model)
outputs, introspection_data = introspector.generate_with_introspection(
    samples, generate_cfg, prompts=[prompt]
)

whisper_out = introspector.get_encoder_outputs()['whisper']
plot_embedding_heatmap(
    whisper_out['last_hidden_state'],
    title="Whisper Encoder Analysis",
    output_path="custom_analysis.png",
)
```

---

## 5. Notes and troubleshooting

- **Overhead** — introspection is slower because it captures activations,
  computes statistics, renders plots, and logs to MLflow. Use it for
  debugging/analysis, not production.
- **Memory** — captured activations are detached and moved to CPU, but still cost
  extra memory. On GPU OOM, use `--device cpu` or process one file at a time.
- **Missing dependencies** — `pip install matplotlib seaborn mlflow torch transformers`.
- **No visualizations** — check MLflow logs, output-directory permissions, and
  that the matplotlib backend is `Agg` (set automatically in
  `visualization_utils.py`).
- **Missing encoder outputs** — verify the model actually has the corresponding
  sub-module; BEATs in particular is only hooked when `model.beats_path` is set.

## References

- Code: `$SQA_ROOT/model_introspection.py`, `$SQA_ROOT/visualization_utils.py`,
  `$SQA_ROOT/introspect_one.py`, `$SQA_ROOT/api_inference.py`
- Papers: SALMONN (arxiv 2310.13289), Whisper (2212.04356),
  BEATs (2212.09058), Q-Former / BLIP-2 (2301.12597)
- MLflow: https://mlflow.org/docs/latest/
