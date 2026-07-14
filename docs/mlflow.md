# MLflow Tracking

MLflow records every speech-quality assessment produced by the SALMONN SQA
system: prompts, responses, MOS scores, processing times, and system
diagnostics. This lets you compare runs, monitor quality over time, and keep an
audit trail of inputs and outputs.

Throughout this doc, `$SQA_ROOT` refers to the repository root (the directory
containing `api_inference.py`).

## Quickstart

```bash
# 1. Start the API (MLflow tracking initializes automatically on startup).
python api_inference.py --config salmonn_sqa/inference_config.yaml --device cuda:0
# or: ./start_api.sh

# 2. Assess a file — this request is logged to MLflow automatically.
curl -X POST "http://localhost:8000/assess" \
  -F "file=@003_noisy.wav" \
  -F "prompt=Assess the speech quality"

# 3. View results.
mlflow ui          # then open http://localhost:5000
```

For offline / local batch processing without the API, see
[Where it's logged](#where-its-logged).

## What gets tracked

### Single assessment (`/assess`)

Each `/assess` request creates one MLflow run.

| Type | Name | Description |
|------|------|-------------|
| Param | `audio_filename` | Uploaded filename |
| Param | `file_size_bytes`, `file_size_mb` | File size |
| Param | `prompt` | Assessment prompt used |
| Param | `formatted_prompt_length` | Length of the formatted prompt |
| Param | `model` | `SALMONN-7B` |
| Param | `endpoint` | `/assess` |
| Param | `sys_before_*`, `sys_after_*` | System diagnostics before/after inference (device, CUDA availability, GPU name, GPU memory allocated/reserved/total) |
| Metric | `processing_time_seconds` | Total request-to-response time |
| Metric | `inference_time_seconds` | Time in `model.generate()` |
| Metric | `overhead_time_seconds` | File I/O, pre/post-processing |
| Metric | `mos_score` | Mean Opinion Score (1.0–5.0), when extracted |
| Metric | `gpu_memory_delta_mb` | GPU memory change during inference (GPU only) |
| Artifact | `assessment.txt` | Full LLM assessment text |
| Artifact | `original_prompt.txt` | Prompt used |
| Artifact | `result.json` | Complete result dictionary |
| Artifact | `input_audio/` | Uploaded audio file |
| Tag | `status` | `success` / `failed` |
| Tag | `task` | `speech_quality_assessment` |
| Tag | `file_extension` | `.wav`, `.mp3`, etc. |
| Tag | `has_mos_score` | Whether a MOS score was extracted |
| Tag | `error_type` | Exception type, if failed |

> **Metric naming.** The API logs `processing_time_seconds` (not
> `processing_time`). Use that exact name in search queries and exports.

### Batch via the API (`/assess-batch`)

`/assess-batch` creates a **parent run** with one **nested run per file**
(each nested run carries the single-assessment fields above).

Parent-run params: `num_files`, `batch_prompt`, `endpoint` (`/assess-batch`),
`model`, `filenames` (first 10).

Parent-run metrics: `batch_total_time_seconds`, `files_successful`,
`files_failed`, `success_rate`, `avg_processing_time_per_file`,
`total_processing_time_files`, `avg_mos_score`, `min_mos_score`,
`max_mos_score`, `std_mos_score`, `median_mos_score`.

Parent-run artifact: `batch_summary.json`.

### Local batch (`batch_process_sqa.py`)

`batch_process_sqa.py` loads the model directly (no API) and logs the whole
directory as a **single** run.

Params: `audio_directory`, `num_files`, `config_path`, `device`, `prompt`,
`introspection_enabled`. Tags: `task=batch_sqa_processing`,
`dataset=<dir name>`, `status`.

Metrics: `num_files_processed`, `num_success`, `num_failed`,
`total_time_seconds`, `avg_inference_time_seconds`,
`min_inference_time_seconds`, `max_inference_time_seconds`, `avg_mos_score`,
`min_mos_score`, `max_mos_score`, `num_files_with_mos`. Artifact:
`batch_results.json`.

## Where it's logged

There are two independent logging paths — pick based on whether the API is
running.

**Through the API.** When the API is up, every `/assess` and `/assess-batch`
call opens its own MLflow run. Nothing else is required; just send requests.

```bash
# Single file
curl -X POST "http://localhost:8000/assess" \
  -F "file=@003_noisy.wav" -F "prompt=Assess the speech quality"

# Batch
curl -X POST "http://localhost:8000/assess-batch" \
  -F "files=@a.wav" -F "files=@b.wav" -F "files=@c.wav"
```

**Offline, without the API.** `batch_process_sqa.py` loads the model in-process
and logs one run for the directory:

```bash
python batch_process_sqa.py noisy_testset_wav_16k \
  --config salmonn_sqa/inference_config.yaml \
  --device cuda:0 \
  --experiment "SALMONN_Batch_SQA_Noisy_Testset"
# or just: ./run_batch_sqa.sh
```

> **Do not point `batch_test_mlflow.py` at the instrumented API for tracking.**
> That script calls the API *and* opens its own MLflow run, so each file gets
> logged twice (once by the API, once by the script). Use it only as an API
> smoke test, not as a tracking tool. For real batch tracking use
> `/assess-batch` or `batch_process_sqa.py`.

### Configuration

Both paths honor the same environment variables:

```bash
# Experiment name (API default: SALMONN_API_Production;
# batch_process_sqa.py default: SALMONN_Batch_SQA)
export MLFLOW_EXPERIMENT_NAME="My_Experiment"

# Tracking backend (default: local file store at mlruns/)
export MLFLOW_TRACKING_URI="mlruns"
```

## Viewing in the MLflow UI

```bash
mlflow ui                 # http://localhost:5000
mlflow ui --port 5001     # if 5000 is taken
```

- **Experiments / Runs:** list, sort, and compare runs side by side.
- **Run details:** params, metrics, downloadable artifacts (audio,
  `assessment.txt`, `result.json`), and tags.
- **Compare:** select multiple runs to plot metrics together and diff params.

### Search syntax

Enter these in the UI search box (or pass as `filter_string` to
`mlflow.search_runs`):

```
metrics.mos_score > 4.0
metrics.processing_time_seconds < 5.0
tags.status = "success"
tags.file_extension = ".wav"
params.audio_filename LIKE "%noisy%"
metrics.mos_score > 3.5 AND tags.status = "success"
```

### Export to CSV

```python
import mlflow

experiment = mlflow.get_experiment_by_name("SALMONN_API_Production")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.mos_score > 4.0 and tags.status = 'success'",
)
runs[["params.audio_filename", "metrics.mos_score",
      "metrics.processing_time_seconds", "tags.status"]].to_csv(
    "results.csv", index=False)
```

## Backends

### Local file store (default)

By default MLflow writes to the `mlruns/` directory at `$SQA_ROOT/mlruns/`.

`mlruns/` is **gitignored** and must not be committed: it is regenerable and
its `meta.yaml` files embed absolute filesystem paths. If you ever need to
pin the store explicitly, use a path relative to the repo or `$SQA_ROOT`
rather than a hardcoded one:

```bash
export MLFLOW_TRACKING_URI="$SQA_ROOT/mlruns"   # or simply: ./mlruns
```

### SQLite / database

A database backend avoids the file-store deprecation warning and scales
better:

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
# or, for production:
# export MLFLOW_TRACKING_URI="postgresql://user:password@db-host:5432/mlflow"
# export MLFLOW_TRACKING_URI="mysql://user:password@db-host:3306/mlflow"
```

### Remote tracking server / Azure ML

For team collaboration, run a tracking server and point the API at it:

```bash
mlflow server \
  --backend-store-uri postgresql://user:password@db-host:5432/mlflow \
  --default-artifact-root s3://my-bucket/mlflow-artifacts \
  --host 0.0.0.0 --port 5000

export MLFLOW_TRACKING_URI="http://mlflow-server:5000"
```

On Azure ML, set the tracking URI from the workspace:

```python
from azureml.core import Workspace
import mlflow

ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
```

## Troubleshooting

**MLflow not logging.** Check the API startup logs for:

```
INFO - MLflow tracking initialized: mlruns
INFO - MLflow experiment: SALMONN_API_Production
```

If initialization warns, the API still serves requests but skips tracking.

**Runs not appearing.** Confirm the tracking URI and working directory match
where you launched `mlflow ui`:

```bash
python -c "import mlflow; print(mlflow.get_tracking_uri())"
```

**Port 5000 already in use.**

```bash
lsof -i :5000
mlflow ui --port 5001
```

**Disk space.** Audio artifacts add up. Inspect and prune:

```bash
du -sh mlruns/
mlflow experiments delete --experiment-id <id>
```

**"FileStore is deprecated."** Switch to a database backend
(`export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"`).

## Resources

- [MLflow documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow search syntax](https://mlflow.org/docs/latest/search-runs.html)
- [Azure ML + MLflow](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow)
