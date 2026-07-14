# SALMONN Speech Quality Assessment API — Deployment Guide

A FastAPI service that wraps the SALMONN-based descriptive Speech Quality
Assessment (SQA) model. Given an audio file, it returns a natural-language
quality assessment plus a numeric MOS score.

## Table of Contents

1. [Overview](#overview)
2. [API Endpoints](#api-endpoints)
3. [Running Locally](#running-locally)
4. [Docker / docker-compose](#docker--docker-compose)
5. [Kubernetes / Azure Options](#kubernetes--azure-options)
6. [Monitoring, Security, and Cost](#monitoring-security-and-cost)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### Architecture

```
Audio file
   │
   ▼
FastAPI application (api_inference.py)
   │
   ▼
SALMONN model (~7B params)
   ├─ Whisper (speech encoder)
   ├─ BEATs (audio encoder)
   ├─ Vicuna-7B (LLM)
   └─ LoRA adapters
   │
   ▼
JSON response (assessment + MOS score)
```

### Key files

| File | Purpose |
|------|---------|
| `api_inference.py` | FastAPI application wrapping SALMONN inference |
| `client_example.py` | Python client library (`SALMONNClient`) with usage examples |
| `requirements-api.txt` | Python dependencies for the API |
| `start_api.sh` / `test_api.sh` | Local start and smoke-test scripts |
| `Dockerfile` / `docker-compose.yml` / `.dockerignore` | Container build and run |
| `k8s/` | Kubernetes manifests (deployment, service, pvc, configmap, hpa) |
| `salmonn_sqa/inference_config.yaml` | Model and checkpoint paths + inference settings |

### Model code and weights are not in the repo

The SALMONN model code and its weights (Vicuna-7B, Whisper-large-v2, BEATs,
and the fine-tuned SALMONN checkpoint) are **not committed** to this
repository. They are gitignored and fetched separately via:

```bash
./scripts/setup_salmonn.sh
```

Run this once before starting the API. It populates `salmonn_sqa/models/`
(encoders + LLM) and `salmonn_sqa/ckpt/` (the fine-tuned checkpoint).

### Paths are driven by `SQA_ROOT`

The model and checkpoint paths in `salmonn_sqa/inference_config.yaml` are
resolved relative to the `SQA_ROOT` environment variable, which **defaults to
the repository root**. Set it explicitly only if you run from a different
working directory or mount the repo elsewhere (e.g. in a container):

```bash
export SQA_ROOT=/path/to/sqa   # defaults to repo root if unset
```

Config entries reference `$SQA_ROOT/salmonn_sqa/...` rather than hardcoded
absolute paths, for example:

```yaml
model:
  llama_path:   "$SQA_ROOT/salmonn_sqa/models/vicuna-7b-v1_5"
  whisper_path: "$SQA_ROOT/salmonn_sqa/models/whisper-large-v2"
  beats_path:   "$SQA_ROOT/salmonn_sqa/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
  ckpt:         "$SQA_ROOT/salmonn_sqa/ckpt/finetuned_SALMONN_7B_2.pth"
```

### System requirements

- Python 3.10+
- CUDA 11.8+ for GPU inference (CPU inference is possible but slow)
- NVIDIA GPU with 16GB+ VRAM (T4, V100, or A100)
- ~20GB disk for model weights, 16GB+ RAM

Performance (single GPU): model loads once at startup (~30–60s); inference is
~2–5s per file depending on length and GPU; throughput is roughly 10–20
requests/minute, bounded by GPU memory for concurrent requests.

---

## API Endpoints

Base URL (local): `http://localhost:8000`. Interactive Swagger docs are served
at `http://localhost:8000/docs`.

### `GET /`

Returns API metadata (name, version, and a map of available endpoints).

### `GET /health`

Returns service status and loaded-model information.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "timestamp": 1699564800.0
}
```

### `POST /assess`

Assess a single audio file. `multipart/form-data`:

- `file` (required) — audio file (WAV recommended; MP3 etc. also accepted)
- `prompt` (optional form field) — custom assessment prompt; a default is used
  if omitted
- `enable_introspection` (optional query param, default `false`) — emit detailed
  model-internals logging; slower, intended for debugging

```bash
# Default prompt
curl -X POST "http://localhost:8000/assess" \
  -F "file=@audio.wav"

# Custom prompt
curl -X POST "http://localhost:8000/assess" \
  -F "file=@audio.wav" \
  -F "prompt=Rate the speech quality from 1-5."
```

Response:

```json
{
  "assessment": "The speech demonstrates good clarity with natural prosody...",
  "mos_score": 3.9,
  "processing_time": 2.34,
  "audio_filename": "audio.wav"
}
```

### `POST /assess-batch`

Assess multiple files in one request. `multipart/form-data` with repeated
`files` fields plus an optional `prompt`.

```bash
curl -X POST "http://localhost:8000/assess-batch" \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav"
```

### Python client

`client_example.py` provides `SALMONNClient` (`health_check`, `assess_audio`,
`assess_batch`, `wait_for_ready`):

```python
from client_example import SALMONNClient

client = SALMONNClient(base_url="http://localhost:8000")

health = client.health_check()
print(f"Status: {health['status']}")

result = client.assess_audio("audio.wav")
print(f"MOS Score: {result['mos_score']}")
print(f"Assessment: {result['assessment']}")

# Batch
results = client.assess_batch(["audio1.wav", "audio2.wav"])
```

Or with `requests` directly:

```python
import requests

with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/assess",
        files={"file": f},
        data={"prompt": "Rate the speech quality from 1-5."},  # optional
    )
result = response.json()
print(result["mos_score"], result["assessment"])
```

---

## Running Locally

1. Fetch the model code and weights (one time):

   ```bash
   ./scripts/setup_salmonn.sh
   ```

2. Install API dependencies:

   ```bash
   pip install -r requirements-api.txt
   ```

3. Start the server (or use the convenience script `./start_api.sh`):

   ```bash
   python api_inference.py \
     --host 0.0.0.0 \
     --port 8000 \
     --config salmonn_sqa/inference_config.yaml \
     --device cuda:0
   ```

4. Verify it is up:

   ```bash
   ./test_api.sh
   # or
   curl http://localhost:8000/health
   ```

The API is then available at `http://localhost:8000` (docs at `/docs`,
health at `/health`). For CPU-only inference, pass `--device cpu`.

---

## Docker / docker-compose

The repo ignores model weights, so mount `salmonn_sqa/models/` and
`salmonn_sqa/ckpt/` into the container — they are not baked into the image.
Run `./scripts/setup_salmonn.sh` on the host first.

### docker-compose (recommended)

```bash
docker-compose up --build

# Verify
curl http://localhost:8000/health
```

### Manual docker build/run

```bash
docker build -t salmonn-sqa-api:latest .

docker run -d \
  --name salmonn-api \
  --gpus all \
  -p 8000:8000 \
  -v "$SQA_ROOT/salmonn_sqa/models:/app/salmonn_sqa/models:ro" \
  -v "$SQA_ROOT/salmonn_sqa/ckpt:/app/salmonn_sqa/ckpt:ro" \
  -v "$SQA_ROOT/salmonn_sqa/inference_config.yaml:/app/salmonn_sqa/inference_config.yaml:ro" \
  -e CONFIG_PATH=/app/salmonn_sqa/inference_config.yaml \
  -e DEVICE=cuda:0 \
  salmonn-sqa-api:latest
```

`$SQA_ROOT` defaults to the repo root; substitute `$(pwd)` when invoking from
the repository directory. View logs with `docker logs -f salmonn-api` (or
`docker-compose logs -f`).

---

## Kubernetes / Azure Options

The `k8s/` directory holds manifests for an AKS deployment with GPU support:
`deployment.yaml`, `service.yaml` (LoadBalancer + ClusterIP), `pvc.yaml`
(model volumes), `configmap.yaml`, and `hpa.yaml` (autoscaler). See
`k8s/README.md` for the full walkthrough.

Choose a target by scale and operational preference:

| Option | Best for | GPU notes |
|--------|----------|-----------|
| Azure Container Instances (ACI) | Quickest one-off deploy | K80 only — likely too slow for the 7B model |
| Azure VM with GPU | Simplest migration | Pick any NC-series SKU (e.g. NC6s_v3 / V100) |
| Azure Machine Learning | Managed production endpoints | Online endpoint with `/health` liveness/readiness, `/assess` scoring |
| Azure Kubernetes Service (AKS) | High scale / autoscaling | GPU node pool + NVIDIA device plugin |
| Azure Batch | Bulk/offline batch jobs | Use `/assess-batch` per job |

### Azure VM with GPU (simplest migration)

```bash
az vm create \
  --resource-group salmonn-rg --name salmonn-vm \
  --image Ubuntu2204 --size Standard_NC6s_v3 \
  --admin-username azureuser --generate-ssh-keys

az vm open-port --resource-group salmonn-rg --name salmonn-vm --port 8000

# On the VM: install Docker + NVIDIA Container Toolkit, copy repo + weights,
# then: docker-compose up -d
```

### Azure Kubernetes Service (high scale)

```bash
az aks create \
  --resource-group salmonn-rg --name salmonn-aks \
  --node-count 1 --node-vm-size Standard_NC6s_v3 \
  --enable-addons monitoring --generate-ssh-keys

az aks get-credentials --resource-group salmonn-rg --name salmonn-aks

# Install the NVIDIA device plugin, then apply the manifests in k8s/
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
kubectl apply -f k8s/
```

### Azure Machine Learning (managed endpoint)

Build and push the image to an Azure Container Registry, create an AML
workspace and a GPU compute target (e.g. `Standard_NC6s_v3`), then define a
managed online endpoint and deployment. Point the inference config at the API
routes: `/health` for liveness and readiness, `/assess` for scoring on port
8000. See the
[AML online endpoints docs](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints).

---

## Monitoring, Security, and Cost

### Monitoring

Every response includes a `processing_time` metric. For production, add
Azure Application Insights, Prometheus + Grafana, or Azure Monitor for request
rate, latency, and error tracking.

### Scaling

- Vertical: larger GPU VMs (NC12s_v3, NC24s_v3).
- Horizontal: multiple instances behind a load balancer (HPA on AKS).
- Throughput: use `/assess-batch` for multiple files; consider a queue
  (Redis/RabbitMQ) for long-running async jobs.

### Security (hardening for production)

The current service is suitable for development and testing. Before exposing
it publicly:

1. Add authentication (API key or OAuth/JWT).
2. Add rate limiting to prevent abuse.
3. Validate uploaded file types, sizes, and content.
4. Terminate TLS / serve over HTTPS.
5. Restrict the network: private endpoints and firewall rules.
6. Store secrets in Azure Key Vault.

### Cost (Azure, approximate)

| Option | VM/SKU | GPU | Cost/Hour | Cost/Month* |
|--------|--------|-----|-----------|-------------|
| ACI | K80 | 1× K80 | $0.50 | $360 |
| VM | NC6s_v3 | 1× V100 | $3.06 | $2,203 |
| VM | NC4as_T4_v3 | 1× T4 | $0.53 | $382 |
| AKS | NC6s_v3 | 1× V100 | $3.06 | $2,203 |
| Azure ML | NC6s_v3 | 1× V100 | $3.50 | $2,520 |

*24/7 operation. Reduce cost with auto-shutdown when idle, Spot VMs (60–80%
savings), reserved instances (40–70% savings), and right-sizing (start on T4,
scale up only if needed).

---

## Troubleshooting

**Models not found / fail to load**
Run `./scripts/setup_salmonn.sh`, then confirm the weights exist and the
config paths resolve:

```bash
ls -lh salmonn_sqa/models/
ls -lh salmonn_sqa/ckpt/
cat salmonn_sqa/inference_config.yaml   # check $SQA_ROOT-based paths
```

**CUDA out of memory**
Reduce concurrent requests, or run on CPU: `python api_inference.py --device cpu`.

**Slow inference**
Confirm the GPU is in use (`nvidia-smi`), use a faster GPU (V100 → A100), and
consider int8 quantization.

**API not starting**
Check logs for errors, confirm dependencies are installed, and ensure port
8000 is free.

---

This API wrapper follows the same license as SALMONN (Apache 2.0). For model
details, see the [SALMONN repository](https://github.com/bytedance/SALMONN).
