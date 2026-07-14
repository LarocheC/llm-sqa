"""
FastAPI server for SALMONN descriptive Speech Quality Assessment.

Endpoints:
  GET  /            API info
  GET  /health      health check
  POST /assess      assess one audio file (optional ?enable_introspection=true)
  POST /assess-batch  assess several files under a single MLflow run

Model loading, the SQA prompt, audio preprocessing, prompt formatting, MOS
extraction and the inference context all live in ``salmonn_core`` so this server
and ``batch_process_sqa.py`` cannot diverge.
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import mlflow
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

import salmonn_core
from salmonn_core import (
    DEFAULT_SQA_PROMPT,
    clean_output,
    extract_mos,
    format_sqa_prompt,
    inference_context,
    is_degenerate,
    prepare_audio_sample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Optional introspection (matplotlib/seaborn). Degrades gracefully if absent.
try:
    from model_introspection import ModelIntrospector, create_introspection_summary
    from visualization_utils import create_introspection_visualizations

    INTROSPECTION_AVAILABLE = True
except ImportError as e:  # pragma: no cover - optional dependency
    logger.warning("Model introspection not available: %s", e)
    INTROSPECTION_AVAILABLE = False

# Module-level model state, populated on startup.
model = None
wav_processor = None
config = None
device = None
model_introspector = None

MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "SALMONN_API_Production")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: float


class AssessmentResponse(BaseModel):
    assessment: str
    mos_score: Optional[float] = None
    processing_time: float
    audio_filename: str


def get_system_diagnostics() -> dict:
    """GPU memory / device diagnostics for MLflow logging."""
    diagnostics = {
        "device": str(device) if device else "unknown",
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available() and device and "cuda" in str(device):
        try:
            idx = 0 if ":" not in str(device) else int(str(device).split(":")[1])
            diagnostics.update(
                {
                    "gpu_name": torch.cuda.get_device_name(idx),
                    "gpu_memory_allocated_mb": torch.cuda.memory_allocated(idx) / 1024**2,
                    "gpu_memory_reserved_mb": torch.cuda.memory_reserved(idx) / 1024**2,
                    "gpu_memory_total_mb": torch.cuda.get_device_properties(idx).total_memory / 1024**2,
                }
            )
        except Exception as e:  # pragma: no cover - diagnostics best-effort
            logger.warning("Failed to get GPU diagnostics: %s", e)
            diagnostics["gpu_error"] = str(e)
    return diagnostics


def load_model(cfg_path: str, device_name: str = "cuda:0"):
    """Load the model into module globals via the shared core."""
    global model, wav_processor, config, device, model_introspector

    bundle = salmonn_core.load_model(cfg_path, device_name)
    model, wav_processor, config, device = (
        bundle.model,
        bundle.wav_processor,
        bundle.config,
        bundle.device,
    )

    if INTROSPECTION_AVAILABLE:
        try:
            model_introspector = ModelIntrospector(model)
            logger.info("Model introspection tools initialized.")
        except Exception as e:
            logger.warning("Failed to initialize introspection tools: %s", e)
            model_introspector = None


def _infer(audio_path: str, prompt: str, enable_introspection: bool) -> dict:
    """Run SQA on a local audio file. No MLflow side effects — the caller logs.

    Returns a dict with the assessment text, MOS, timings, diagnostics, and any
    introspection payload.
    """
    start = time.time()
    diagnostics_before = get_system_diagnostics()

    samples = prepare_audio_sample(audio_path, wav_processor, device=device)
    formatted_prompt = format_sqa_prompt(prompt, config.config.model.prompt_template)

    use_intro = bool(enable_introspection and INTROSPECTION_AVAILABLE and model_introspector)
    introspection_data = None
    encoder_outputs = None

    inference_start = time.time()
    with inference_context(device):
        if use_intro:
            output_list, introspection_data = model_introspector.generate_with_introspection(
                samples, config.config.generate, prompts=[formatted_prompt]
            )
            output = output_list[0]
        else:
            output = model.generate(samples, config.config.generate, prompts=[formatted_prompt])[0]
    if use_intro:
        encoder_outputs = model_introspector.get_encoder_outputs()
    inference_time = time.time() - inference_start

    return {
        "assessment": output,
        "mos_score": extract_mos(output),
        "inference_time": inference_time,
        "processing_time": time.time() - start,
        "diagnostics_before": diagnostics_before,
        "diagnostics_after": get_system_diagnostics(),
        "introspection_data": introspection_data,
        "encoder_outputs": encoder_outputs,
    }


app = FastAPI(
    title="SALMONN Speech Quality Assessment API",
    description="Descriptive speech quality assessment using SALMONN.",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup and initialize MLflow."""
    cfg_path = os.getenv("CONFIG_PATH", salmonn_core.DEFAULT_CONFIG_PATH)
    device_name = os.getenv("DEVICE", "cuda:0")

    if not os.path.exists(cfg_path):
        raise RuntimeError(f"Config file not found: {cfg_path}")

    load_model(cfg_path, device_name)

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info("MLflow tracking: %s (experiment=%s)", MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME)
    except Exception as e:
        logger.warning("Failed to initialize MLflow, continuing without it: %s", e)


@app.get("/", response_model=dict)
async def root():
    return {
        "name": "SALMONN Speech Quality Assessment API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/assess": "POST — assess one audio file",
            "/assess-batch": "POST — assess multiple audio files",
            "/docs": "Swagger UI",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device),
        timestamp=time.time(),
    )


@app.post("/assess", response_model=AssessmentResponse)
async def assess_speech_quality(
    file: UploadFile = File(..., description="Audio file (WAV recommended)"),
    prompt: Optional[str] = Form(default=DEFAULT_SQA_PROMPT, description="Assessment prompt"),
    enable_introspection: bool = Query(default=False, description="Capture model internals (slower)"),
):
    """Assess speech quality for one audio file and return a descriptive analysis + MOS."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    temp_path = None
    run_name = f"assess_{file.filename}_{int(time.time())}"
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix or ".wav") as tf:
            tf.write(content)
            temp_path = tf.name

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("audio_filename", file.filename)
            mlflow.log_param("file_size_mb", round(len(content) / 1024**2, 2))
            mlflow.log_param("prompt", prompt)
            mlflow.log_param("model", "SALMONN-7B")
            mlflow.log_param("endpoint", "/assess")

            logger.info("Processing %s", file.filename)
            result = _infer(temp_path, prompt, enable_introspection)
            output = result["assessment"]
            mos_score = result["mos_score"]

            mlflow.log_metric("processing_time_seconds", result["processing_time"])
            mlflow.log_metric("inference_time_seconds", result["inference_time"])
            if mos_score is not None:
                mlflow.log_metric("mos_score", mos_score)
            mlflow.log_params(
                {f"sys_after_{k}": v for k, v in result["diagnostics_after"].items() if isinstance(v, (str, int, float, bool))}
            )

            mlflow.log_text(clean_output(output), "assessment.txt")
            mlflow.set_tag("status", "success")
            mlflow.set_tag("task", "speech_quality_assessment")
            mlflow.set_tag("has_mos_score", str(mos_score is not None))
            mlflow.set_tag("degenerate_output", str(is_degenerate(output)))
            try:
                mlflow.log_artifact(temp_path, "input_audio")
            except Exception as e:
                logger.warning("Failed to log audio artifact: %s", e)

            _log_introspection(result)

            return AssessmentResponse(
                assessment=clean_output(output),
                mos_score=mos_score,
                processing_time=result["processing_time"],
                audio_filename=file.filename,
            )

    except HTTPException:
        raise
    except Exception as e:
        try:
            if mlflow.active_run():
                mlflow.set_tag("status", "failed")
                mlflow.log_param("error", str(e))
        except Exception:
            pass
        logger.error("Error processing audio: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning("Failed to delete temp file: %s", e)


def _log_introspection(result: dict):
    """Log introspection summary + visualizations if present."""
    introspection_data = result.get("introspection_data")
    if not introspection_data:
        return
    try:
        mlflow.log_dict(create_introspection_summary(introspection_data), "introspection_summary.json")
        encoder_outputs = result.get("encoder_outputs")
        if encoder_outputs:
            with tempfile.TemporaryDirectory() as viz_dir:
                for viz_path in create_introspection_visualizations(
                    introspection_data, encoder_outputs, Path(viz_dir)
                ):
                    mlflow.log_artifact(str(viz_path), "introspection_visualizations")
    except Exception as e:
        logger.error("Failed to log introspection data: %s", e, exc_info=True)


@app.post("/assess-batch")
async def assess_batch(
    files: list[UploadFile] = File(..., description="Audio files"),
    prompt: Optional[str] = Form(default=DEFAULT_SQA_PROMPT, description="Assessment prompt"),
):
    """Assess several files under a SINGLE MLflow run (no nested per-file runs)."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    batch_start = time.time()
    run_name = f"batch_assess_{len(files)}_files_{int(time.time())}"
    results = []
    mos_scores = []
    processing_times = []
    successful = degenerate = failed = 0

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("num_files", len(files))
        mlflow.log_param("prompt", prompt)
        mlflow.log_param("endpoint", "/assess-batch")

        for file in files:
            temp_path = None
            try:
                content = await file.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix or ".wav") as tf:
                    tf.write(content)
                    temp_path = tf.name

                result = _infer(temp_path, prompt, enable_introspection=False)
                output = result["assessment"]
                mlflow.log_text(clean_output(output), f"outputs/{file.filename}.txt")

                if is_degenerate(output):
                    degenerate += 1
                elif result["mos_score"] is not None:
                    mos_scores.append(result["mos_score"])
                processing_times.append(result["processing_time"])
                successful += 1
                results.append(
                    {
                        "filename": file.filename,
                        "success": True,
                        "mos_score": result["mos_score"],
                        "assessment": clean_output(output),
                    }
                )
            except Exception as e:
                failed += 1
                results.append({"filename": file.filename, "success": False, "error": str(e)})
                logger.error("Failed on %s: %s", file.filename, e)
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass

        mlflow.log_metric("batch_total_time_seconds", time.time() - batch_start)
        mlflow.log_metric("files_successful", successful)
        mlflow.log_metric("files_failed", failed)
        mlflow.log_metric("files_degenerate", degenerate)
        if processing_times:
            mlflow.log_metric("avg_processing_time_per_file", sum(processing_times) / len(processing_times))
        if mos_scores:  # average over genuine (non-degenerate) assessments only
            mlflow.log_metric("avg_mos_score", sum(mos_scores) / len(mos_scores))
            mlflow.log_metric("min_mos_score", min(mos_scores))
            mlflow.log_metric("max_mos_score", max(mos_scores))
            mlflow.log_metric("num_files_with_mos", len(mos_scores))
        mlflow.set_tag("status", "completed")
        mlflow.set_tag("task", "batch_speech_quality_assessment")

    return {"results": results, "total": len(files), "degenerate": degenerate}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SALMONN SQA API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=salmonn_core.DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    os.environ["CONFIG_PATH"] = args.config
    os.environ["DEVICE"] = args.device
    uvicorn.run("api_inference:app", host=args.host, port=args.port, reload=args.reload)
