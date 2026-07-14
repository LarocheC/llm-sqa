"""
Batch descriptive Speech Quality Assessment over a directory of audio files,
logged to a single MLflow run.

All model loading, prompting, audio prep, inference and MOS extraction are
shared with the API through ``salmonn_core`` so the two paths cannot diverge.
Embedding introspection is OFF by default (it is a debugging aid, not part of
the assessment); enable it with ``--introspection``.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import mlflow
from tqdm import tqdm

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


def process_single_file(audio_file: Path, sqa: salmonn_core.SQAModel, prompt: str, introspector=None) -> Dict[str, Any]:
    """Run SQA on one file. Never raises — failures are returned as a result."""
    start = time.time()
    try:
        samples = prepare_audio_sample(str(audio_file), sqa.wav_processor, device=sqa.device)
        formatted = format_sqa_prompt(prompt, sqa.config.config.model.prompt_template)

        with inference_context(sqa.device):
            if introspector is not None:
                output_list, _ = introspector.generate_with_introspection(
                    samples, sqa.config.config.generate, prompts=[formatted]
                )
                output = output_list[0]
            else:
                output = sqa.model.generate(samples, sqa.config.config.generate, prompts=[formatted])[0]

        return {
            "status": "success",
            "filename": audio_file.name,
            "output": clean_output(output),
            "mos_score": extract_mos(output),
            "degenerate": is_degenerate(output),
            "inference_time": time.time() - start,
        }
    except Exception as e:
        logger.error("Failed to process %s: %s", audio_file.name, e)
        return {
            "status": "failed",
            "filename": audio_file.name,
            "error": str(e),
            "inference_time": time.time() - start,
        }


def batch_process_directory(
    audio_dir: str,
    cfg_path: str = salmonn_core.DEFAULT_CONFIG_PATH,
    device: str = "cuda:0",
    prompt: str = None,
    mlflow_experiment: str = "SALMONN_Batch_SQA",
    enable_introspection: bool = False,
    max_files: int = None,
    file_pattern: str = "*.wav",
) -> Dict[str, Any]:
    """Process every audio file under ``audio_dir`` and log results to MLflow."""
    start_time = time.time()
    prompt = prompt or DEFAULT_SQA_PROMPT

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment(mlflow_experiment)

    audio_dir = Path(audio_dir)
    audio_files = sorted(audio_dir.glob(file_pattern))
    if max_files:
        audio_files = audio_files[:max_files]
    if not audio_files:
        logger.error("No audio files found in %s", audio_dir)
        return {"status": "no_files", "directory": str(audio_dir)}

    logger.info("Found %d audio files to process", len(audio_files))
    sqa = salmonn_core.load_model(cfg_path, device)

    introspector = None
    if enable_introspection:
        from model_introspection import ModelIntrospector

        introspector = ModelIntrospector(sqa.model)
        logger.info("Introspection enabled.")

    run_name = f"batch_sqa_{audio_dir.name}_{int(time.time())}"
    with mlflow.start_run(run_name=run_name) as run:
        logger.info("MLflow Run ID: %s", run.info.run_id)
        mlflow.log_param("audio_directory", str(audio_dir))
        mlflow.log_param("num_files", len(audio_files))
        mlflow.log_param("config_path", cfg_path)
        mlflow.log_param("device", sqa.device)
        mlflow.log_param("prompt", prompt)
        mlflow.log_param("introspection_enabled", enable_introspection)
        mlflow.set_tag("task", "batch_sqa_processing")
        mlflow.set_tag("dataset", audio_dir.name)

        results = []
        mos_scores = []
        inference_times = []
        failed_files = []
        degenerate = 0

        for audio_file in tqdm(audio_files, desc="Assessing"):
            result = process_single_file(audio_file, sqa, prompt, introspector)
            results.append(result)

            if result["status"] == "success":
                inference_times.append(result["inference_time"])
                mlflow.log_text(result["output"], f"outputs/{result['filename']}.txt")
                if result["degenerate"]:
                    degenerate += 1
                elif result["mos_score"] is not None:
                    mos_scores.append(result["mos_score"])
            else:
                failed_files.append(result["filename"])

        total_time = time.time() - start_time
        num_success = sum(1 for r in results if r["status"] == "success")

        mlflow.log_metric("num_files_processed", len(audio_files))
        mlflow.log_metric("num_success", num_success)
        mlflow.log_metric("num_failed", len(failed_files))
        mlflow.log_metric("num_degenerate", degenerate)
        mlflow.log_metric("total_time_seconds", total_time)
        if inference_times:
            mlflow.log_metric("avg_inference_time_seconds", sum(inference_times) / len(inference_times))
        # avg_mos_score is computed over GENuine assessments only — never over
        # parroted/degenerate outputs (the old code averaged the leaked example).
        if mos_scores:
            mlflow.log_metric("avg_mos_score", sum(mos_scores) / len(mos_scores))
            mlflow.log_metric("min_mos_score", min(mos_scores))
            mlflow.log_metric("max_mos_score", max(mos_scores))
            mlflow.log_metric("num_files_with_mos", len(mos_scores))

        results_summary = {
            "total_files": len(audio_files),
            "successful": num_success,
            "failed": len(failed_files),
            "degenerate": degenerate,
            "failed_files": failed_files,
            "total_time_seconds": total_time,
            "avg_mos_score": sum(mos_scores) / len(mos_scores) if mos_scores else None,
            "results": results,
        }
        mlflow.log_dict(results_summary, "batch_results.json")
        mlflow.set_tag("status", "success")

        logger.info("=" * 60)
        logger.info("Done: %d/%d succeeded, %d degenerate, %d failed",
                    num_success, len(audio_files), degenerate, len(failed_files))
        if degenerate:
            logger.warning(
                "%d/%d outputs were degenerate (echoed/too-short) and excluded from avg_mos_score.",
                degenerate, num_success,
            )
        if mos_scores:
            logger.info("avg MOS (genuine only): %.3f over %d files", sum(mos_scores) / len(mos_scores), len(mos_scores))
        logger.info("MLflow Run ID: %s  (view with: mlflow ui)", run.info.run_id)

        return {"status": "success", "run_id": run.info.run_id, "summary": results_summary}


def main():
    parser = argparse.ArgumentParser(description="Batch descriptive SQA with MLflow logging")
    parser.add_argument("audio_dir", help="Directory containing audio files")
    parser.add_argument("--config", default=salmonn_core.DEFAULT_CONFIG_PATH, help="Model config path")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0, cpu, ...)")
    parser.add_argument("--prompt", default=None, help="Custom prompt (defaults to the canonical SQA prompt)")
    parser.add_argument("--experiment", default="SALMONN_Batch_SQA", help="MLflow experiment name")
    parser.add_argument("--introspection", action="store_true", help="Capture embedding internals (slower)")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files (for testing)")
    parser.add_argument("--pattern", default="*.wav", help="Glob pattern (default *.wav)")
    args = parser.parse_args()

    if not os.path.isdir(args.audio_dir):
        logger.error("Directory not found: %s", args.audio_dir)
        sys.exit(1)

    try:
        result = batch_process_directory(
            audio_dir=args.audio_dir,
            cfg_path=args.config,
            device=args.device,
            prompt=args.prompt,
            mlflow_experiment=args.experiment,
            enable_introspection=args.introspection,
            max_files=args.max_files,
            file_pattern=args.pattern,
        )
        if result["status"] == "success":
            s = result["summary"]
            logger.info("Processed %d/%d files (run %s)", s["successful"], s["total_files"], result["run_id"])
    except Exception as e:
        logger.error("Batch processing failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
