"""
Batch testing script for SALMONN Speech Quality Assessment with MLflow tracking.

This script tests multiple audio files and tracks all results using MLflow,
including LLM outputs, MOS scores, and processing metrics.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import mlflow.data
from mlflow.models import infer_signature
import pandas as pd

from client_example import SALMONNClient


def setup_mlflow(experiment_name: str = "SALMONN_Speech_Quality_Assessment"):
    """
    Setup MLflow experiment and tracking.

    Args:
        experiment_name: Name of the MLflow experiment
    """
    # Set tracking URI (local by default, can be changed to remote server)
    mlflow.set_tracking_uri("mlruns")

    # Create or get experiment
    experiment = mlflow.set_experiment(experiment_name)

    print(f"MLflow Experiment: {experiment_name}")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print()

    return experiment


def assess_audio_with_tracking(
    client: SALMONNClient,
    audio_path: Path,
    prompt: Optional[str] = None,
    run_name: Optional[str] = None
) -> Dict:
    """
    Assess a single audio file and track results with MLflow.

    Args:
        client: SALMONN API client
        audio_path: Path to audio file
        prompt: Custom assessment prompt
        run_name: Name for the MLflow run

    Returns:
        Assessment results dictionary
    """
    # Use filename as run name if not provided
    if run_name is None:
        run_name = audio_path.stem

    with mlflow.start_run(run_name=run_name, nested=True):
        # Log parameters
        mlflow.log_param("audio_file", audio_path.name)
        mlflow.log_param("audio_path", str(audio_path))
        mlflow.log_param("file_size_bytes", audio_path.stat().st_size)

        if prompt:
            mlflow.log_param("custom_prompt", prompt)

        # Perform assessment
        try:
            result = client.assess_audio(str(audio_path), prompt=prompt)

            # Log metrics
            if result.get('mos_score') is not None:
                mlflow.log_metric("mos_score", result['mos_score'])
            mlflow.log_metric("processing_time_seconds", result['processing_time'])

            # Log the LLM output as text artifact
            assessment_text = result['assessment']
            mlflow.log_text(assessment_text, "assessment.txt")

            # Log the full result as JSON
            mlflow.log_dict(result, "result.json")

            # Log the audio file as artifact
            mlflow.log_artifact(str(audio_path), "audio")

            # Tag the run
            mlflow.set_tag("status", "success")
            mlflow.set_tag("model", "SALMONN-7B")
            mlflow.set_tag("task", "speech_quality_assessment")

            # For newer MLflow versions, you can also use mlflow.llm APIs
            # Track prompt and response
            mlflow.log_param("llm_input_type", "audio")
            mlflow.log_text(assessment_text, "llm_response.txt")

            print(f"✓ {audio_path.name}: MOS={result.get('mos_score', 'N/A')}, "
                  f"Time={result['processing_time']:.2f}s")

            return result

        except Exception as e:
            # Log error
            mlflow.log_param("error", str(e))
            mlflow.set_tag("status", "failed")

            print(f"✗ {audio_path.name}: Error - {e}")

            return {
                "error": str(e),
                "audio_filename": audio_path.name
            }


def batch_assess_with_mlflow(
    audio_dir: Path,
    api_url: str = "http://localhost:8000",
    pattern: str = "*.wav",
    prompt: Optional[str] = None,
    experiment_name: str = "SALMONN_Speech_Quality_Assessment",
    run_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Batch assess multiple audio files with MLflow tracking.

    Args:
        audio_dir: Directory containing audio files
        api_url: URL of the SALMONN API
        pattern: File pattern to match (e.g., "*.wav", "*.mp3")
        prompt: Custom assessment prompt for all files
        experiment_name: MLflow experiment name
        run_name: Name for the parent run

    Returns:
        DataFrame with all results
    """
    # Setup MLflow
    experiment = setup_mlflow(experiment_name)

    # Initialize client
    client = SALMONNClient(base_url=api_url)

    # Check API health
    print("Checking API health...")
    try:
        health = client.health_check()
        if health['status'] != 'healthy':
            raise RuntimeError(f"API is not healthy: {health}")
        print(f"✓ API is healthy on {health['device']}\n")
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}")
        return pd.DataFrame()

    # Find audio files
    audio_files = sorted(Path(audio_dir).glob(pattern))

    if not audio_files:
        print(f"No audio files found matching '{pattern}' in {audio_dir}")
        return pd.DataFrame()

    print(f"Found {len(audio_files)} audio files\n")

    # Create parent run for the batch
    if run_name is None:
        run_name = f"batch_assessment_{time.strftime('%Y%m%d_%H%M%S')}"

    results = []

    with mlflow.start_run(run_name=run_name) as parent_run:
        # Log batch parameters
        mlflow.log_param("audio_directory", str(audio_dir))
        mlflow.log_param("file_pattern", pattern)
        mlflow.log_param("num_files", len(audio_files))
        mlflow.log_param("api_url", api_url)

        if prompt:
            mlflow.log_param("batch_prompt", prompt)

        # Start batch processing
        start_time = time.time()

        # Assess each file
        for i, audio_file in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] Processing {audio_file.name}...")

            result = assess_audio_with_tracking(
                client=client,
                audio_path=audio_file,
                prompt=prompt,
                run_name=audio_file.stem
            )

            results.append(result)

        # Calculate aggregate metrics
        total_time = time.time() - start_time

        mos_scores = [r['mos_score'] for r in results if r.get('mos_score') is not None]
        processing_times = [r['processing_time'] for r in results if 'processing_time' in r]

        # Log aggregate metrics
        mlflow.log_metric("total_processing_time", total_time)
        mlflow.log_metric("files_processed", len(audio_files))
        mlflow.log_metric("files_succeeded", len([r for r in results if 'error' not in r]))
        mlflow.log_metric("files_failed", len([r for r in results if 'error' in r]))

        if mos_scores:
            mlflow.log_metric("avg_mos_score", sum(mos_scores) / len(mos_scores))
            mlflow.log_metric("min_mos_score", min(mos_scores))
            mlflow.log_metric("max_mos_score", max(mos_scores))
            mlflow.log_metric("std_mos_score", pd.Series(mos_scores).std())

        if processing_times:
            mlflow.log_metric("avg_processing_time", sum(processing_times) / len(processing_times))

        # Create summary DataFrame
        df = pd.DataFrame(results)

        # Save summary as CSV artifact
        summary_path = "batch_results.csv"
        df.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path)
        os.remove(summary_path)

        # Log summary statistics as JSON
        summary_stats = {
            "total_files": len(audio_files),
            "successful": len([r for r in results if 'error' not in r]),
            "failed": len([r for r in results if 'error' in r]),
            "total_time": total_time,
            "mos_statistics": {
                "mean": float(sum(mos_scores) / len(mos_scores)) if mos_scores else None,
                "min": float(min(mos_scores)) if mos_scores else None,
                "max": float(max(mos_scores)) if mos_scores else None,
                "std": float(pd.Series(mos_scores).std()) if mos_scores else None,
            } if mos_scores else None
        }

        mlflow.log_dict(summary_stats, "summary_statistics.json")

        print(f"\n{'='*60}")
        print(f"Batch Assessment Complete")
        print(f"{'='*60}")
        print(f"Total files: {len(audio_files)}")
        print(f"Successful: {summary_stats['successful']}")
        print(f"Failed: {summary_stats['failed']}")
        print(f"Total time: {total_time:.2f}s")

        if mos_scores:
            print(f"\nMOS Score Statistics:")
            print(f"  Mean: {summary_stats['mos_statistics']['mean']:.2f}")
            print(f"  Min:  {summary_stats['mos_statistics']['min']:.2f}")
            print(f"  Max:  {summary_stats['mos_statistics']['max']:.2f}")
            print(f"  Std:  {summary_stats['mos_statistics']['std']:.2f}")

        print(f"\nMLflow Run ID: {parent_run.info.run_id}")
        print(f"View results: mlflow ui")
        print(f"{'='*60}\n")

    return df


def compare_prompts(
    audio_file: Path,
    prompts: List[str],
    api_url: str = "http://localhost:8000",
    experiment_name: str = "SALMONN_Prompt_Comparison"
) -> pd.DataFrame:
    """
    Compare different prompts on the same audio file.

    Args:
        audio_file: Path to audio file
        prompts: List of prompts to test
        api_url: URL of the SALMONN API
        experiment_name: MLflow experiment name

    Returns:
        DataFrame with comparison results
    """
    setup_mlflow(experiment_name)
    client = SALMONNClient(base_url=api_url)

    results = []

    run_name = f"prompt_comparison_{audio_file.stem}_{time.strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("audio_file", str(audio_file))
        mlflow.log_param("num_prompts", len(prompts))

        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] Testing prompt: {prompt[:50]}...")

            result = assess_audio_with_tracking(
                client=client,
                audio_path=audio_file,
                prompt=prompt,
                run_name=f"prompt_{i}"
            )

            result['prompt'] = prompt
            result['prompt_index'] = i
            results.append(result)

        # Save comparison
        df = pd.DataFrame(results)
        df.to_csv("prompt_comparison.csv", index=False)
        mlflow.log_artifact("prompt_comparison.csv")
        os.remove("prompt_comparison.csv")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Batch test SALMONN API with MLflow tracking"
    )
    parser.add_argument(
        "audio_dir",
        type=str,
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--pattern",
        default="*.wav",
        help="File pattern to match (default: *.wav)"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="SALMONN API URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--prompt",
        help="Custom assessment prompt"
    )
    parser.add_argument(
        "--experiment",
        default="SALMONN_Speech_Quality_Assessment",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--run-name",
        help="Name for the MLflow run"
    )

    args = parser.parse_args()

    # Run batch assessment
    df = batch_assess_with_mlflow(
        audio_dir=Path(args.audio_dir),
        api_url=args.api_url,
        pattern=args.pattern,
        prompt=args.prompt,
        experiment_name=args.experiment,
        run_name=args.run_name
    )

    # Display results
    if not df.empty:
        print("\nResults Summary:")
        print(df[['audio_filename', 'mos_score', 'processing_time']].to_string())


if __name__ == "__main__":
    main()


# ============================================================================
# Usage Examples
# ============================================================================

"""
Example 1: Batch assess all WAV files in a directory
-----------------------------------------------------

python batch_test_mlflow.py ./audio_samples

Example 2: Custom prompt
------------------------

python batch_test_mlflow.py ./audio_samples \
  --prompt "Rate the speech quality on a scale of 1-5"

Example 3: Different file pattern
----------------------------------

python batch_test_mlflow.py ./audio_samples \
  --pattern "*.mp3" \
  --experiment "MP3_Quality_Test"

Example 4: Programmatic usage
------------------------------

from batch_test_mlflow import batch_assess_with_mlflow
from pathlib import Path

df = batch_assess_with_mlflow(
    audio_dir=Path("./audio_samples"),
    pattern="*.wav",
    experiment_name="My_Experiment"
)

print(df)

Example 5: Compare different prompts
-------------------------------------

from batch_test_mlflow import compare_prompts
from pathlib import Path

prompts = [
    "Assess the speech quality",
    "Rate the naturalness of this speech",
    "Evaluate clarity and intelligibility"
]

df = compare_prompts(
    audio_file=Path("test.wav"),
    prompts=prompts
)

Example 6: View MLflow UI
--------------------------

After running the batch test, view results:

mlflow ui

Then open http://localhost:5000 in your browser
"""
