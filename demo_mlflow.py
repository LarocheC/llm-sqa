"""
Quick demo script to test MLflow tracking with SALMONN API.

This is a simplified example showing how to track LLM outputs.
"""

import mlflow
from pathlib import Path
from client_example import SALMONNClient


def demo_single_file_tracking():
    """Demo tracking a single audio assessment."""
    print("=" * 60)
    print("Demo: Single File Assessment with MLflow Tracking")
    print("=" * 60)

    # Setup MLflow
    mlflow.set_experiment("SALMONN_Demo")

    # Initialize client
    client = SALMONNClient(base_url="http://localhost:8000")

    # Assess audio with tracking
    audio_file = "test_audio_samples/sample1_noisy.wav"

    with mlflow.start_run(run_name="demo_assessment"):
        # Log parameters
        mlflow.log_param("audio_file", audio_file)
        mlflow.log_param("model", "SALMONN-7B")

        print(f"\nAssessing: {audio_file}")
        result = client.assess_audio(audio_file)

        # Log metrics
        mlflow.log_metric("mos_score", result['mos_score'])
        mlflow.log_metric("processing_time", result['processing_time'])

        # Log LLM output
        mlflow.log_text(result['assessment'], "llm_output.txt")

        # Log full result
        mlflow.log_dict(result, "full_result.json")

        # Log audio file
        mlflow.log_artifact(audio_file, "input_audio")

        print(f"\n✓ Assessment complete!")
        print(f"  MOS Score: {result['mos_score']}")
        print(f"  Processing Time: {result['processing_time']:.2f}s")
        print(f"\nAssessment:")
        print(f"  {result['assessment'][:200]}...")

    print(f"\n{'='*60}")
    print("Run 'mlflow ui' to view results in your browser")
    print("=" * 60)


if __name__ == "__main__":
    demo_single_file_tracking()
