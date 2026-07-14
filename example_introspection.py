"""
Example script demonstrating model introspection usage.

This script shows how to use the introspection features to analyze
the SALMONN model's internal workings.
"""

import argparse
import requests


def introspect_via_api(audio_file: str, api_url: str = "http://localhost:8000"):
    """
    Example 1: Run introspection via API.

    Args:
        audio_file: Path to audio file
        api_url: API URL
    """
    print("=" * 60)
    print("Example 1: Introspection via API")
    print("=" * 60)

    # Make request with introspection enabled
    files = {'file': open(audio_file, 'rb')}
    params = {'enable_introspection': 'true'}

    print(f"\nProcessing: {audio_file}")
    print("Introspection enabled: True")

    response = requests.post(
        f"{api_url}/assess",
        files=files,
        params=params
    )

    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Assessment complete!")
        print(f"  MOS Score: {result.get('mos_score', 'N/A')}")
        print(f"  Processing Time: {result['processing_time']:.2f}s")
        print(f"\nAssessment (first 200 chars):")
        print(f"  {result['assessment'][:200]}...")
        print(f"\nTo view introspection data:")
        print(f"  1. Run: mlflow ui")
        print(f"  2. Open: http://localhost:5000")
        print(f"  3. Find the latest run")
        print(f"  4. View artifacts → introspection_visualizations/")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")


def compare_samples_with_introspection(
    clean_file: str,
    noisy_file: str,
    api_url: str = "http://localhost:8000"
):
    """
    Example 2: Compare clean vs noisy samples.

    Args:
        clean_file: Path to clean audio
        noisy_file: Path to noisy audio
        api_url: API URL
    """
    print("\n" + "=" * 60)
    print("Example 2: Comparing Clean vs Noisy Audio")
    print("=" * 60)

    results = {}

    for label, audio_file in [("clean", clean_file), ("noisy", noisy_file)]:
        print(f"\nProcessing {label} sample: {audio_file}")

        files = {'file': open(audio_file, 'rb')}
        params = {'enable_introspection': 'true'}

        response = requests.post(
            f"{api_url}/assess",
            files=files,
            params=params
        )

        if response.status_code == 200:
            result = response.json()
            results[label] = result
            print(f"  ✓ MOS Score: {result.get('mos_score', 'N/A')}")
            print(f"  ✓ Processing Time: {result['processing_time']:.2f}s")
        else:
            print(f"  ✗ Error: {response.status_code}")

    if len(results) == 2:
        print(f"\n{'-' * 60}")
        print("Comparison:")
        print(f"{'-' * 60}")
        clean_mos = results['clean'].get('mos_score')
        noisy_mos = results['noisy'].get('mos_score')

        if clean_mos and noisy_mos:
            diff = clean_mos - noisy_mos
            print(f"  Clean MOS:  {clean_mos:.2f}")
            print(f"  Noisy MOS:  {noisy_mos:.2f}")
            print(f"  Difference: {diff:+.2f}")
            print(f"\nView detailed introspection data in MLflow UI to understand")
            print(f"how the model's internal representations differ.")


def analyze_introspection_output():
    """
    Example 3: Analyze introspection output from MLflow.

    This shows how to programmatically access introspection data.
    """
    print("\n" + "=" * 60)
    print("Example 3: Analyzing Introspection Output")
    print("=" * 60)

    print("\nTo access introspection data programmatically:\n")

    code = """
import mlflow
import json

# Get the latest run
experiment = mlflow.get_experiment_by_name("SALMONN_API_Production")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.introspection_enabled = 'True'",
    max_results=1,
    order_by=["start_time DESC"]
)

if not runs.empty:
    run_id = runs.iloc[0]['run_id']
    print(f"Analyzing run: {run_id}")

    # Download introspection summary
    client = mlflow.tracking.MlflowClient()
    summary_path = client.download_artifacts(
        run_id, "introspection_summary.json"
    )

    # Load and analyze
    with open(summary_path) as f:
        summary = json.load(f)

    print(f"Pipeline stages: {len(summary['pipeline_stages'])}")
    for stage in summary['pipeline_stages']:
        print(f"  - {stage['name']}:")
        print(f"    Calls: {stage['num_calls']}")
        for act in stage['activations']:
            if 'shape' in act:
                print(f"    Shape: {act['shape']}")
                if 'stats' in act:
                    print(f"    Mean: {act['stats'].get('mean', 'N/A')}")
                    print(f"    Std:  {act['stats'].get('std', 'N/A')}")

    # Download visualizations
    viz_dir = client.download_artifacts(
        run_id, "introspection_visualizations"
    )
    print(f"\\nVisualizations downloaded to: {viz_dir}")
"""

    print(code)


def prompt_comparison_example(
    audio_file: str,
    api_url: str = "http://localhost:8000"
):
    """
    Example 4: Compare different prompts on the same audio.

    Args:
        audio_file: Path to audio file
        api_url: API URL
    """
    print("\n" + "=" * 60)
    print("Example 4: Comparing Different Prompts")
    print("=" * 60)

    prompts = [
        "<Speech><SpeechHere></Speech> Assess the overall speech quality.",
        "<Speech><SpeechHere></Speech> Rate the clarity of this speech.",
        "<Speech><SpeechHere></Speech> Evaluate the naturalness of the voice.",
    ]

    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: {prompt[:50]}...")

        files = {'file': open(audio_file, 'rb')}
        data = {'prompt': prompt}
        params = {'enable_introspection': 'true'}

        response = requests.post(
            f"{api_url}/assess",
            files=files,
            data=data,
            params=params
        )

        if response.status_code == 200:
            result = response.json()
            results.append({
                'prompt': prompt,
                'mos_score': result.get('mos_score'),
                'assessment': result['assessment']
            })
            print(f"  ✓ MOS Score: {result.get('mos_score', 'N/A')}")
        else:
            print(f"  ✗ Error: {response.status_code}")

    print(f"\n{'-' * 60}")
    print("Summary:")
    print(f"{'-' * 60}")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['prompt'][:40]}...")
        print(f"   MOS: {result['mos_score']}")
        print(f"   Assessment: {result['assessment'][:80]}...")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Examples of model introspection usage"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Path to audio file (optional, for running examples)"
    )
    parser.add_argument(
        "--clean-file",
        help="Path to clean audio for comparison"
    )
    parser.add_argument(
        "--noisy-file",
        help="Path to noisy audio for comparison"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API URL"
    )
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific example (1-4)"
    )

    args = parser.parse_args()

    # Show usage if no audio file provided
    if not args.audio_file and not args.example:
        print("Model Introspection Examples")
        print("=" * 60)
        print("\nUsage:")
        print("  python example_introspection.py <audio_file>")
        print("\nExamples:")
        print("  # Run all examples")
        print("  python example_introspection.py audio.wav")
        print()
        print("  # Compare clean vs noisy")
        print("  python example_introspection.py \\")
        print("    --clean-file clean.wav --noisy-file noisy.wav")
        print()
        print("  # Run specific example")
        print("  python example_introspection.py audio.wav --example 1")
        print()
        print("Available examples:")
        print("  1. Basic introspection via API")
        print("  2. Compare clean vs noisy audio")
        print("  3. Analyze introspection output programmatically")
        print("  4. Compare different prompts")
        print()
        return

    # Example 3 doesn't need audio file
    if args.example == 3 or (not args.audio_file and not args.example):
        analyze_introspection_output()
        return

    # Run examples
    if args.example == 1 or not args.example:
        if args.audio_file:
            introspect_via_api(args.audio_file, args.api_url)

    if args.example == 2 or (args.clean_file and args.noisy_file):
        if args.clean_file and args.noisy_file:
            compare_samples_with_introspection(
                args.clean_file,
                args.noisy_file,
                args.api_url
            )
        elif not args.example:
            print("\nSkipping Example 2: --clean-file and --noisy-file required")

    if args.example == 3:
        analyze_introspection_output()

    if args.example == 4 or not args.example:
        if args.audio_file:
            prompt_comparison_example(args.audio_file, args.api_url)


if __name__ == "__main__":
    main()
