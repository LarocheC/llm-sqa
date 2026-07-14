"""
Python client example for SALMONN Speech Quality Assessment API.

This module provides a simple client class for interacting with the API.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests


class SALMONNClient:
    """Client for SALMONN Speech Quality Assessment API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 60):
        """
        Initialize the SALMONN API client.

        Args:
            base_url: Base URL of the API (default: http://localhost:8000)
            timeout: Request timeout in seconds (default: 60)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> Dict:
        """
        Check API health status.

        Returns:
            Dictionary with health status information
        """
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def assess_audio(
        self,
        audio_path: Union[str, Path],
        prompt: Optional[str] = None
    ) -> Dict:
        """
        Assess speech quality from an audio file.

        Args:
            audio_path: Path to the audio file
            prompt: Custom prompt for assessment (optional)

        Returns:
            Dictionary with assessment results including:
                - assessment: Detailed text assessment
                - mos_score: MOS score (1.0-5.0) if extracted
                - processing_time: Time taken for inference
                - audio_filename: Name of the processed file
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/wav")}
            data = {}
            if prompt:
                data["prompt"] = prompt

            response = requests.post(
                f"{self.base_url}/assess",
                files=files,
                data=data,
                timeout=self.timeout
            )

        response.raise_for_status()
        return response.json()

    def assess_batch(
        self,
        audio_paths: List[Union[str, Path]],
        prompt: Optional[str] = None
    ) -> Dict:
        """
        Assess multiple audio files in batch.

        Args:
            audio_paths: List of paths to audio files
            prompt: Custom prompt for assessment (optional)

        Returns:
            Dictionary with results for each file
        """
        files = []
        for path in audio_paths:
            audio_path = Path(path)
            if not audio_path.exists():
                print(f"Warning: File not found: {audio_path}")
                continue
            files.append(("files", (audio_path.name, open(audio_path, "rb"), "audio/wav")))

        data = {}
        if prompt:
            data["prompt"] = prompt

        try:
            response = requests.post(
                f"{self.base_url}/assess-batch",
                files=files,
                data=data,
                timeout=self.timeout * len(files)
            )
            response.raise_for_status()
            return response.json()
        finally:
            # Close all file handles
            for _, (_, file_obj, _) in files:
                file_obj.close()

    def wait_for_ready(self, max_wait: int = 120, check_interval: int = 5) -> bool:
        """
        Wait for the API to become ready.

        Args:
            max_wait: Maximum time to wait in seconds
            check_interval: Time between checks in seconds

        Returns:
            True if API is ready, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                health = self.health_check()
                if health.get("status") == "healthy" and health.get("model_loaded"):
                    return True
            except Exception:
                pass
            time.sleep(check_interval)
        return False


def main():
    """Example usage of the SALMONN client."""
    import argparse

    parser = argparse.ArgumentParser(description="SALMONN SQA API Client")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--prompt", help="Custom assessment prompt")
    parser.add_argument("--output", help="Output JSON file path")

    args = parser.parse_args()

    # Initialize client
    client = SALMONNClient(base_url=args.url)

    # Check health
    print("Checking API health...")
    try:
        health = client.health_check()
        print(f"✓ API Status: {health['status']}")
        print(f"  Model loaded: {health['model_loaded']}")
        print(f"  Device: {health['device']}")
        print()
    except Exception as e:
        print(f"✗ Error: Cannot connect to API at {args.url}")
        print(f"  {e}")
        return 1

    # Assess audio
    print(f"Assessing audio: {args.audio_file}")
    print("This may take 10-30 seconds...")
    print()

    try:
        result = client.assess_audio(args.audio_file, prompt=args.prompt)

        print("=" * 60)
        print("ASSESSMENT RESULTS")
        print("=" * 60)
        print(f"\nFile: {result['audio_filename']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"\nMOS Score: {result.get('mos_score', 'N/A')}")
        print(f"\nDetailed Assessment:")
        print("-" * 60)
        print(result['assessment'])
        print("-" * 60)

        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n✓ Results saved to: {args.output}")

        return 0

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return 1
    except requests.exceptions.RequestException as e:
        print(f"✗ API Error: {e}")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())


# ============================================================================
# Usage Examples
# ============================================================================

"""
Example 1: Basic usage
----------------------

from client_example import SALMONNClient

client = SALMONNClient(base_url="http://localhost:8000")

# Assess single file
result = client.assess_audio("audio.wav")
print(f"MOS Score: {result['mos_score']}")
print(f"Assessment: {result['assessment']}")


Example 2: Custom prompt
------------------------

client = SALMONNClient()

result = client.assess_audio(
    "audio.wav",
    prompt="Evaluate the speech quality and rate it from 1-5."
)


Example 3: Batch processing
---------------------------

audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]

results = client.assess_batch(audio_files)

for item in results['results']:
    if item['success']:
        print(f"{item['filename']}: MOS {item['result']['mos_score']}")
    else:
        print(f"{item['filename']}: Error - {item['error']}")


Example 4: Wait for API to be ready
-----------------------------------

client = SALMONNClient(base_url="http://my-api.azurewebsites.net")

if client.wait_for_ready(max_wait=120):
    result = client.assess_audio("audio.wav")
else:
    print("API did not become ready in time")


Example 5: Error handling
-------------------------

try:
    result = client.assess_audio("audio.wav")
    if result['mos_score'] and result['mos_score'] < 3.0:
        print("Low quality audio detected")
except FileNotFoundError:
    print("Audio file not found")
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")


Example 6: Using in a script
----------------------------

import sys
from pathlib import Path

client = SALMONNClient()

# Process all WAV files in a directory
audio_dir = Path("audio_samples")
results = []

for audio_file in audio_dir.glob("*.wav"):
    try:
        result = client.assess_audio(audio_file)
        results.append({
            "file": audio_file.name,
            "mos": result.get("mos_score"),
            "assessment": result["assessment"]
        })
    except Exception as e:
        print(f"Error processing {audio_file}: {e}", file=sys.stderr)

# Calculate average MOS
mos_scores = [r["mos"] for r in results if r["mos"] is not None]
if mos_scores:
    avg_mos = sum(mos_scores) / len(mos_scores)
    print(f"Average MOS Score: {avg_mos:.2f}")
"""
