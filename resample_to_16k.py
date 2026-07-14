"""
Resample all audio files in a directory to 16kHz
"""
import argparse
from pathlib import Path
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm


def resample_file(input_path: Path, output_path: Path, target_sr: int = 16000):
    """Resample a single audio file to target sample rate."""
    # Load audio
    audio, sr = sf.read(input_path)

    # Convert stereo to mono
    if len(audio.shape) == 2:
        audio = audio[:, 0]

    # Resample if needed
    if sr != target_sr:
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio_resampled = resampler(audio_tensor).squeeze(0).numpy()
    else:
        audio_resampled = audio

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio_resampled, target_sr)


def main():
    parser = argparse.ArgumentParser(description="Resample audio files to 16kHz")
    parser.add_argument("input_dir", help="Input directory with audio files")
    parser.add_argument("output_dir", help="Output directory for resampled files")
    parser.add_argument("--pattern", default="*.wav", help="File pattern to match")
    parser.add_argument("--target-sr", type=int, default=16000, help="Target sample rate")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Find all audio files
    audio_files = sorted(list(input_dir.glob(args.pattern)))
    print(f"Found {len(audio_files)} files to resample")

    # Resample all files
    for audio_file in tqdm(audio_files, desc="Resampling"):
        output_file = output_dir / audio_file.name
        resample_file(audio_file, output_file, args.target_sr)

    print(f"\nResampled {len(audio_files)} files to {output_dir}")


if __name__ == "__main__":
    main()
