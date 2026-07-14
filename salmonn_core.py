"""
Shared inference core for SALMONN-based *descriptive* Speech Quality Assessment.

This module is the single source of truth that both the FastAPI server
(``api_inference.py``) and the batch CLI (``batch_process_sqa.py``) import, so
their behaviour can never drift. It owns:

  * locating and importing the vendored SALMONN package (kept out of git; see
    ``scripts/setup_salmonn.sh``)
  * exporting ``SQA_ROOT`` for the config's ``${oc.env:SQA_ROOT}`` interpolation
    so model paths are portable (no hardcoded ``/home/...``)
  * loading the model + Whisper feature extractor
  * the canonical SQA prompt
  * audio preprocessing (always resampled to 16 kHz mono)
  * prompt formatting, MOS extraction, the inference context, and a guard that
    detects degenerate / parroted outputs
"""

import argparse
import contextlib
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import WhisperFeatureExtractor

logger = logging.getLogger(__name__)

# Repo root = directory containing this file.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Make the vendored SALMONN package importable, and expose SQA_ROOT so the
# config's ${oc.env:SQA_ROOT} interpolation resolves to portable model paths.
SALMONN_DIR = os.path.join(REPO_ROOT, "salmonn_sqa", "SALMONN")
if SALMONN_DIR not in sys.path:
    sys.path.insert(0, SALMONN_DIR)
os.environ.setdefault("SQA_ROOT", REPO_ROOT)

from config import Config  # noqa: E402  (vendored SALMONN, on sys.path above)
from models.salmonn import SALMONN  # noqa: E402
from utils import move_to_cuda  # noqa: E402

DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "salmonn_sqa", "inference_config.yaml")

# Canonical SQA prompt — instruction-only, with NO worked example.
#
# The previous prompt ended with `Example: The speech has good clarity... {"MOS": 3.9}`
# and SALMONN parroted it verbatim in ~63% of outputs, producing no real
# description and a near-constant MOS. Never put a copyable answer in the prompt.
DEFAULT_SQA_PROMPT = (
    "You are a speech quality assessor. Listen to the audio and describe its "
    "quality in detail. Comment on clarity, naturalness, intelligibility, "
    "listening effort, and any background noise, distortion, or artifacts you "
    "actually hear (mention roughly when they occur). Base every statement on "
    "what is present in this specific recording. After the description, on a new "
    "line, output a single JSON object with your Mean Opinion Score from 1.0 "
    '(bad) to 5.0 (excellent): {"MOS": <number>}.'
)


@dataclass
class SQAModel:
    """Bundle of the loaded model and everything needed to run inference."""

    model: object
    wav_processor: object
    config: object
    device: str


def get_embedding_layer(model):
    """Return the LLaMA input-embedding layer, accounting for the LoRA wrapper."""
    if getattr(model, "lora", False):
        return model.llama_model.model.model.embed_tokens
    return model.llama_model.model.embed_tokens


def load_model(cfg_path: str = DEFAULT_CONFIG_PATH, device_name: str = "cuda:0") -> SQAModel:
    """Load the SALMONN model and Whisper feature extractor from ``cfg_path``."""
    logger.info("Loading model from config: %s (device=%s)", cfg_path, device_name)

    # SALMONN's Config expects an argparse Namespace with cfg_path/options.
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=cfg_path)
    parser.add_argument("--device", type=str, default=device_name)
    parser.add_argument("--options", nargs="+", help="override config settings")
    args = parser.parse_args(["--cfg-path", cfg_path, "--device", device_name])

    config = Config(args)
    device = device_name
    if "cuda" in device and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info("Loading SALMONN model...")
    model = SALMONN.from_config(config.config.model)
    model.to(device)
    model.eval()

    logger.info("Loading Whisper feature extractor...")
    wav_processor = WhisperFeatureExtractor.from_pretrained(config.config.model.whisper_path)

    logger.info("Model loaded successfully.")
    return SQAModel(model=model, wav_processor=wav_processor, config=config, device=device)


def prepare_audio_sample(wav_path, wav_processor, device="cpu", target_sr: int = 16000) -> dict:
    """Load audio, convert to mono, **always** resample to 16 kHz, pad/truncate, featurize.

    Always routing through resampling removes the dual-code-path hazard of the
    old sr==16000 fast-path (which silently trusted the file's header).
    """
    audio, sr = sf.read(wav_path)  # float64, like upstream prepare_one_sample

    if audio.ndim == 2:  # stereo -> first channel
        audio = audio[:, 0]

    if sr != target_sr:
        # torchaudio needs float32 for resampling; cast back to float64 after.
        audio_t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(audio_t)
        audio = audio.squeeze(0).numpy().astype(np.float64)
        sr = target_sr

    if len(audio) < sr:  # pad to at least 1 s
        audio = np.concatenate([audio, np.zeros(sr - len(audio), dtype=audio.dtype)])
    audio = audio[: sr * 30]  # cap at 30 s

    # NOTE: raw_wav must stay float64. Casting to float32 makes it eligible for
    # fp16 autocast downcasting, which drives the BEATs encoder to NaN and yields
    # garbage tokens. The spectrogram is float32 (the Whisper extractor's choice).
    spectrogram = wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"]
    samples = {
        "spectrogram": spectrogram,
        "raw_wav": torch.from_numpy(audio).unsqueeze(0),
        "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
    }
    if "cuda" in str(device):
        samples = move_to_cuda(samples)
    return samples


def format_sqa_prompt(prompt: str, prompt_template: str) -> str:
    """Wrap the user prompt with the ``<Speech>`` placeholder and apply the template."""
    prompt = prompt.strip()
    if "<SpeechHere>" not in prompt:
        prompt = "<Speech><SpeechHere></Speech> " + prompt
    return prompt_template.format(prompt)


_MOS_JSON = re.compile(r'"MOS"\s*:\s*([0-5](?:\.[0-9]+)?)')
# The SQA-finetuned model writes the score in prose: "MOS of 4.0",
# "Mean Opinion Score is 5.0", "MOS: 3.9". Match MOS / Mean Opinion Score
# followed (within a short window of non-digits) by the value.
_MOS_LABEL = re.compile(
    r"(?:MOS|mean opinion score)\b[^0-9]{0,20}?([0-5](?:\.[0-9]+)?)", re.IGNORECASE
)
_MOS_BARE = re.compile(r"^\s*([0-5](?:\.[0-9]+)?)\s*$")


def extract_mos(text: str) -> Optional[float]:
    """Extract a MOS in [1.0, 5.0] or ``None``.

    Order: JSON ``{"MOS": x}`` -> prose ``MOS of x`` / ``Mean Opinion Score is x``
    -> a whole-output bare score (terse replies like "4.5"). Deliberately does
    not pick up arbitrary in-text numbers (e.g. "1.5 seconds").
    """
    if not text:
        return None
    for rx in (_MOS_JSON, _MOS_LABEL):
        m = rx.search(text)
        if m:
            value = float(m.group(1))
            if 1.0 <= value <= 5.0:
                return value
    m = _MOS_BARE.match(clean_output(text))
    if m:
        value = float(m.group(1))
        if 1.0 <= value <= 5.0:
            return value
    return None


@contextlib.contextmanager
def inference_context(device):
    """``no_grad`` everywhere, plus fp16 ``autocast`` on CUDA.

    fp16 (not bf16) is required: SALMONN's speech encoders + Q-Former were
    trained/validated in fp16 and produce garbage embeddings under bf16
    autocast. With beam search / greedy decoding (``do_sample=False``) fp16 is
    numerically stable; only multinomial *sampling* is fp16-unstable, which is
    why decoding stays on beam search (see inference_config.yaml).
    """
    if "cuda" in str(device):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        with torch.no_grad():
            yield


def generate_sqa(sqa: SQAModel, prompt: str = DEFAULT_SQA_PROMPT, *, samples=None, wav_path=None) -> str:
    """Run one SQA generation and return the raw model text.

    Provide either pre-built ``samples`` or a ``wav_path`` to load+prepare.
    """
    if samples is None:
        if wav_path is None:
            raise ValueError("generate_sqa needs either samples or wav_path")
        samples = prepare_audio_sample(wav_path, sqa.wav_processor, device=sqa.device)
    formatted = format_sqa_prompt(prompt, sqa.config.config.model.prompt_template)
    with inference_context(sqa.device):
        return sqa.model.generate(samples, sqa.config.config.generate, prompts=[formatted])[0]


def clean_output(text: str) -> str:
    """Strip the model's BOS/EOS markers for display/storage."""
    return text.replace("<s>", "").replace("</s>", "").strip()


def is_degenerate(text: str) -> bool:
    """Flag outputs that are not real assessments (empty, too short, or a
    placeholder echo like ``good clarity... {"MOS": 4.0}``).

    A safety net so a future prompt regression can't silently corrupt the
    aggregate MOS again.
    """
    if not text:
        return True
    cleaned = clean_output(text)
    if len(cleaned) < 40:
        return True
    # An ellipsis in a short output is the tell-tale of a copied placeholder.
    if "..." in cleaned and len(cleaned) < 120:
        return True
    return False
