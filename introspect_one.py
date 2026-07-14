"""
Comprehensive Introspection Test for SALMONN SQA

This script runs all available debugging introspection features, decodes
embedding tokens given to the LLM for quality assessment, and logs everything
to MLflow.

Features:
- Captures all pipeline stage activations
- Decodes LLaMA embeddings to readable tokens
- Creates comprehensive visualizations
- Logs all data to MLflow for analysis
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
import torch
from transformers import WhisperFeatureExtractor

# Add SALMONN directory to path
salmonn_path = os.path.join(os.path.dirname(__file__), "salmonn_sqa", "SALMONN")
sys.path.insert(0, salmonn_path)

from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample
from model_introspection import (
    ModelIntrospector,
    TokenAnalyzer,
    EmbeddingDecoder,
    create_introspection_summary
)
from visualization_utils import create_introspection_visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Canonical SQA prompt — single source of truth (instruction-only, no leaky
# worked example). See salmonn_core for why.
from salmonn_core import DEFAULT_SQA_PROMPT as DEFAULT_PROMPT


def load_model(cfg_path: str, device_name: str = "cuda:0"):
    """
    Load the SALMONN model and processor.

    Args:
        cfg_path: Path to config file
        device_name: Device to use (cuda:0, cpu, etc.)

    Returns:
        Tuple of (model, wav_processor, config, device)
    """
    logger.info(f"Loading model from config: {cfg_path}")
    logger.info(f"Using device: {device_name}")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=cfg_path)
    parser.add_argument("--device", type=str, default=device_name)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config"
    )
    args = parser.parse_args(["--cfg-path", cfg_path, "--device", device_name])

    # Load config
    config = Config(args)
    device = args.device

    # Check if CUDA is available if cuda device is specified
    if "cuda" in device and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load model
    logger.info("Loading SALMONN model...")
    model = SALMONN.from_config(config.config.model)
    model.to(device)
    model.eval()

    # Load wav processor
    logger.info("Loading Whisper feature extractor...")
    wav_processor = WhisperFeatureExtractor.from_pretrained(
        config.config.model.whisper_path
    )

    logger.info("Model loaded successfully!")

    return model, wav_processor, config, device


def decode_llama_embeddings(
    introspector: ModelIntrospector,
    model: SALMONN,
    top_k: int = 5,
    max_positions: int = 50
) -> Dict[str, Any]:
    """
    Decode LLaMA embeddings to readable tokens.

    Args:
        introspector: ModelIntrospector instance with captured embeddings
        model: SALMONN model instance
        top_k: Number of top nearest tokens to return per position
        max_positions: Maximum number of sequence positions to decode

    Returns:
        Dictionary containing decoded embedding information
    """
    logger.info("Decoding LLaMA embeddings to readable tokens...")

    # Get the embedding layer
    if model.lora:
        embedding_layer = model.llama_model.model.model.embed_tokens
    else:
        embedding_layer = model.llama_model.model.embed_tokens

    # Create decoder
    decoder = EmbeddingDecoder(embedding_layer, model.llama_tokenizer)

    # Get captured embeddings
    llama_embeddings_list = introspector.get_llama_embeddings()

    if not llama_embeddings_list:
        logger.warning("No LLaMA embeddings captured!")
        return {
            'status': 'no_embeddings',
            'message': 'No LLaMA embeddings were captured during inference'
        }

    # Use the last captured embeddings (most complete)
    embeddings = llama_embeddings_list[-1]
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Decode embeddings
    num_positions = min(embeddings.shape[1], max_positions)
    sequence_embeddings = embeddings[0, :num_positions, :]

    # Get decoded tokens for each position
    decoded_sequence = decoder.decode_embedding_sequence(
        sequence_embeddings,
        top_k=top_k,
        metric='cosine'
    )

    # Get formatted output
    formatted_output = decoder.decode_and_format(
        sequence_embeddings,
        top_k=top_k,
        metric='cosine'
    )

    # Also decode the projection output (audio embeddings)
    projection_output = introspector.get_projection_output()
    audio_decoded = None

    if projection_output is not None:
        logger.info(f"Decoding audio projection output (shape: {projection_output.shape})...")
        # Take first few audio token embeddings
        if isinstance(projection_output, torch.Tensor) and projection_output.dim() == 3:
            # Projection output shape is typically [seq_len, batch_size, hidden_dim]
            # or [batch_size, seq_len, hidden_dim] depending on the model
            # Check which dimension is larger to determine the sequence dimension
            seq_dim = 0 if projection_output.shape[0] > projection_output.shape[1] else 1
            batch_dim = 1 - seq_dim  # The other dimension

            num_audio_tokens = min(10, projection_output.shape[seq_dim])

            if seq_dim == 0:
                # Shape is [seq_len, batch_size, hidden_dim]
                audio_embeddings = projection_output[:num_audio_tokens, 0, :]
            else:
                # Shape is [batch_size, seq_len, hidden_dim]
                audio_embeddings = projection_output[0, :num_audio_tokens, :]

            audio_decoded = decoder.decode_embedding_sequence(
                audio_embeddings,
                top_k=top_k,
                metric='cosine'
            )

    return {
        'status': 'success',
        'embeddings_shape': list(embeddings.shape),
        'num_positions_decoded': num_positions,
        'top_k': top_k,
        'decoded_sequence': decoded_sequence,
        'formatted_output': formatted_output,
        'audio_projection_decoded': audio_decoded,
        'total_embedding_calls': len(llama_embeddings_list)
    }


def run_full_introspection_test(
    audio_file: str,
    cfg_path: str = "salmonn_sqa/inference_config.yaml",
    device: str = "cuda:0",
    prompt: Optional[str] = None,
    mlflow_experiment: str = "SALMONN_Introspection_Debug",
    embedding_top_k: int = 5,
    max_decode_positions: int = 50
) -> Dict[str, Any]:
    """
    Run comprehensive introspection test on an audio file.

    Args:
        audio_file: Path to audio file to test
        cfg_path: Path to model config file
        device: Device to use for inference
        prompt: Optional custom prompt (uses default if None)
        mlflow_experiment: MLflow experiment name
        embedding_top_k: Number of top tokens to decode per position
        max_decode_positions: Maximum sequence positions to decode

    Returns:
        Dictionary with test results
    """
    start_time = time.time()

    # Use default prompt if none provided
    if prompt is None:
        prompt = DEFAULT_PROMPT

    # Initialize MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment(mlflow_experiment)

    # Load model
    logger.info("=" * 80)
    logger.info("FULL INTROSPECTION TEST")
    logger.info("=" * 80)

    model, wav_processor, config, device = load_model(cfg_path, device)

    # Prepare sample
    logger.info(f"\nProcessing audio file: {audio_file}")
    samples = prepare_one_sample(audio_file, wav_processor)

    # Format prompt
    if "<SpeechHere>" in prompt:
        formatted_prompt = config.config.model.prompt_template.format(prompt.strip())
    else:
        formatted_prompt = config.config.model.prompt_template.format(
            "<Speech><SpeechHere></Speech> " + prompt.strip()
        )

    # Start MLflow run
    run_name = f"full_introspection_{Path(audio_file).stem}_{int(time.time())}"

    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"\nMLflow Run ID: {run.info.run_id}")

        # Log parameters
        mlflow.log_param("audio_file", audio_file)
        mlflow.log_param("config_path", cfg_path)
        mlflow.log_param("device", device)
        mlflow.log_param("prompt", prompt)
        mlflow.log_param("embedding_top_k", embedding_top_k)
        mlflow.log_param("max_decode_positions", max_decode_positions)
        mlflow.log_param("test_type", "full_introspection")

        # Set tags
        mlflow.set_tag("introspection_enabled", "True")
        mlflow.set_tag("embedding_decoding", "True")
        mlflow.set_tag("task", "comprehensive_debug_test")

        # Create introspection tools
        logger.info("\nInitializing introspection tools...")
        introspector = ModelIntrospector(model)
        token_analyzer = TokenAnalyzer(model.llama_tokenizer)

        # Run inference with introspection
        logger.info("\nRunning inference with full introspection enabled...")
        inference_start = time.time()

        with torch.cuda.amp.autocast(dtype=torch.float16) if "cuda" in device else torch.no_grad():
            output_list, introspection_data = introspector.generate_with_introspection(
                samples, config.config.generate, prompts=[formatted_prompt]
            )
            output = output_list[0]

        inference_time = time.time() - inference_start
        logger.info(f"\nInference complete in {inference_time:.2f}s")
        logger.info(f"Output: {output[:200]}...")

        # Log inference metrics
        mlflow.log_metric("inference_time_seconds", inference_time)
        mlflow.log_text(output, "assessment_output.txt")

        # Get encoder outputs
        logger.info("\nExtracting encoder outputs...")
        encoder_outputs = introspector.get_encoder_outputs()
        logger.info(f"Encoder outputs captured: {list(encoder_outputs.keys())}")

        # Decode embeddings
        logger.info("\n" + "=" * 80)
        logger.info("DECODING LLAMA EMBEDDINGS")
        logger.info("=" * 80)

        embedding_decode_start = time.time()
        decoded_data = decode_llama_embeddings(
            introspector,
            model,
            top_k=embedding_top_k,
            max_positions=max_decode_positions
        )
        embedding_decode_time = time.time() - embedding_decode_start

        logger.info(f"\nEmbedding decoding complete in {embedding_decode_time:.2f}s")
        logger.info(f"Status: {decoded_data['status']}")

        if decoded_data['status'] == 'success':
            logger.info(f"Decoded {decoded_data['num_positions_decoded']} positions")
            logger.info(f"\nFormatted decoded output:\n{decoded_data['formatted_output'][:500]}...")

        # Log embedding decoding results
        mlflow.log_metric("embedding_decode_time_seconds", embedding_decode_time)
        mlflow.log_dict(decoded_data, "embedding_decoding.json")

        if decoded_data['status'] == 'success':
            mlflow.log_text(decoded_data['formatted_output'], "embeddings_decoded.txt")

        # Create introspection summary
        logger.info("\n" + "=" * 80)
        logger.info("CREATING INTROSPECTION SUMMARY")
        logger.info("=" * 80)

        introspection_summary = create_introspection_summary(introspection_data)
        logger.info(f"\nPipeline stages captured: {len(introspection_summary['pipeline_stages'])}")

        for stage in introspection_summary['pipeline_stages']:
            logger.info(f"  - {stage['name']}: {stage['num_calls']} calls")
            for act in stage['activations']:
                if 'shape' in act:
                    logger.info(f"    Shape: {act['shape']}")

        # Log introspection data
        mlflow.log_dict(introspection_summary, "introspection_summary.json")
        mlflow.log_dict(introspection_data, "introspection_detailed.json")

        # Create visualizations
        logger.info("\n" + "=" * 80)
        logger.info("CREATING VISUALIZATIONS")
        logger.info("=" * 80)

        viz_start = time.time()
        with tempfile.TemporaryDirectory() as viz_dir:
            viz_paths = create_introspection_visualizations(
                introspection_data,
                encoder_outputs,
                Path(viz_dir)
            )

            logger.info(f"\nCreated {len(viz_paths)} visualizations:")
            for viz_path in viz_paths:
                logger.info(f"  - {viz_path.name}")
                mlflow.log_artifact(str(viz_path), "introspection_visualizations")

        viz_time = time.time() - viz_start
        mlflow.log_metric("visualization_time_seconds", viz_time)

        # Log audio file
        try:
            mlflow.log_artifact(audio_file, "input_audio")
        except Exception as e:
            logger.warning(f"Failed to log audio artifact: {e}")

        # Calculate total time
        total_time = time.time() - start_time
        mlflow.log_metric("total_test_time_seconds", total_time)

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("TEST COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nTotal time: {total_time:.2f}s")
        logger.info(f"  - Inference: {inference_time:.2f}s")
        logger.info(f"  - Embedding decoding: {embedding_decode_time:.2f}s")
        logger.info(f"  - Visualizations: {viz_time:.2f}s")
        logger.info(f"\nMLflow Run ID: {run.info.run_id}")
        logger.info(f"View results with: mlflow ui")

        mlflow.set_tag("status", "success")

        return {
            'status': 'success',
            'run_id': run.info.run_id,
            'output': output,
            'introspection_summary': introspection_summary,
            'decoded_embeddings': decoded_data,
            'timing': {
                'total': total_time,
                'inference': inference_time,
                'embedding_decode': embedding_decode_time,
                'visualization': viz_time
            },
            'visualizations_created': len(viz_paths)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive introspection test on SALMONN SQA model"
    )
    parser.add_argument(
        "audio_file",
        help="Path to audio file to test"
    )
    parser.add_argument(
        "--config",
        default="salmonn_sqa/inference_config.yaml",
        help="Path to model config file"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use (cuda:0, cpu, etc.)"
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom prompt (uses default if not specified)"
    )
    parser.add_argument(
        "--experiment",
        default="SALMONN_Introspection_Debug",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--embedding-top-k",
        type=int,
        default=5,
        help="Number of top tokens to decode per embedding position"
    )
    parser.add_argument(
        "--max-decode-positions",
        type=int,
        default=50,
        help="Maximum number of sequence positions to decode"
    )

    args = parser.parse_args()

    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        logger.error(f"Audio file not found: {args.audio_file}")
        sys.exit(1)

    # Run test
    try:
        result = run_full_introspection_test(
            audio_file=args.audio_file,
            cfg_path=args.config,
            device=args.device,
            prompt=args.prompt,
            mlflow_experiment=args.experiment,
            embedding_top_k=args.embedding_top_k,
            max_decode_positions=args.max_decode_positions
        )

        logger.info("\n" + "=" * 80)
        logger.info("SUCCESS!")
        logger.info("=" * 80)
        logger.info(f"\nRun ID: {result['run_id']}")
        logger.info(f"Created {result['visualizations_created']} visualizations")
        logger.info(f"Decoded {result['decoded_embeddings'].get('num_positions_decoded', 0)} embedding positions")
        logger.info(f"\nView results: mlflow ui")

    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error("TEST FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
