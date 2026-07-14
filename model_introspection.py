"""
Model Introspection Utilities for SALMONN

This module provides tools to capture and analyze intermediate outputs
from the SALMONN model during inference, including:
- Encoder outputs (Whisper, BEATs)
- Q-Former outputs
- LLM decoder states
- Token representations
- Attention patterns

Usage:
    introspector = ModelIntrospector(model)
    outputs, introspection_data = introspector.generate_with_introspection(samples, config)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import json
from pathlib import Path


class ActivationCapture:
    """Hook to capture activations from a specific layer."""

    def __init__(self, name: str):
        self.name = name
        self.activations = []
        self.inputs = []

    def __call__(self, module, input, output):
        """Hook function called during forward pass."""
        # Store input (detach and move to CPU to save memory)
        if isinstance(input, tuple):
            self.inputs.append(tuple(i.detach().cpu() if isinstance(i, torch.Tensor) else i for i in input))
        else:
            self.inputs.append(input.detach().cpu() if isinstance(input, torch.Tensor) else input)

        # Store output
        if isinstance(output, tuple):
            self.activations.append(tuple(o.detach().cpu() if isinstance(o, torch.Tensor) else o for o in output))
        elif hasattr(output, 'last_hidden_state'):
            # For transformer outputs
            self.activations.append({
                'last_hidden_state': output.last_hidden_state.detach().cpu(),
                'hidden_states': output.hidden_states if hasattr(output, 'hidden_states') and output.hidden_states else None,
                'attentions': output.attentions if hasattr(output, 'attentions') and output.attentions else None,
            })
        else:
            self.activations.append(output.detach().cpu() if isinstance(output, torch.Tensor) else output)

    def clear(self):
        """Clear stored activations."""
        self.activations = []
        self.inputs = []


class ModelIntrospector:
    """
    Introspection utility for SALMONN model.

    Captures intermediate outputs at each stage of the model pipeline:
    1. Whisper encoder output
    2. BEATs encoder output (if available)
    3. Q-Former output
    4. Linear projection output
    5. LLaMA embeddings
    6. LLaMA decoder hidden states
    """

    def __init__(self, model):
        """
        Initialize introspector with a SALMONN model.

        Args:
            model: SALMONN model instance
        """
        self.model = model
        self.hooks = []
        self.captures = {}

    def _register_hooks(self):
        """Register forward hooks on all components of interest."""
        # Clear existing hooks
        self.remove_hooks()

        # Clear old captures
        for capture in self.captures.values():
            capture.clear()

        # Hook into Whisper encoder
        if hasattr(self.model, 'speech_encoder'):
            capture = ActivationCapture('whisper_encoder')
            hook = self.model.speech_encoder.register_forward_hook(capture)
            self.hooks.append(hook)
            self.captures['whisper_encoder'] = capture

        # Hook into BEATs encoder
        if hasattr(self.model, 'beats') and self.model.beats_path:
            capture = ActivationCapture('beats_encoder')
            hook = self.model.beats.register_forward_hook(capture)
            self.hooks.append(hook)
            self.captures['beats_encoder'] = capture

        # Hook into layer normalization
        if hasattr(self.model, 'ln_speech'):
            capture = ActivationCapture('ln_speech')
            hook = self.model.ln_speech.register_forward_hook(capture)
            self.hooks.append(hook)
            self.captures['ln_speech'] = capture

        if hasattr(self.model, 'ln_audio'):
            capture = ActivationCapture('ln_audio')
            hook = self.model.ln_audio.register_forward_hook(capture)
            self.hooks.append(hook)
            self.captures['ln_audio'] = capture

        # Hook into Q-Former
        if hasattr(self.model, 'speech_Qformer'):
            capture = ActivationCapture('qformer')
            hook = self.model.speech_Qformer.bert.register_forward_hook(capture)
            self.hooks.append(hook)
            self.captures['qformer'] = capture

        # Hook into linear projection
        if hasattr(self.model, 'speech_llama_proj'):
            capture = ActivationCapture('linear_projection')
            hook = self.model.speech_llama_proj.register_forward_hook(capture)
            self.hooks.append(hook)
            self.captures['linear_projection'] = capture

        # Hook into LLaMA embedding layer
        if hasattr(self.model, 'llama_model'):
            if self.model.lora:
                embed_layer = self.model.llama_model.model.model.embed_tokens
            else:
                embed_layer = self.model.llama_model.model.embed_tokens
            capture = ActivationCapture('llama_embeddings')
            hook = embed_layer.register_forward_hook(capture)
            self.hooks.append(hook)
            self.captures['llama_embeddings'] = capture

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        # Don't clear captures - we may need them after hook removal
        # Captures will be cleared when new hooks are registered via _register_hooks()

    def generate_with_introspection(
        self,
        samples: Dict[str, torch.Tensor],
        generate_cfg: Dict,
        prompts: Optional[List[str]] = None
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Generate text while capturing intermediate outputs.

        Args:
            samples: Input samples dict with 'spectrogram', 'raw_wav', etc.
            generate_cfg: Generation configuration
            prompts: Optional list of prompts

        Returns:
            Tuple of (generated_text_list, introspection_data_dict)
        """
        # Register hooks
        self._register_hooks()

        # Run generation
        try:
            outputs = self.model.generate(samples, generate_cfg, prompts=prompts)
        finally:
            # Always remove hooks after generation
            pass  # We'll remove them in extract_introspection_data

        # Extract introspection data
        introspection_data = self.extract_introspection_data()

        # Remove hooks
        self.remove_hooks()

        return outputs, introspection_data

    def extract_introspection_data(self) -> Dict[str, Any]:
        """
        Extract and format introspection data from captures.

        Returns:
            Dictionary containing introspection data with statistics and metadata
        """
        data = {}

        for name, capture in self.captures.items():
            if not capture.activations:
                continue

            component_data = {
                'num_calls': len(capture.activations),
                'activations': [],
            }

            for idx, activation in enumerate(capture.activations):
                activation_info = self._analyze_activation(activation, name)
                component_data['activations'].append(activation_info)

            data[name] = component_data

        return data

    def _analyze_activation(self, activation: Any, component_name: str) -> Dict[str, Any]:
        """
        Analyze a single activation and extract metadata.

        Args:
            activation: The activation tensor or dict
            component_name: Name of the component

        Returns:
            Dictionary with activation metadata
        """
        info = {'component': component_name}

        if isinstance(activation, torch.Tensor):
            info.update({
                'shape': list(activation.shape),
                'dtype': str(activation.dtype),
                'mean': float(activation.float().mean()),
                'std': float(activation.float().std()),
                'min': float(activation.float().min()),
                'max': float(activation.float().max()),
                'norm': float(torch.norm(activation.float())),
            })
        elif isinstance(activation, dict):
            info['type'] = 'dict'
            for key, value in activation.items():
                if isinstance(value, torch.Tensor):
                    info[key] = {
                        'shape': list(value.shape),
                        'dtype': str(value.dtype),
                        'mean': float(value.float().mean()),
                        'std': float(value.float().std()),
                        'min': float(value.float().min()),
                        'max': float(value.float().max()),
                    }
                elif value is not None:
                    info[key] = str(type(value))
        elif isinstance(activation, tuple):
            info['type'] = 'tuple'
            info['length'] = len(activation)
            for idx, item in enumerate(activation):
                if isinstance(item, torch.Tensor):
                    info[f'item_{idx}'] = {
                        'shape': list(item.shape),
                        'dtype': str(item.dtype),
                        'mean': float(item.float().mean()),
                        'std': float(item.float().std()),
                    }

        return info

    def get_encoder_outputs(self) -> Dict[str, torch.Tensor]:
        """Get the encoder outputs (Whisper and BEATs)."""
        outputs = {}

        if 'whisper_encoder' in self.captures and self.captures['whisper_encoder'].activations:
            outputs['whisper'] = self.captures['whisper_encoder'].activations[-1]

        if 'beats_encoder' in self.captures and self.captures['beats_encoder'].activations:
            outputs['beats'] = self.captures['beats_encoder'].activations[-1]

        return outputs

    def get_qformer_outputs(self) -> Optional[torch.Tensor]:
        """Get Q-Former output."""
        if 'qformer' in self.captures and self.captures['qformer'].activations:
            return self.captures['qformer'].activations[-1]
        return None

    def get_projection_output(self) -> Optional[torch.Tensor]:
        """Get linear projection output."""
        if 'linear_projection' in self.captures and self.captures['linear_projection'].activations:
            return self.captures['linear_projection'].activations[-1]
        return None

    def get_llama_embeddings(self) -> List[torch.Tensor]:
        """Get all LLaMA embeddings."""
        if 'llama_embeddings' in self.captures:
            return self.captures['llama_embeddings'].activations
        return []


class TokenAnalyzer:
    """Analyze and visualize token representations."""

    def __init__(self, tokenizer):
        """
        Initialize token analyzer.

        Args:
            tokenizer: LLaMA tokenizer
        """
        self.tokenizer = tokenizer

    def decode_tokens(self, token_ids: torch.Tensor) -> List[str]:
        """
        Decode token IDs to text.

        Args:
            token_ids: Tensor of token IDs

        Returns:
            List of decoded tokens
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()

        tokens = []
        for token_id in token_ids:
            if isinstance(token_id, np.ndarray):
                token_id = int(token_id)
            token = self.tokenizer.decode([token_id])
            tokens.append(token)

        return tokens

    def analyze_generation(self, output_ids: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze generated token sequence.

        Args:
            output_ids: Generated token IDs

        Returns:
            Analysis dictionary
        """
        if isinstance(output_ids, torch.Tensor):
            output_ids = output_ids.cpu().numpy()

        tokens = self.decode_tokens(output_ids)

        return {
            'num_tokens': len(tokens),
            'tokens': tokens,
            'token_ids': output_ids.tolist() if isinstance(output_ids, np.ndarray) else output_ids,
            'decoded_text': self.tokenizer.decode(output_ids, skip_special_tokens=False),
        }

    def get_token_embeddings(
        self,
        token_ids: torch.Tensor,
        embedding_layer: nn.Embedding
    ) -> torch.Tensor:
        """
        Get embeddings for given token IDs.

        Args:
            token_ids: Token IDs
            embedding_layer: Embedding layer

        Returns:
            Token embeddings
        """
        with torch.no_grad():
            embeddings = embedding_layer(token_ids)
        return embeddings


def create_introspection_summary(introspection_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a human-readable summary of introspection data.

    Args:
        introspection_data: Raw introspection data

    Returns:
        Summary dictionary
    """
    summary = {
        'pipeline_stages': [],
        'total_components': len(introspection_data),
    }

    stage_order = [
        'whisper_encoder',
        'beats_encoder',
        'ln_speech',
        'ln_audio',
        'qformer',
        'linear_projection',
        'llama_embeddings',
    ]

    for stage_name in stage_order:
        if stage_name in introspection_data:
            stage_info = introspection_data[stage_name]

            stage_summary = {
                'name': stage_name,
                'num_calls': stage_info['num_calls'],
                'activations': []
            }

            for activation in stage_info['activations']:
                if 'shape' in activation:
                    stage_summary['activations'].append({
                        'shape': activation['shape'],
                        'stats': {
                            'mean': activation.get('mean'),
                            'std': activation.get('std'),
                            'norm': activation.get('norm'),
                        }
                    })

            summary['pipeline_stages'].append(stage_summary)

    return summary


def save_introspection_data(
    introspection_data: Dict[str, Any],
    output_path: Path,
    include_tensors: bool = False
):
    """
    Save introspection data to disk.

    Args:
        introspection_data: Introspection data dictionary
        output_path: Path to save data
        include_tensors: Whether to save full tensors (can be large)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save summary as JSON
    summary = create_introspection_summary(introspection_data)
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save detailed data
    if include_tensors:
        # Save tensors separately
        tensors = {}
        for component_name, component_data in introspection_data.items():
            for idx, activation in enumerate(component_data.get('activations', [])):
                # This would need to save actual tensors
                # For now, just save metadata
                pass

    # Save metadata
    metadata = {}
    for component_name, component_data in introspection_data.items():
        metadata[component_name] = {
            'num_calls': component_data['num_calls'],
            'num_activations': len(component_data.get('activations', [])),
        }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


class EmbeddingDecoder:
    """
    Decode embeddings back to nearest tokens in vocabulary.

    This class helps interpret embedding vectors by finding which tokens
    in the vocabulary they are most similar to.
    """

    def __init__(self, embedding_layer: nn.Embedding, tokenizer):
        """
        Initialize the embedding decoder.

        Args:
            embedding_layer: The model's embedding layer (e.g., model.embed_tokens)
            tokenizer: The tokenizer to decode token IDs to text
        """
        self.embedding_layer = embedding_layer
        self.tokenizer = tokenizer
        self.vocab_size = embedding_layer.num_embeddings

        # Cache the full vocabulary embedding matrix
        with torch.no_grad():
            self.vocab_embeddings = embedding_layer.weight.clone()  # (vocab_size, hidden_dim)

    def find_nearest_tokens(
        self,
        embedding: torch.Tensor,
        top_k: int = 5,
        metric: str = 'cosine'
    ) -> List[Dict[str, Any]]:
        """
        Find the nearest tokens in vocabulary for a given embedding.

        Args:
            embedding: The embedding vector (hidden_dim,) or (batch, hidden_dim)
            top_k: Number of nearest tokens to return
            metric: Distance metric ('cosine' or 'euclidean')

        Returns:
            List of dicts with token_id, token_text, and similarity/distance
        """
        # Ensure embedding is 2D (1, hidden_dim) if it's 1D
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # (1, hidden_dim)

        # Move to same device as vocab embeddings
        embedding = embedding.to(self.vocab_embeddings.device)

        if metric == 'cosine':
            # Compute cosine similarity
            # Normalize embeddings
            embedding_norm = torch.nn.functional.normalize(embedding, p=2, dim=1)
            vocab_norm = torch.nn.functional.normalize(self.vocab_embeddings, p=2, dim=1)

            # Compute similarity (batch, vocab_size)
            similarities = torch.matmul(embedding_norm, vocab_norm.t())

            # Get top-k
            top_k_values, top_k_indices = torch.topk(similarities[0], k=min(top_k, self.vocab_size))

            results = []
            for idx, (token_id, similarity) in enumerate(zip(top_k_indices, top_k_values)):
                token_id = int(token_id)
                token_text = self.tokenizer.decode([token_id])
                results.append({
                    'rank': idx + 1,
                    'token_id': token_id,
                    'token_text': token_text,
                    'cosine_similarity': float(similarity)
                })

        elif metric == 'euclidean':
            # Compute euclidean distance
            # (batch, vocab_size, hidden_dim)
            distances = torch.cdist(embedding, self.vocab_embeddings, p=2)

            # Get top-k (smallest distances)
            top_k_values, top_k_indices = torch.topk(
                distances[0],
                k=min(top_k, self.vocab_size),
                largest=False  # Get smallest distances
            )

            results = []
            for idx, (token_id, distance) in enumerate(zip(top_k_indices, top_k_values)):
                token_id = int(token_id)
                token_text = self.tokenizer.decode([token_id])
                results.append({
                    'rank': idx + 1,
                    'token_id': token_id,
                    'token_text': token_text,
                    'euclidean_distance': float(distance)
                })

        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'")

        return results

    def decode_embedding_sequence(
        self,
        embeddings: torch.Tensor,
        top_k: int = 3,
        metric: str = 'cosine'
    ) -> List[List[Dict[str, Any]]]:
        """
        Decode a sequence of embeddings.

        Args:
            embeddings: Tensor of shape (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
            top_k: Number of nearest tokens per position
            metric: Distance metric to use

        Returns:
            List of lists, where each inner list contains top_k nearest tokens for that position
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)  # (1, seq_len, hidden_dim)

        batch_size, seq_len, hidden_dim = embeddings.shape

        # Process first batch only for now
        batch_results = []
        for position in range(seq_len):
            embedding_vec = embeddings[0, position, :]  # (hidden_dim,)
            nearest = self.find_nearest_tokens(embedding_vec, top_k=top_k, metric=metric)
            batch_results.append(nearest)

        return batch_results

    def decode_and_format(
        self,
        embeddings: torch.Tensor,
        top_k: int = 3,
        metric: str = 'cosine'
    ) -> str:
        """
        Decode embeddings and format as human-readable string.

        Args:
            embeddings: Embedding tensor
            top_k: Number of nearest tokens to show
            metric: Distance metric to use

        Returns:
            Formatted string showing nearest tokens for each position
        """
        results = self.decode_embedding_sequence(embeddings, top_k=top_k, metric=metric)

        output = []
        output.append(f"Embedding sequence decoding (top-{top_k} nearest tokens):")
        output.append("=" * 70)

        for pos, nearest_tokens in enumerate(results):
            output.append(f"\nPosition {pos}:")
            for token_info in nearest_tokens:
                rank = token_info['rank']
                token_text = token_info['token_text']
                token_id = token_info['token_id']

                if 'cosine_similarity' in token_info:
                    score = token_info['cosine_similarity']
                    output.append(f"  {rank}. '{token_text}' (id={token_id}, sim={score:.4f})")
                else:
                    score = token_info['euclidean_distance']
                    output.append(f"  {rank}. '{token_text}' (id={token_id}, dist={score:.4f})")

        return "\n".join(output)


def visualize_embedding_statistics(
    embeddings: torch.Tensor,
    title: str = "Embedding Statistics"
) -> Dict[str, Any]:
    """
    Compute statistics for visualization of embeddings.

    Args:
        embeddings: Tensor of embeddings (batch_size, seq_len, hidden_dim)
        title: Title for visualization

    Returns:
        Dictionary with visualization data
    """
    if embeddings.dim() == 3:
        batch_size, seq_len, hidden_dim = embeddings.shape
    else:
        raise ValueError(f"Expected 3D tensor, got shape {embeddings.shape}")

    # Compute statistics across the embedding dimension
    mean_across_dim = embeddings.mean(dim=-1)  # (batch_size, seq_len)
    std_across_dim = embeddings.std(dim=-1)
    norm_across_seq = torch.norm(embeddings, dim=-1)

    # Compute statistics across the sequence dimension
    mean_across_seq = embeddings.mean(dim=1)  # (batch_size, hidden_dim)
    std_across_seq = embeddings.std(dim=1)

    return {
        'title': title,
        'shape': {
            'batch_size': batch_size,
            'sequence_length': seq_len,
            'hidden_dimension': hidden_dim,
        },
        'statistics': {
            'mean_across_dim': {
                'mean': float(mean_across_dim.mean()),
                'std': float(mean_across_dim.std()),
                'min': float(mean_across_dim.min()),
                'max': float(mean_across_dim.max()),
            },
            'std_across_dim': {
                'mean': float(std_across_dim.mean()),
                'std': float(std_across_dim.std()),
                'min': float(std_across_dim.min()),
                'max': float(std_across_dim.max()),
            },
            'norm_across_seq': {
                'mean': float(norm_across_seq.mean()),
                'std': float(norm_across_seq.std()),
                'min': float(norm_across_seq.min()),
                'max': float(norm_across_seq.max()),
            },
            'global': {
                'mean': float(embeddings.mean()),
                'std': float(embeddings.std()),
                'min': float(embeddings.min()),
                'max': float(embeddings.max()),
                'norm': float(torch.norm(embeddings)),
            }
        }
    }
