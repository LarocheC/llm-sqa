"""
Visualization utilities for SALMONN model introspection.

Provides functions to create plots and visualizations of:
- Encoder outputs
- Attention patterns
- Token embeddings
- Activation distributions
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import torch
from pathlib import Path


def plot_activation_statistics(
    introspection_data: Dict[str, Any],
    output_path: Optional[Path] = None,
    figsize: tuple = (15, 10)
) -> plt.Figure:
    """
    Plot statistics of activations across all pipeline stages.

    Args:
        introspection_data: Introspection data dictionary
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    stage_names = []
    means = []
    stds = []
    norms = []

    for component_name, component_data in introspection_data.items():
        for activation in component_data.get('activations', []):
            if 'mean' in activation:
                stage_names.append(component_name)
                means.append(activation['mean'])
                stds.append(activation['std'])
                norms.append(activation.get('norm', 0))

    if not stage_names:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No activation data available',
                ha='center', va='center', fontsize=16)
        return fig

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Plot means
    axes[0].bar(range(len(stage_names)), means, color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Mean Activation')
    axes[0].set_title('Mean Activation Values Across Pipeline Stages')
    axes[0].set_xticks(range(len(stage_names)))
    axes[0].set_xticklabels(stage_names, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)

    # Plot stds
    axes[1].bar(range(len(stage_names)), stds, color='coral', alpha=0.7)
    axes[1].set_ylabel('Std Deviation')
    axes[1].set_title('Standard Deviation Across Pipeline Stages')
    axes[1].set_xticks(range(len(stage_names)))
    axes[1].set_xticklabels(stage_names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)

    # Plot norms
    axes[2].bar(range(len(stage_names)), norms, color='mediumseagreen', alpha=0.7)
    axes[2].set_ylabel('L2 Norm')
    axes[2].set_title('L2 Norm Across Pipeline Stages')
    axes[2].set_xticks(range(len(stage_names)))
    axes[2].set_xticklabels(stage_names, rotation=45, ha='right')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_embedding_heatmap(
    embedding: torch.Tensor,
    title: str = "Embedding Heatmap",
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 8),
    max_seq_len: int = 100,
    max_dim: int = 256
) -> plt.Figure:
    """
    Plot heatmap of embedding tensor.

    Args:
        embedding: Embedding tensor (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
        title: Plot title
        output_path: Optional path to save figure
        figsize: Figure size
        max_seq_len: Maximum sequence length to plot
        max_dim: Maximum embedding dimension to plot

    Returns:
        Matplotlib figure
    """
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()

    # Handle batch dimension
    if embedding.ndim == 3:
        embedding = embedding[0]  # Take first batch

    # Truncate if too large
    seq_len, hidden_dim = embedding.shape
    if seq_len > max_seq_len:
        embedding = embedding[:max_seq_len, :]
    if hidden_dim > max_dim:
        # Sample dimensions evenly
        indices = np.linspace(0, hidden_dim - 1, max_dim, dtype=int)
        embedding = embedding[:, indices]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(embedding.T, aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(embedding).max(), vmax=np.abs(embedding).max())

    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Embedding Dimension')
    ax.set_title(title)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Activation Value')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_embedding_distribution(
    embedding: torch.Tensor,
    title: str = "Embedding Distribution",
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot distribution of embedding values.

    Args:
        embedding: Embedding tensor
        title: Plot title
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()

    # Flatten embedding
    values = embedding.flatten()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    axes[0].hist(values, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Activation Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{title} - Histogram')
    axes[0].grid(axis='y', alpha=0.3)

    # Add statistics text
    stats_text = f'Mean: {values.mean():.4f}\n'
    stats_text += f'Std: {values.std():.4f}\n'
    stats_text += f'Min: {values.min():.4f}\n'
    stats_text += f'Max: {values.max():.4f}'
    axes[0].text(0.98, 0.98, stats_text,
                transform=axes[0].transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    # Box plot
    axes[1].boxplot([values], vert=True)
    axes[1].set_ylabel('Activation Value')
    axes[1].set_title(f'{title} - Box Plot')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_sequence_norms(
    embedding: torch.Tensor,
    title: str = "Sequence Norms",
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 5)
) -> plt.Figure:
    """
    Plot L2 norms across sequence positions.

    Args:
        embedding: Embedding tensor (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
        title: Plot title
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu()

    # Handle batch dimension
    if embedding.dim() == 3:
        embedding = embedding[0]  # Take first batch

    # Compute norms
    norms = torch.norm(embedding, dim=-1).numpy()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Line plot
    axes[0].plot(norms, linewidth=2, color='steelblue')
    axes[0].set_xlabel('Sequence Position')
    axes[0].set_ylabel('L2 Norm')
    axes[0].set_title(f'{title} - L2 Norms')
    axes[0].grid(alpha=0.3)

    # Histogram of norms
    axes[1].hist(norms, bins=50, alpha=0.7, color='coral', edgecolor='black')
    axes[1].set_xlabel('L2 Norm')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Norms')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_pipeline_flow(
    introspection_data: Dict[str, Any],
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 8)
) -> plt.Figure:
    """
    Visualize the data flow through the pipeline stages.

    Args:
        introspection_data: Introspection data dictionary
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Extract shapes from each stage
    stages = []
    shapes = []

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
            if stage_info.get('activations'):
                activation = stage_info['activations'][0]
                if 'shape' in activation:
                    stages.append(stage_name)
                    shapes.append(activation['shape'])

    if not stages:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No pipeline data available',
                ha='center', va='center', fontsize=16)
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    # Plot as a flow diagram
    y_positions = np.linspace(0.9, 0.1, len(stages))

    for idx, (stage, shape, y_pos) in enumerate(zip(stages, shapes, y_positions)):
        # Draw box
        box_width = 0.3
        box_height = 0.08
        rect = plt.Rectangle((0.35, y_pos - box_height/2), box_width, box_height,
                            facecolor='lightblue', edgecolor='steelblue', linewidth=2)
        ax.add_patch(rect)

        # Add text
        shape_str = ' × '.join(map(str, shape))
        ax.text(0.5, y_pos, f'{stage}\n{shape_str}',
               ha='center', va='center', fontsize=10, weight='bold')

        # Draw arrow to next stage
        if idx < len(stages) - 1:
            ax.arrow(0.5, y_pos - box_height/2 - 0.01,
                    0, -(y_positions[idx] - y_positions[idx+1]) + box_height + 0.02,
                    head_width=0.03, head_length=0.02,
                    fc='gray', ec='gray', alpha=0.7)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('SALMONN Pipeline Data Flow', fontsize=16, weight='bold', pad=20)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def create_introspection_visualizations(
    introspection_data: Dict[str, Any],
    encoder_outputs: Dict[str, torch.Tensor],
    output_dir: Path
) -> List[Path]:
    """
    Create all introspection visualizations and save to directory.

    Args:
        introspection_data: Introspection data dictionary
        encoder_outputs: Dictionary of encoder outputs
        output_dir: Directory to save visualizations

    Returns:
        List of paths to saved figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    # 1. Activation statistics
    try:
        fig = plot_activation_statistics(
            introspection_data,
            output_path=output_dir / 'activation_statistics.png'
        )
        plt.close(fig)
        saved_paths.append(output_dir / 'activation_statistics.png')
    except Exception as e:
        print(f"Error creating activation statistics plot: {e}")

    # 2. Pipeline flow
    try:
        fig = plot_pipeline_flow(
            introspection_data,
            output_path=output_dir / 'pipeline_flow.png'
        )
        plt.close(fig)
        saved_paths.append(output_dir / 'pipeline_flow.png')
    except Exception as e:
        print(f"Error creating pipeline flow plot: {e}")

    # 3. Encoder output visualizations
    for encoder_name, encoder_output in encoder_outputs.items():
        try:
            # Get the actual tensor
            if isinstance(encoder_output, dict) and 'last_hidden_state' in encoder_output:
                tensor = encoder_output['last_hidden_state']
            elif isinstance(encoder_output, tuple):
                tensor = encoder_output[0]
            else:
                tensor = encoder_output

            if isinstance(tensor, torch.Tensor):
                # Heatmap
                fig = plot_embedding_heatmap(
                    tensor,
                    title=f'{encoder_name} Encoder Output',
                    output_path=output_dir / f'{encoder_name}_heatmap.png'
                )
                plt.close(fig)
                saved_paths.append(output_dir / f'{encoder_name}_heatmap.png')

                # Distribution
                fig = plot_embedding_distribution(
                    tensor,
                    title=f'{encoder_name} Encoder Output',
                    output_path=output_dir / f'{encoder_name}_distribution.png'
                )
                plt.close(fig)
                saved_paths.append(output_dir / f'{encoder_name}_distribution.png')

                # Sequence norms
                fig = plot_sequence_norms(
                    tensor,
                    title=f'{encoder_name} Encoder Output',
                    output_path=output_dir / f'{encoder_name}_norms.png'
                )
                plt.close(fig)
                saved_paths.append(output_dir / f'{encoder_name}_norms.png')
        except Exception as e:
            print(f"Error creating {encoder_name} visualizations: {e}")

    return saved_paths
