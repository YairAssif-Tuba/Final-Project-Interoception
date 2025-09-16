#!/usr/bin/env python3
"""
Visualize the recurrent weight matrix of trained InsulaModule.
Creates a heatmap showing the connectivity pattern and weight magnitudes.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from standalone_insula_module.insula_module import InsulaModule


def parse_args():
    p = argparse.ArgumentParser(description='Visualize InsulaModule recurrent weights')
    p.add_argument('--checkpoint_dir', type=str, required=True,
                   help='Path to checkpoint directory with trained weights')
    p.add_argument('--save_path', type=str, default='weight_heatmap.png',
                   help='Path to save the heatmap plot')
    p.add_argument('--figsize', type=int, nargs=2, default=[12, 10],
                   help='Figure size (width height)')
    return p.parse_args()


def load_config():
    """Load the main config.json to get architecture parameters."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def visualize_weights(checkpoint_dir: str, save_path: str, figsize: tuple):
    """Create weight heatmap visualization."""
    
    # Load config and weights
    config = load_config()
    arch = config['model_architecture']
    n_gINS = arch['n_gINS']
    n_dINS = arch['n_dINS'] 
    n_aINS = arch['n_aINS']
    n_total = arch['n_insula']
    
    print(f"📊 Creating weight heatmap for InsulaModule")
    print(f"   Architecture: gINS={n_gINS}, dINS={n_dINS}, aINS={n_aINS} (total={n_total})")
    
    # Load weights
    weights_path = os.path.join(checkpoint_dir, 'insula_weights_best.pt')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    weights = torch.load(weights_path, map_location='cpu')
    weight_hh = weights['weight_hh'].numpy()  # [N, N]
    
    print(f"   Weight matrix shape: {weight_hh.shape}")
    print(f"   Weight range: [{weight_hh.min():.4f}, {weight_hh.max():.4f}]")
    print(f"   Non-zero weights: {np.count_nonzero(weight_hh)} / {weight_hh.size} ({100*np.count_nonzero(weight_hh)/weight_hh.size:.1f}%)")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Full weight matrix heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(weight_hh, cmap='RdBu_r', aspect='equal', 
                     vmin=-np.abs(weight_hh).max(), vmax=np.abs(weight_hh).max())
    ax1.set_title('Recurrent Weight Matrix (weight_hh)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('From Neuron', fontsize=12)
    ax1.set_ylabel('To Neuron', fontsize=12)
    
    # Add region boundaries and labels
    gINS_end = n_gINS
    dINS_end = n_gINS + n_dINS
    aINS_end = n_total
    
    # Add grid lines to separate regions
    ax1.axhline(gINS_end - 0.5, color='white', linewidth=2)
    ax1.axhline(dINS_end - 0.5, color='white', linewidth=2)
    ax1.axvline(gINS_end - 0.5, color='white', linewidth=2)
    ax1.axvline(dINS_end - 0.5, color='white', linewidth=2)
    
    # Add region labels
    ax1.text(n_gINS/2, -2, 'gINS', ha='center', va='top', fontsize=10, fontweight='bold')
    ax1.text(n_gINS + n_dINS/2, -2, 'dINS', ha='center', va='top', fontsize=10, fontweight='bold')
    ax1.text(n_gINS + n_dINS + n_aINS/2, -2, 'aINS', ha='center', va='top', fontsize=10, fontweight='bold')
    
    ax1.text(-2, n_gINS/2, 'gINS', ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)
    ax1.text(-2, n_gINS + n_dINS/2, 'dINS', ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)
    ax1.text(-2, n_gINS + n_dINS + n_aINS/2, 'aINS', ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Weight Value', fontsize=12)
    
    # Right plot: Binary connectivity pattern
    ax2 = axes[1]
    connectivity = (np.abs(weight_hh) > 1e-6).astype(float)
    im2 = ax2.imshow(connectivity, cmap='Greys', aspect='equal', vmin=0, vmax=1)
    ax2.set_title('Connectivity Pattern (|weight| > 1e-6)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('From Neuron', fontsize=12)
    ax2.set_ylabel('To Neuron', fontsize=12)
    
    # Add same region boundaries
    ax2.axhline(gINS_end - 0.5, color='red', linewidth=2)
    ax2.axhline(dINS_end - 0.5, color='red', linewidth=2)
    ax2.axvline(gINS_end - 0.5, color='red', linewidth=2)
    ax2.axvline(dINS_end - 0.5, color='red', linewidth=2)
    
    # Add region labels
    ax2.text(n_gINS/2, -2, 'gINS', ha='center', va='top', fontsize=10, fontweight='bold')
    ax2.text(n_gINS + n_dINS/2, -2, 'dINS', ha='center', va='top', fontsize=10, fontweight='bold')
    ax2.text(n_gINS + n_dINS + n_aINS/2, -2, 'aINS', ha='center', va='top', fontsize=10, fontweight='bold')
    
    ax2.text(-2, n_gINS/2, 'gINS', ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)
    ax2.text(-2, n_gINS + n_dINS/2, 'dINS', ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)
    ax2.text(-2, n_gINS + n_dINS + n_aINS/2, 'aINS', ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Connection Exists', fontsize=12)
    
    # Add connectivity statistics
    stats_text = f"""Connectivity Statistics:
gINS→gINS: {np.count_nonzero(connectivity[:n_gINS, :n_gINS])} / {n_gINS*n_gINS}
gINS→dINS: {np.count_nonzero(connectivity[:n_gINS, n_gINS:n_gINS+n_dINS])} / {n_gINS*n_dINS}
dINS→dINS: {np.count_nonzero(connectivity[n_gINS:n_gINS+n_dINS, n_gINS:n_gINS+n_dINS])} / {n_dINS*n_dINS}
dINS→aINS: {np.count_nonzero(connectivity[n_gINS:n_gINS+n_dINS, n_gINS+n_dINS:])} / {n_dINS*n_aINS}
aINS→aINS: {np.count_nonzero(connectivity[n_gINS+n_dINS:, n_gINS+n_dINS:])} / {n_aINS*n_aINS}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Weight heatmap saved to: {save_path}")
    
    return weight_hh, connectivity


def main():
    args = parse_args()
    
    print("=== InsulaModule Weight Visualization ===\n")
    
    weight_hh, connectivity = visualize_weights(
        args.checkpoint_dir, 
        args.save_path, 
        tuple(args.figsize)
    )
    
    # Print additional statistics
    print(f"\n📈 Weight Statistics:")
    print(f"   Mean absolute weight: {np.abs(weight_hh).mean():.6f}")
    print(f"   Std of weights: {weight_hh.std():.6f}")
    print(f"   Diagonal mean: {np.diag(weight_hh).mean():.6f}")
    print(f"   Off-diagonal mean: {(weight_hh - np.diag(np.diag(weight_hh))).mean():.6f}")


if __name__ == "__main__":
    main()
