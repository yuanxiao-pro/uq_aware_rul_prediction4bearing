#!/usr/bin/env python3
"""
Visualize RUL distribution from 200 forward passes.

Reads a CSV file with 200 forward pass results, computes mean and variance
for a given time point index, and plots a Gaussian distribution with the
true RUL and mean RUL as vertical lines.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.stats import norm


def load_csv_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load CSV data and extract y_true, y_pred samples, and y_var.
    
    Returns:
        y_true: True RUL values [n_timepoints]
        y_pred_samples: Predicted RUL samples [n_timepoints, n_samples=200]
        y_var: Variance values [n_timepoints]
    """
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    
    # First column is y_true
    y_true = data[:, 0]
    
    # Last column is y_var
    y_var = data[:, -1]
    
    # Columns 1 to 200 are y_pred_sample_0 to y_pred_sample_199
    y_pred_samples = data[:, 1:201]
    
    return y_true, y_pred_samples, y_var


def compute_statistics(y_pred_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and variance from samples.
    
    Args:
        y_pred_samples: [n_timepoints, n_samples] array of predictions
    
    Returns:
        mean: [n_timepoints] mean values
        std: [n_timepoints] standard deviation values
    """
    mean = np.mean(y_pred_samples, axis=1)
    std = np.std(y_pred_samples, axis=1, ddof=1)  # Sample std (Bessel's correction)
    return mean, std


def plot_rul_distribution(
    y_true: float,
    y_pred_mean: float,
    y_pred_std: float,
    time_idx: int,
    output_path: str = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """Plot Gaussian distribution with true RUL and mean RUL as vertical lines.
    
    Args:
        y_true: True RUL value
        y_pred_mean: Mean of predicted RUL samples
        y_pred_std: Standard deviation of predicted RUL samples
        time_idx: Time point index (for title)
        output_path: Path to save figure (if None, show interactively)
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create x-axis range for PDF (mean ± 4 std for good coverage)
    # x_min = min(y_true, y_pred_mean) - 4 * y_pred_std
    # x_max = max(y_true, y_pred_mean) + 4 * y_pred_std
    x_min = 0
    x_max = 1
    x = np.linspace(x_min, x_max, 1000)
    
    # Compute Gaussian PDF
    pdf = norm.pdf(x, loc=y_pred_mean, scale=y_pred_std)
    
    # Plot PDF as filled area (shaded)
    ax.fill_between(x, pdf, alpha=0.3, color='blue', label=f'Gaussian PDF (μ={y_pred_mean:.4f}, σ={y_pred_std:.4f})')
    
    # Plot PDF line
    ax.plot(x, pdf, 'b-', linewidth=2, alpha=0.7)
    
    # Draw vertical line for true RUL
    ax.axvline(y_true, color='red', linestyle='--', linewidth=2, label=f'True RUL = {y_true:.4f}')
    
    # Draw vertical line for mean RUL
    ax.axvline(y_pred_mean, color='green', linestyle='--', linewidth=2, label=f'Mean RUL = {y_pred_mean:.4f}')
    
    # Add text annotations
    ax.text(y_true, ax.get_ylim()[1] * 0.95, f'True: {y_true:.4f}', 
            rotation=90, verticalalignment='top', horizontalalignment='right',
            color='red', fontsize=10, fontweight='bold')
    ax.text(y_pred_mean, ax.get_ylim()[1] * 0.85, f'Mean: {y_pred_mean:.4f}', 
            rotation=90, verticalalignment='top', horizontalalignment='right',
            color='green', fontsize=10, fontweight='bold')
    
    # Labels and title
    ax.set_xlabel('RUL (Remaining Useful Life)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'RUL Distribution at Time Point {time_idx}\n'
                 f'Mean={y_pred_mean:.4f}, Std={y_pred_std:.4f}, True={y_true:.4f}', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    
    # Set x-axis ticks with interval of 0.2
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Figure saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize RUL distribution from 200 forward passes."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_xjtu/c2_Bearing2_2_labeled_fpt_scaler_fbtcn_result.csv",
        help="Path to CSV file with forward pass results.",
    )
    parser.add_argument(
        "--time-idx",
        type=int,
        default=67,
        help="Time point index (0-based) to visualize.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for figure (if not specified, show interactively).",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[10, 6],
        help="Figure size (width height). Default: 10 6",
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"[INFO] Loading CSV from {args.csv_path}...")
    y_true, y_pred_samples, y_var = load_csv_data(args.csv_path)
    
    n_timepoints = len(y_true)
    print(f"[INFO] Loaded {n_timepoints} time points with {y_pred_samples.shape[1]} samples each.")
    
    # Validate time index
    if args.time_idx < 0 or args.time_idx >= n_timepoints:
        print(f"[ERROR] Time index {args.time_idx} is out of range [0, {n_timepoints-1}]")
        sys.exit(1)
    
    # Extract data for the specified time point
    true_rul = y_true[args.time_idx]
    pred_samples = y_pred_samples[args.time_idx, :]
    
    # Compute statistics
    mean = np.mean(pred_samples)
    # Use y_var column for variance, then compute std = sqrt(variance)
    variance_from_csv = y_var[args.time_idx]
    std = np.sqrt(variance_from_csv)  # Standard deviation from y_var column
    
    print(f"\n[INFO] Time Point {args.time_idx}:")
    print(f"  True RUL: {true_rul:.6f}")
    print(f"  Mean RUL: {mean:.6f}")
    print(f"  Std RUL (from y_var):  {std:.6f}")
    print(f"  Variance (from y_var): {variance_from_csv:.6f}")
    print(f"  Std from samples: {np.std(pred_samples, ddof=1):.6f}")
    
    # Determine output path
    if args.output is None:
        csv_name = Path(args.csv_path).stem
        output_path = f"rul_distribution_{csv_name}_t{args.time_idx}.png"
    else:
        output_path = args.output
    
    # Plot
    plot_rul_distribution(
        y_true=true_rul,
        y_pred_mean=mean,
        y_pred_std=std,
        time_idx=args.time_idx,
        output_path=output_path,
        figsize=tuple(args.figsize),
    )
    
    print(f"[INFO] Visualization complete!")


if __name__ == "__main__":
    main()
