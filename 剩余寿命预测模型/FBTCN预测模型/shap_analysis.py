#!/usr/bin/env python3
"""
Standalone SHAP analysis for the Bayesian TCN (FBTCN) RUL model.

What it does
------------
1) Load a trained FBTCN checkpoint and the scaler-normalized bearing data.
2) Flatten sequence features and run SHAP KernelExplainer on a subset.
3) Save raw SHAP values, per-feature mean |SHAP|, and summary plots.

This script leaves existing training/eval code untouched and only reads
the saved weights + dataset files under `datasetresult/`.
"""

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch

try:
    import shap
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing deps for SHAP script. Please install:\n"
        "  pip install shap matplotlib pandas"
    ) from exc

# Allow importing the existing helpers in this folder
sys.path.append(os.path.dirname(__file__))
from auto_train_fbtcn_sa import get_data_dir, load_bearing_data  # type: ignore
from fbtcn_sa_model import BayesianTCN  # type: ignore


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(cfg: Dict, model_path: str, device: torch.device) -> BayesianTCN:
    # Load checkpoint first to infer model parameters if needed
    state = torch.load(model_path, map_location=device)
    
    # Try to infer attention_dim and TCN output channels from checkpoint
    attention_dim = cfg.get("attention_dim", 1)
    num_channels = cfg["num_channels"].copy() if isinstance(cfg["num_channels"], list) else cfg["num_channels"]
    
    try:
        # Infer attention_dim from attention.query.weight
        if "attention.query.weight" in state:
            checkpoint_attention_dim = state["attention.query.weight"].shape[0]
            if attention_dim != checkpoint_attention_dim:
                print(f"[INFO] Config attention_dim ({attention_dim}) != checkpoint attention_dim ({checkpoint_attention_dim}). Using checkpoint value.")
                attention_dim = checkpoint_attention_dim
        
        # Infer TCN output channels from the last TCN layer's conv1 output channels
        # Find the last network layer (network.N where N is the highest index)
        network_keys = [k for k in state.keys() if k.startswith("network.") and ".conv1.mu_kernel" in k]
        if network_keys:
            # Sort to find the last layer (highest index)
            network_keys.sort(key=lambda x: int(x.split(".")[1]))
            last_layer_key = network_keys[-1]
            # conv1.mu_kernel shape is [out_channels, in_channels, kernel_size]
            checkpoint_tcn_output_channels = state[last_layer_key].shape[0]
            
            # Adjust num_channels to match checkpoint's TCN output channels
            if isinstance(num_channels, list) and len(num_channels) > 0:
                config_tcn_output_channels = num_channels[-1]
                if config_tcn_output_channels != checkpoint_tcn_output_channels:
                    print(f"[INFO] Config TCN output channels ({config_tcn_output_channels}) != checkpoint TCN output channels ({checkpoint_tcn_output_channels}). Adjusting num_channels.")
                    # Create a new list with the last element adjusted
                    num_channels = num_channels.copy()
                    num_channels[-1] = checkpoint_tcn_output_channels
            elif not isinstance(num_channels, list):
                # If num_channels is not a list, create one based on checkpoint
                num_channels = [checkpoint_tcn_output_channels]
                print(f"[INFO] Creating num_channels list from checkpoint: {num_channels}")
    except Exception as e:
        print(f"[WARN] Could not infer parameters from checkpoint: {e}. Using config values.")
    
    model = BayesianTCN(
        input_dim=cfg["input_dim"],
        num_channels=num_channels,
        attention_dim=attention_dim,
        kernel_size=cfg["kernel_size"],
        conv_posterior_rho_init=cfg.get("conv_posterior_rho_init", -2),
        output_posterior_rho_init=cfg.get("output_posterior_rho_init", -1),
        dropout=cfg.get("dropout", 0.3),
        output_dim=cfg.get("output_dim", 1),
        attention_mode=cfg.get("attention_mode", "self"),
    ).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys while loading state_dict: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading state_dict: {unexpected}")

    model.eval()
    return model


def prepare_data(
    bearing_name: str, data_dir: str, sample_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    data, labels = load_bearing_data([bearing_name], data_dir)
    if len(data) == 0:
        raise ValueError(
            f"No data found for bearing '{bearing_name}' in {data_dir}. "
            "Check the file prefix (e.g., c1_Bearing1_1_labeled)."
        )
    data = data.to(torch.float32)
    labels = labels.to(torch.float32)
    sample_size = min(sample_size, len(data))
    return data[:sample_size], labels[:sample_size]


def compute_shap(
    model: torch.nn.Module,
    data_flat: np.ndarray,
    seq_len: int,
    input_dim: int,
    background_size: int,
    nsamples: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run SHAP KernelExplainer on flattened inputs.
    """
    background = data_flat[:background_size]

    def model_predict(flat_batch: np.ndarray) -> np.ndarray:
        batch = torch.tensor(flat_batch, dtype=torch.float32, device=device)
        batch = batch.view(batch.shape[0], seq_len, input_dim)
        with torch.no_grad():
            mu, _, _ = model(batch)
        return mu.squeeze(-1).detach().cpu().numpy()

    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(data_flat, nsamples=nsamples)
    # KernelExplainer returns list when model has multiple outputs; keep the first.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    return np.asarray(shap_values)


def save_outputs(
    shap_values: np.ndarray,
    data_flat: np.ndarray,
    seq_len: int,
    input_dim: int,
    output_dir: str,
    bearing_name: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义特征名称（11个特征）
    base_feature_names = [
        'Kurtosis', 'Fractal Dimension', 'Peak Factor',
        'Energy Ratio', 'Spectral Flatness', 'Mean', 
        'Variance', 'Skewness', 'Peak Vibration', 'DE', 'FFT Mean'
    ]
    
    # 如果input_dim与特征名称数量不匹配，使用通用名称或截断/扩展
    if input_dim <= len(base_feature_names):
        feature_names_list = base_feature_names[:input_dim]
    else:
        # 如果input_dim大于11，使用通用名称
        feature_names_list = base_feature_names + [f"Feature_{i}" for i in range(len(base_feature_names), input_dim)]
    
    # 创建完整的特征名称（包含时间步）
    feature_names = [f"t{t}_{feat}" for t in range(seq_len) for feat in feature_names_list]

    np.save(os.path.join(output_dir, f"{bearing_name}_shap_values.npy"), shap_values)

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_abs_matrix = mean_abs.reshape(seq_len, input_dim)
    pd.DataFrame(
        mean_abs_matrix,
        index=[f"t{t}" for t in range(seq_len)],
        columns=feature_names_list,  # 使用特征名称而不是 f0, f1, ...
    ).to_csv(os.path.join(output_dir, f"{bearing_name}_mean_abs_shap.csv"), encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        data_flat,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{bearing_name}_shap_summary_bar.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_values,
        data_flat,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{bearing_name}_shap_summary_scatter.png"), dpi=200)
    plt.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SHAP values for FBTCN.")
    parser.add_argument(
        "--config",
        default="/mnt/uq_aware_rul_prediction4bearing-main/config/ablation/A_fbtcn_config_ablation_no_rds_sa_revise.json",
        help="Config JSON used to build the model.",
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds_sa_revise/xjtu_to_xjtu/Bearing1_best_model_fbtcn.pt",
        help="Trained FBTCN checkpoint (.pt).",
    )
    parser.add_argument(
        "--dataset-type",
        default="xjtu_made",
        help="Dataset folder name under datasetresult/ (e.g., xjtu_made, femto_made).",
    )
    parser.add_argument(
        "--bearing-name",
        default="c1_Bearing1_5_labeled",
        help="Bearing prefix to load (substring that appears in *_data / *_label filenames).",
    )
    parser.add_argument(
        "--output-dir",
        default="./shap_outputs",
        help="Where to save SHAP values and plots.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="How many samples to explain (truncated if data is smaller).",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=50,
        help="Background sample count for KernelExplainer.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=200,
        help="Number of SHAP permutations (higher = more accurate, slower).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    cfg = load_config(args.config)
    data_dir = get_data_dir(args.dataset_type)

    data, labels = prepare_data(args.bearing_name, data_dir, args.sample_size)
    seq_len, input_dim = data.shape[1], data.shape[2]
    data_np = data.cpu().numpy()
    data_flat = data_np.reshape(data_np.shape[0], -1)

    if args.background_size > len(data_flat):
        args.background_size = len(data_flat)

    model = build_model(cfg, args.model_path, device)

    shap_values = compute_shap(
        model=model,
        data_flat=data_flat,
        seq_len=seq_len,
        input_dim=input_dim,
        background_size=args.background_size,
        nsamples=args.nsamples,
        device=device,
    )

    save_outputs(
        shap_values=shap_values,
        data_flat=data_flat,
        seq_len=seq_len,
        input_dim=input_dim,
        output_dir=args.output_dir,
        bearing_name=args.bearing_name,
    )

    print(
        f"SHAP finished. Saved to {args.output_dir} "
        f"(bearing={args.bearing_name}, samples={len(data_flat)}, device={device})."
    )


if __name__ == "__main__":
    main()
