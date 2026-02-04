#!/usr/bin/env python3
"""
t-SNE and UMAP visualization for bearing RUL prediction data.

What it does
------------
1) Load bearing data and optionally a trained model.
2) Extract features (raw or from model intermediate layers).
3) Apply t-SNE and/or UMAP for dimensionality reduction.
4) Visualize with different color schemes (by bearing, by label, by dataset).
5) Save visualization plots (individual and combined).

Usage
-----
# Visualize raw features with t-SNE only
python tsne_visualization.py --mode raw --bearing-names c1_Bearing1_1_labeled c1_Bearing1_2_labeled

# Visualize with both t-SNE and UMAP
python tsne_visualization.py --mode tcn --use-umap --config config/fbtcn_config.json --model-path best_model_fbtcn_stable.pt

# Visualize attention features with UMAP
python tsne_visualization.py --mode attention --use-umap --config config/fbtcn_config.json --model-path best_model_fbtcn_stable.pt
"""

import argparse
import csv
import itertools
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import load as joblib_load

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.font_manager import FontProperties
    from matplotlib.gridspec import GridSpec
except ImportError as exc:
    raise SystemExit(
        "Missing deps for t-SNE script. Please install:\n"
        "  pip install scikit-learn matplotlib"
    ) from exc

try:
    import umap
except ImportError:
    umap = None
    print("[WARN] UMAP not installed. Install with: pip install umap-learn")

# Allow importing the existing helpers in this folder
sys.path.append(os.path.dirname(__file__))
from auto_train_fbtcn_sa import get_data_dir, load_bearing_data  # type: ignore
from fbtcn_sa_model import BayesianTCN  # type: ignore

# Set global font to Times New Roman (新罗马) for all plots
plt.rcParams["font.family"] = "Times New Roman"


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(cfg: Dict, model_path: str, device: torch.device) -> BayesianTCN:
    """Build and load trained model."""
    model = BayesianTCN(
        input_dim=cfg["input_dim"],
        num_channels=cfg["num_channels"],
        attention_dim=cfg.get("attention_dim", 1),
        kernel_size=cfg["kernel_size"],
        conv_posterior_rho_init=cfg.get("conv_posterior_rho_init", -2),
        output_posterior_rho_init=cfg.get("output_posterior_rho_init", -1),
        dropout=cfg.get("dropout", 0.3),
        output_dim=cfg.get("output_dim", 1),
        attention_mode=cfg.get("attention_mode", "self"),
    ).to(device)

    state = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys while loading state_dict: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading state_dict: {unexpected}")

    model.eval()
    return model


def extract_features_raw(data: torch.Tensor) -> np.ndarray:
    """Extract raw features by flattening sequence data."""
    # Handle different data shapes:
    # - [batch, seq_len, input_dim] -> flatten to [batch, seq_len * input_dim]
    # - [batch, input_dim] -> keep as [batch, input_dim]
    # - [batch, 1, input_dim] -> flatten to [batch, input_dim]
    if len(data.shape) == 3:
        # [batch, seq_len, input_dim]
        return data.view(data.shape[0], -1).cpu().numpy()
    elif len(data.shape) == 2:
        # [batch, input_dim] - already 2D
        return data.cpu().numpy()
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")


def extract_features_tcn(model: BayesianTCN, data: torch.Tensor, device: torch.device) -> np.ndarray:
    """Extract features from TCN layers (before attention)."""
    model.eval()
    
    with torch.no_grad():
        # Permute to [batch, input_dim, seq_len]
        x = data.permute(0, 2, 1).to(device)
        
        # Pass through TCN layers
        for block in model.network:
            x, _ = block(x)
        
        # Global average pooling: [batch, channels, seq_len] -> [batch, channels]
        x = model.avgpool(x).squeeze(-1)
        
        return x.cpu().numpy()


def extract_features_attention(model: BayesianTCN, data: torch.Tensor, device: torch.device) -> np.ndarray:
    """Extract features from attention layer output (before pooling to preserve sequence information)."""
    model.eval()
    
    with torch.no_grad():
        # 处理不同形状的输入数据
        original_shape = data.shape
        if len(data.shape) == 2:
            # 如果数据是2D [batch, input_dim]，需要添加seq_len维度
            # 假设seq_len=1，reshape为 [batch, 1, input_dim]
            data = data.unsqueeze(1)  # [batch, input_dim] -> [batch, 1, input_dim]
        
        # Permute to [batch, input_dim, seq_len]
        x = data.permute(0, 2, 1).to(device)
        
        # Pass through TCN layers
        for block in model.network:
            x, _ = block(x)
        # 此时 x shape: [batch, channels, seq_len]
        
        # 保存TCN输出，以备特征维度不足时使用
        tcn_features_flat = x.view(x.size(0), -1)  # [batch, channels * seq_len]
        
        # Pass through attention layer (按照新的模型forward方法的逻辑)
        if model.attention is not None:
            # 转置为序列格式: [batch, channels, seq_len] -> [batch, seq_len, channels]
            # 不再对channels维度做平均，保留所有通道信息
            x_attn = x.permute(0, 2, 1)  # [batch, channels, seq_len] -> [batch, seq_len, channels]
            
            # 应用自注意力: [batch, seq_len, channels] -> [batch, seq_len, attention_dim]
            x_attn = model.attention(x_attn)  # [batch, seq_len, attention_dim]
            
            # 展平attention输出: [batch, seq_len, attention_dim] -> [batch, seq_len * attention_dim]
            attention_features_flat = x_attn.view(x_attn.size(0), -1)
            
            # 检查特征维度是否足够进行t-SNE
            if attention_features_flat.shape[1] >= 2:
                # 如果attention特征维度足够，使用attention特征
                x = attention_features_flat
            else:
                # 如果attention特征维度不足（如seq_len=1且attention_dim=1），使用TCN特征
                # TCN特征肯定有足够的维度（channels * seq_len >= channels >= 32）
                print(f"[INFO] Attention features have only {attention_features_flat.shape[1]} dimension(s). Using TCN features instead.")
                x = tcn_features_flat
        else:
            # 如果没有注意力层，使用TCN输出并展平
            x = tcn_features_flat  # [batch, channels * seq_len]
        
        return x.cpu().numpy()


def apply_tsne(features: np.ndarray, n_components: int = 2, perplexity: float = 30.0, 
               random_state: int = 42, max_samples: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply t-SNE dimensionality reduction."""
    # Check feature dimensions
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)
    
    n_features = features.shape[1] if len(features.shape) > 1 else 1
    
    # t-SNE requires at least n_components features
    if n_features < n_components:
        raise ValueError(
            f"Cannot apply t-SNE with n_components={n_components} when features have only {n_features} dimension(s).\n"
            f"Feature shape: {features.shape}\n"
            f"Solution: Use '--mode tcn' instead of '--mode attention' to get higher-dimensional features, "
            f"or increase attention_dim in your config file."
        )
    
    sample_indices = None
    if max_samples is not None and len(features) > max_samples:
        # Randomly sample for faster computation
        np.random.seed(random_state)
        sample_indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[sample_indices]
        print(f"[INFO] Sampling {max_samples} samples for t-SNE (from {len(features)} total)")
    
    print(f"[INFO] Applying t-SNE to {len(features)} samples with shape {features.shape}...")
    print(f"[INFO] Perplexity: {perplexity}, Random state: {random_state}")
    
    # Adjust perplexity if needed
    n_samples = len(features)
    if perplexity >= n_samples:
        perplexity = max(1, n_samples - 1)
        print(f"[WARN] Adjusted perplexity to {perplexity}")
    
    # Use max_iter for sklearn >= 1.5, fallback to n_iter for older versions
    try:
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            max_iter=1000,
            verbose=1
        )
    except TypeError:
        # Fallback for older sklearn versions
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=1000,
            verbose=1
        )
    
    embeddings = tsne.fit_transform(features)
    print(f"[INFO] t-SNE completed. Embeddings shape: {embeddings.shape}")
    
    return embeddings, sample_indices


def apply_umap(features: np.ndarray, n_components: int = 2, n_neighbors: int = 15, 
                min_dist: float = 0.1, random_state: int = 42, max_samples: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply UMAP dimensionality reduction."""
    if umap is None:
        raise ImportError("UMAP is not installed. Install with: pip install umap-learn")
    
    # Check feature dimensions
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)
    
    n_features = features.shape[1] if len(features.shape) > 1 else 1
    
    # UMAP requires at least n_components features
    if n_features < n_components:
        raise ValueError(
            f"Cannot apply UMAP with n_components={n_components} when features have only {n_features} dimension(s).\n"
            f"Feature shape: {features.shape}\n"
            f"Solution: Use '--mode tcn' instead of '--mode attention' to get higher-dimensional features, "
            f"or increase attention_dim in your config file."
        )
    
    sample_indices = None
    if max_samples is not None and len(features) > max_samples:
        # Randomly sample for faster computation
        np.random.seed(random_state)
        sample_indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[sample_indices]
        print(f"[INFO] Sampling {max_samples} samples for UMAP (from {len(features)} total)")
    
    print(f"[INFO] Applying UMAP to {len(features)} samples with shape {features.shape}...")
    print(f"[INFO] n_neighbors: {n_neighbors}, min_dist: {min_dist}, Random state: {random_state}")
    
    # Adjust n_neighbors if needed
    n_samples = len(features)
    if n_neighbors >= n_samples:
        n_neighbors = max(2, n_samples - 1)
        print(f"[WARN] Adjusted n_neighbors to {n_neighbors}")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=True
    )
    
    embeddings = reducer.fit_transform(features)
    print(f"[INFO] UMAP completed. Embeddings shape: {embeddings.shape}")
    
    return embeddings, sample_indices


def _as_2d_float64(x: np.ndarray) -> np.ndarray:
    """Ensure features are 2D float64 array for metric computation."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x.astype(np.float64, copy=False)


def _pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute squared euclidean distance matrix between rows of x and y."""
    # (x - y)^2 = x^2 + y^2 - 2xy
    x_norm = np.sum(x * x, axis=1, keepdims=True)  # [n, 1]
    y_norm = np.sum(y * y, axis=1, keepdims=True).T  # [1, m]
    d2 = x_norm + y_norm - 2.0 * (x @ y.T)
    return np.maximum(d2, 0.0)


def compute_mmd(
    x: np.ndarray,
    y: np.ndarray,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
    unbiased: bool = True,
    use_median_heuristic: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Compute Maximum Mean Discrepancy (MMD).

    - **RBF kernel**: k(a,b) = exp(-gamma * ||a-b||^2)
    - **Linear kernel**: k(a,b) = a^T b

    Returns (MMD^2, diagnostics_dict) where MMD^2 is non-negative.
    """
    x = _as_2d_float64(x)
    y = _as_2d_float64(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"MMD requires same feature dim, got {x.shape[1]} vs {y.shape[1]}")
    n = x.shape[0]
    m = y.shape[0]
    if n < 2 or m < 2:
        return float("nan"), {}

    diagnostics = {}
    
    if kernel == "rbf":
        d = x.shape[1]
        if gamma is None:
            if use_median_heuristic:
                # Median heuristic: gamma = 1 / median(pairwise_distances^2)
                # Combine both datasets for more robust estimate
                combined = np.vstack([x, y])
                dists_sq = _pairwise_sq_dists(combined[:min(100, len(combined))], 
                                             combined[:min(100, len(combined))])
                # Get upper triangle (excluding diagonal)
                triu_indices = np.triu_indices_from(dists_sq, k=1)
                median_dist_sq = np.median(dists_sq[triu_indices])
                if median_dist_sq > 0:
                    gamma = 1.0 / median_dist_sq
                else:
                    gamma = 1.0 / max(1, d)
                diagnostics["gamma_median_heuristic"] = gamma
                diagnostics["median_dist_sq"] = float(median_dist_sq)
            else:
                gamma = 1.0 / max(1, d)
        diagnostics["gamma"] = gamma
        
        k_xx = np.exp(-gamma * _pairwise_sq_dists(x, x))
        k_yy = np.exp(-gamma * _pairwise_sq_dists(y, y))
        k_xy = np.exp(-gamma * _pairwise_sq_dists(x, y))
        
        # Diagnostic: mean kernel values
        np.fill_diagonal(k_xx, np.nan)
        np.fill_diagonal(k_yy, np.nan)
        diagnostics["mean_k_xx"] = float(np.nanmean(k_xx))
        diagnostics["mean_k_yy"] = float(np.nanmean(k_yy))
        diagnostics["mean_k_xy"] = float(np.nanmean(k_xy))
        np.fill_diagonal(k_xx, 1.0)
        np.fill_diagonal(k_yy, 1.0)
        
    elif kernel == "linear":
        k_xx = x @ x.T
        k_yy = y @ y.T
        k_xy = x @ y.T
    else:
        raise ValueError(f"Unsupported MMD kernel: {kernel}")

    if unbiased:
        # Unbiased U-statistic estimator (exclude diagonal terms)
        np.fill_diagonal(k_xx, 0.0)
        np.fill_diagonal(k_yy, 0.0)
        term_xx = k_xx.sum() / (n * (n - 1))
        term_yy = k_yy.sum() / (m * (m - 1))
        term_xy = 2.0 * k_xy.mean()
        mmd2_raw = term_xx + term_yy - term_xy
        diagnostics["mmd2_term_xx"] = float(term_xx)
        diagnostics["mmd2_term_yy"] = float(term_yy)
        diagnostics["mmd2_term_xy"] = float(term_xy)
        diagnostics["mmd2_raw"] = float(mmd2_raw)
    else:
        term_xx = k_xx.mean()
        term_yy = k_yy.mean()
        term_xy = 2.0 * k_xy.mean()
        mmd2_raw = term_xx + term_yy - term_xy
        diagnostics["mmd2_term_xx"] = float(term_xx)
        diagnostics["mmd2_term_yy"] = float(term_yy)
        diagnostics["mmd2_term_xy"] = float(term_xy)
        diagnostics["mmd2_raw"] = float(mmd2_raw)

    # Return non-negative MMD^2 (but diagnostics contain raw value)
    # Note: mmd2_raw can be negative due to numerical errors in unbiased estimator
    # when distributions are very similar or sample sizes are small. This is normal.
    # We clamp it to 0 to ensure non-negativity as MMD^2 is theoretically always >= 0.
    mmd2 = float(max(mmd2_raw, 0.0))
    return mmd2, diagnostics


def compute_coral(x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Compute CORAL discrepancy between two feature sets.

    Uses the common CORAL loss:
        ||Cov(x) - Cov(y)||_F^2 / (4 * d^2)
    
    Also returns normalized version: ||Cov(x) - Cov(y)||_F^2 / d
    """
    x = _as_2d_float64(x)
    y = _as_2d_float64(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"CORAL requires same feature dim, got {x.shape[1]} vs {y.shape[1]}")
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    if n < 2 or m < 2:
        return float("nan"), {}

    x_c = x - x.mean(axis=0, keepdims=True)
    y_c = y - y.mean(axis=0, keepdims=True)

    cov_x = (x_c.T @ x_c) / max(1, n - 1)
    cov_y = (y_c.T @ y_c) / max(1, m - 1)

    diff = cov_x - cov_y
    fro_norm_sq = np.linalg.norm(diff, ord="fro") ** 2
    
    diagnostics = {
        "fro_norm_sq": float(fro_norm_sq),
        "fro_norm": float(np.sqrt(fro_norm_sq)),
        "feature_dim": int(d),
    }
    
    # Original normalization (divides by 4*d^2, often too small)
    loss_original = fro_norm_sq / (4.0 * (d ** 2))
    # Alternative normalization (divides by d, more interpretable)
    loss_normalized = fro_norm_sq / float(d)
    
    diagnostics["coral_original"] = float(loss_original)
    diagnostics["coral_normalized"] = float(loss_normalized)
    
    # Return original for backward compatibility, but include normalized in diagnostics
    return float(loss_original), diagnostics


def _subsample_rows(x: np.ndarray, max_n: Optional[int], random_state: int) -> np.ndarray:
    if max_n is None or x.shape[0] <= max_n:
        return x
    rng = np.random.default_rng(random_state)
    idx = rng.choice(x.shape[0], size=max_n, replace=False)
    return x[idx]


def compute_pairwise_discrepancy_metrics(
    features: np.ndarray,
    group_values: np.ndarray,
    group_names: Optional[Dict[Any, str]] = None,
    *,
    compute_mmd_flag: bool = True,
    compute_coral_flag: bool = True,
    mmd_kernel: str = "rbf",
    mmd_gamma: Optional[float] = None,
    metrics_max_per_group: Optional[int] = 1000,
    random_state: int = 42,
    use_median_heuristic: bool = True,
) -> List[Dict[str, Any]]:
    """Compute pairwise MMD/CORAL between groups defined by group_values."""
    features = _as_2d_float64(features)
    group_values = np.asarray(group_values)
    if len(features) != len(group_values):
        raise ValueError(
            f"features and group_values length mismatch: {len(features)} vs {len(group_values)}"
        )

    uniques = list(np.unique(group_values))
    rows: List[Dict[str, Any]] = []
    for a, b in itertools.combinations(uniques, 2):
        x = features[group_values == a]
        y = features[group_values == b]

        x = _subsample_rows(x, metrics_max_per_group, random_state=random_state)
        y = _subsample_rows(y, metrics_max_per_group, random_state=random_state + 1)

        row: Dict[str, Any] = {
            "group_a": str(group_names.get(a, a) if group_names else a),
            "group_b": str(group_names.get(b, b) if group_names else b),
            "n_a": int(x.shape[0]),
            "n_b": int(y.shape[0]),
            "feature_dim": int(features.shape[1]),
        }
        if compute_mmd_flag:
            mmd2, mmd_diag = compute_mmd(
                x, y, 
                kernel=mmd_kernel, 
                gamma=mmd_gamma, 
                unbiased=False,  # Use biased estimator to avoid negative values
                use_median_heuristic=use_median_heuristic if mmd_gamma is None else False,
            )
            row["mmd2"] = mmd2
            # Include diagnostics
            row.update({f"mmd_{k}": v for k, v in mmd_diag.items()})
        if compute_coral_flag:
            coral, coral_diag = compute_coral(x, y)
            row["coral"] = coral
            # Include normalized CORAL and diagnostics
            row["coral_normalized"] = coral_diag.get("coral_normalized", coral)
            row.update({f"coral_{k}": v for k, v in coral_diag.items() if k not in ("coral_normalized",)})
        rows.append(row)
    return rows


def save_metrics(rows: List[Dict[str, Any]], output_path_base: str) -> None:
    """Save metrics rows to both JSON and CSV."""
    json_path = f"{output_path_base}.json"
    csv_path = f"{output_path_base}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Metrics saved to {json_path}")

    if not rows:
        # Still write an empty CSV with header for convenience
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("group_a,group_b,n_a,n_b,feature_dim,mmd2,coral\n")
        print(f"[INFO] Metrics saved to {csv_path}")
        return

    fieldnames: List[str] = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[INFO] Metrics saved to {csv_path}")


def set_mixed_font_title(ax, text: str, fontsize: int = 22, weight: str = 'bold'):
    """Set title with mixed fonts: Chinese (SimSun) and Western (Times New Roman).
    
    This function splits the text into Chinese and non-Chinese parts and renders them
    with appropriate fonts. However, matplotlib doesn't support true mixed fonts in
    a single text object, so we use SimSun for text containing Chinese (which handles
    both Chinese and English reasonably well) and Times New Roman for pure English.
    """
    import re
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
    if has_chinese:
        # If contains Chinese, use SimSun (it handles Chinese well)
        # Note: SimSun will also render English, but may not be exactly Times New Roman
        title_font = FontProperties(family='SimSun', size=fontsize, weight=weight)
        ax.set_title(text, fontproperties=title_font)
    else:
        # If only English/Western characters, use Times New Roman
        title_font = FontProperties(family='Times New Roman', size=fontsize, weight=weight)
        ax.set_title(text, fontproperties=title_font)


def format_bearing_name(bearing_name: str) -> str:
    """Format bearing name for display in legend.
    
    Examples:
        c1_Bearing1_1_labeled -> X1-Bearing1_1
        c2_Bearing2_3_labeled -> X2-Bearing2_3
        c3_Bearing3_2_labeled -> X3-Bearing3_2
    """
    # 提取条件编号（c1, c2, c3等）
    if bearing_name.startswith('c'):
        # 找到下划线的位置
        underscore_idx = bearing_name.find('_')
        if underscore_idx > 0:
            condition = bearing_name[1:underscore_idx]  # 提取数字部分
            # 提取Bearing部分（去掉_labeled后缀）
            bearing_part = bearing_name[underscore_idx + 1:]
            if bearing_part.endswith('_labeled'):
                bearing_part = bearing_part[:-8]  # 去掉_labeled
            return f"X{condition}-{bearing_part}"
    
    # 如果没有匹配到格式，返回原名称（去掉_labeled）
    if bearing_name.endswith('_labeled'):
        return bearing_name[:-8]
    return bearing_name


def create_bearing_labels(bearing_names: List[str], data_lengths: List[int]) -> np.ndarray:
    """Create labels array indicating which bearing each sample belongs to."""
    labels = []
    for i, (name, length) in enumerate(zip(bearing_names, data_lengths)):
        labels.extend([i] * length)
    return np.array(labels)


def create_dataset_labels(bearing_names: List[str], data_lengths: List[int]) -> np.ndarray:
    """Create labels array indicating dataset type (xjtu/femto/ottawa)."""
    labels = []
    for name, length in zip(bearing_names, data_lengths):
        if 'xjtu' in name.lower() or 'bearing1' in name.lower() or 'bearing2' in name.lower() or 'bearing3' in name.lower():
            dataset = 'XJTU'
        elif 'femto' in name.lower():
            dataset = 'FEMTO'
        elif 'ottawa' in name.lower():
            dataset = 'Ottawa'
        else:
            dataset = 'Unknown'
        labels.extend([dataset] * length)
    return np.array(labels)


def visualize_tsne(
    embeddings: np.ndarray,
    color_by: str,
    color_values: np.ndarray,
    labels: Optional[np.ndarray] = None,
    output_path: str = "tsne_visualization.svg",
    title: str = "t-SNE Visualization",
    figsize: Tuple[int, int] = (12, 8),
    alpha: float = 0.6,
    s: float = 20.0,
    bearing_names: Optional[List[str]] = None,
    metrics: Optional[List[Dict[str, Any]]] = None,
):
    """Create t-SNE visualization with different color schemes."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if color_by == "bearing":
        # Color by bearing (discrete) - Use high contrast colors
        unique_bearings = np.unique(color_values)
        n_bearings = len(unique_bearings)
        
        # High contrast color palette for 2 groups: Deep Blue and Bright Orange
        # For more groups, use additional high contrast colors
        high_contrast_colors = [
            '#1f77b4',  # Deep Blue
            '#4169E1',  # Royal Blue (changed from Bright Orange)
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf',  # Cyan
        ]
        
        # Use high contrast colors, cycling if needed
        colors = [high_contrast_colors[i % len(high_contrast_colors)] for i in range(n_bearings)]
        
        for i, bearing_idx in enumerate(unique_bearings):
            mask = color_values == bearing_idx
            # 使用格式化的轴承名称作为图例标签
            if bearing_names is not None and int(bearing_idx) < len(bearing_names):
                label = format_bearing_name(bearing_names[int(bearing_idx)])
            else:
                label = f"Bearing {int(bearing_idx) + 1}"
            # 确保所有点使用完全相同的颜色，创建颜色数组避免渐变
            n_points = np.sum(mask)
            # 第一个轴承（索引为0）使用红色
            if int(bearing_idx) == 0:
                point_colors = ['#d62728'] * n_points  # Red
            else:
                point_colors = [colors[i]] * n_points  # 每个点都使用相同的颜色
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=point_colors,
                label=label,
                alpha=alpha,
                s=s,
                edgecolors='none',  # 移除边缘，避免视觉渐变
            )
        # Legend fontsize will be updated after metrics are drawn
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18, prop={'family': 'Times New Roman'})
        
    elif color_by == "dataset":
        # Color by dataset type (discrete) - Use high contrast colors
        unique_datasets = np.unique(color_values)
        n_datasets = len(unique_datasets)
        
        # High contrast color palette for datasets
        high_contrast_colors = [
            '#1f77b4',  # Deep Blue
            '#ff7f0e',  # Bright Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
        ]
        
        # Use high contrast colors, cycling if needed
        colors = [high_contrast_colors[i % len(high_contrast_colors)] for i in range(n_datasets)]
        
        for i, dataset in enumerate(unique_datasets):
            mask = color_values == dataset
            # 确保所有点使用完全相同的颜色，创建颜色数组避免渐变
            n_points = np.sum(mask)
            point_colors = [colors[i]] * n_points  # 每个点都使用相同的颜色
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=point_colors,
                label=str(dataset),
                alpha=alpha,
                s=s,
                edgecolors='none',  # 移除边缘，避免视觉渐变
            )
        # Legend fontsize will be updated after metrics are drawn
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18, prop={'family': 'Times New Roman'})
        
    elif color_by == "label":
        # Color by RUL label (continuous)
        # Set vmin=0 and vmax=1.0 to ensure colorbar shows the full RUL range (0-1)
        vmin = 0.0
        vmax = 1.0
        scatter = ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=color_values,
            cmap='viridis',
            alpha=alpha,
            s=s,
            vmin=vmin,
            vmax=vmax,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        # Set ticks to include 0.0 and 1.0
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_label('RUL', rotation=270, labelpad=20, fontsize=22, fontfamily='Times New Roman')
        cbar.ax.tick_params(labelsize=18)
        
    else:
        # Default: single color
        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            alpha=alpha,
            s=s,
        )
    
    # ax.set_xlabel("t-SNE Component 1", fontsize=22)
    # ax.set_ylabel("t-SNE Component 2", fontsize=22)
    # ax.set_title(title, fontsize=22, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # Set tick label fontsize to match legend
    ax.tick_params(labelsize=18)
    # Set border linewidth to 1.5
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # Add MMD and CORAL metrics text if available
    if metrics:
        # Get first bearing formatted name for filtering
        first_bearing_formatted = None
        if bearing_names and len(bearing_names) > 0:
            first_bearing_formatted = format_bearing_name(bearing_names[0])
        
        # Build text for bearing pairs that include the first bearing
        metrics_lines = []
        for r in metrics:
            group_a = r.get("group_a", "")
            group_b = r.get("group_b", "")
            mmd2 = r.get("mmd2", None)
            coral_norm = r.get("coral_normalized", None)
            
            # Only include pairs that have the first bearing
            if first_bearing_formatted and first_bearing_formatted not in [group_a, group_b]:
                continue
            
            if mmd2 is not None and np.isfinite(mmd2):
                # Ensure MMD² is non-negative (should already be, but add safety check)
                mmd2_display = max(0.0, mmd2)
                group_a_short = group_a.replace("Bearing", "B")
                group_b_short = group_b.replace("Bearing", "B")
                metrics_lines.append(f"{group_a_short} vs {group_b_short}:\n  MMD²={mmd2_display:.2e}")
        
        if metrics_lines:
            # Combine all pairs into one text block
            metrics_text = "\n".join(metrics_lines)
            # Adjust fontsize based on number of pairs
            fontsize = max(12, 18 - len(metrics_lines))
            ax.text(
                0.02, 0.02, metrics_text,
                transform=ax.transAxes,
                fontsize=fontsize,
                verticalalignment='bottom',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1.5),
                fontfamily='Times New Roman'
            )
            # Update legend fontsize to match metrics fontsize
            legend = ax.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_fontsize(fontsize)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='svg')
    print(f"[INFO] Visualization saved to {output_path}")
    plt.close()


def visualize_tsne_multi(
    tsne_embeddings_list: List[np.ndarray],
    color_by: str,
    color_values: np.ndarray,
    labels: Optional[np.ndarray] = None,
    output_path: str = "tsne_multi_visualization.svg",
    title: str = "t-SNE Visualization",
    figsize: Optional[Tuple[int, int]] = None,
    alpha: float = 0.6,
    s: float = 20.0,
    bearing_names: Optional[List[str]] = None,
    metrics_list: Optional[List[Dict[str, Any]]] = None,
    model_titles: Optional[List[str]] = None,
    metrics_csv_paths: Optional[List[str]] = None,
):
    """Create visualization with multiple subplots, one t-SNE subplot for each model."""
    print(f"[DEBUG] visualize_tsne_multi: color_by={color_by}, color_values range: [{color_values.min():.4f}, {color_values.max():.4f}], shape: {color_values.shape}")
    if color_by == "label":
        print(f"[DEBUG] color_values sample (first 5): {color_values[:5]}")
    n_models = len(tsne_embeddings_list)
    
    # If color_by == "label", add a separate colorbar subplot
    has_separate_colorbar = (color_by == "label")
    
    if has_separate_colorbar:
        # Use GridSpec to create layout: 2 main plots + 1 colorbar subplot
        # First two subplots have equal size, colorbar subplot is narrow
        if figsize is None:
            figsize = (14, 6)  # Wider to accommodate colorbar
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.15], wspace=0.3)
        axes_flat = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    else:
        # Determine layout: try to make it roughly square
        if n_models <= 2:
            n_rows, n_cols = 1, n_models
        elif n_models <= 4:
            n_rows, n_cols = 2, 2
        elif n_models <= 6:
            n_rows, n_cols = 2, 3
        elif n_models <= 9:
            n_rows, n_cols = 3, 3
        else:
            n_cols = int(np.ceil(np.sqrt(n_models)))
            n_rows = int(np.ceil(n_models / n_cols))
        
        # Create figure
        if figsize is None:
            figsize = (6 * n_cols, 6 * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_models == 1:
            axes = np.array([axes])
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
    
    # Store scatter plot objects for colorbar creation
    scatter_objects = []
    
    # Helper function to plot on an axis (same as visualize_tsne)
    def plot_on_axis(ax, embeddings, title_suffix, subplot_idx=None, is_last_subplot=False):
        if color_by == "bearing":
            unique_bearings = np.unique(color_values)
            n_bearings = len(unique_bearings)
            
            high_contrast_colors = [
                '#1f77b4',  # Deep Blue
                # '#ff7f0e',  # Bright Orange
                '#2ca02c',  # Green
                # '#d62728',  # Red
                # '#9467bd',  # Purple
                # '#8c564b',  # Brown
                # '#e377c2',  # Pink
                # '#7f7f7f',  # Gray
                # '#bcbd22',  # Olive
                '#17becf',  # Cyan
            ]
            
            colors = [high_contrast_colors[i % len(high_contrast_colors)] for i in range(n_bearings)]
            
            for i, bearing_idx in enumerate(unique_bearings):
                mask = color_values == bearing_idx
                if bearing_names is not None and int(bearing_idx) < len(bearing_names):
                    label = format_bearing_name(bearing_names[int(bearing_idx)])
                else:
                    label = f"Bearing {int(bearing_idx) + 1}"
                n_points = np.sum(mask)
                # 第一个轴承（索引为0）使用红色
                if int(bearing_idx) == 0:
                    point_colors = ['#d62728'] * n_points  # Red
                else:
                    point_colors = [colors[i]] * n_points
                ax.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=point_colors,
                    label=label,
                    alpha=alpha,
                    s=s,
                    edgecolors='none',
                )
            # Legend fontsize will be updated after metrics are drawn
            # For subplot index 1 (second subplot), place legend at upper right
            if subplot_idx == 1:
                ax.legend(loc='upper right', fontsize=18, prop={'family': 'Times New Roman'})
            else:
                ax.legend(loc='best', fontsize=18, prop={'family': 'Times New Roman'})
            
        elif color_by == "dataset":
            unique_datasets = np.unique(color_values)
            n_datasets = len(unique_datasets)
            
            high_contrast_colors = [
                '#1f77b4',  # Deep Blue
                '#ff7f0e',  # Bright Orange
                '#2ca02c',  # Green
                '#d62728',  # Red
                '#9467bd',  # Purple
            ]
            
            colors = [high_contrast_colors[i % len(high_contrast_colors)] for i in range(n_datasets)]
            
            for i, dataset in enumerate(unique_datasets):
                mask = color_values == dataset
                n_points = np.sum(mask)
                point_colors = [colors[i]] * n_points
                ax.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=point_colors,
                    label=str(dataset),
                    alpha=alpha,
                    s=s,
                    edgecolors='none',
                )
            # Legend fontsize will be updated after metrics are drawn
            ax.legend(loc='best', fontsize=18, prop={'family': 'Times New Roman'})
            
        elif color_by == "label":
            # Set vmin=0 and vmax=1.0 to ensure colorbar shows the full RUL range (0-1)
            # This ensures consistent color mapping across all subplots
            vmin = 0.0
            vmax = 1.0
            # Create norm explicitly to ensure scatter plot uses correct range
            norm = Normalize(vmin=vmin, vmax=vmax)
            # Use reversed colormap for all subplots: dark color for RUL=1.0, light color for RUL=0.0
            cmap_to_use = 'viridis_r'  # Reversed viridis: dark for high values, light for low values
            scatter = ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                c=color_values,
                cmap=cmap_to_use,
                alpha=alpha,
                s=s,
                norm=norm,  # Use explicit norm instead of vmin/vmax
            )
            # Store scatter object for colorbar creation
            scatter_objects.append(scatter)
            # Don't show colorbar here - it will be in a separate subplot
        else:
            ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                alpha=alpha,
                s=s,
            )
        
        ax.grid(True, alpha=0.3)
        # Set title with mixed fonts: Chinese (SimSun) and Western (Times New Roman)
        set_mixed_font_title(ax, title_suffix, fontsize=22, weight='bold')
        ax.tick_params(labelsize=18)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    
    # Plot for each model
    for i in range(n_models):
        # Set title with model name and index
        if model_titles and i < len(model_titles):
            title_text = f"({chr(97 + i)}) {model_titles[i]}"
        else:
            title_text = f"({chr(97 + i)}) t-SNE"
        is_last = (i == n_models - 1)
        plot_on_axis(axes_flat[i], tsne_embeddings_list[i], title_text, subplot_idx=i, is_last_subplot=is_last)
        # Add MMD and CORAL metrics text if available
        if metrics_list is not None and i < len(metrics_list):
            metrics = metrics_list[i]
            if metrics:
                # Get first bearing formatted name for filtering
                first_bearing_formatted = None
                if bearing_names and len(bearing_names) > 0:
                    first_bearing_formatted = format_bearing_name(bearing_names[0])
                
                # Build text for bearing pairs that include the first bearing
                metrics_lines = []
                for r in metrics:
                    group_a = r.get("group_a", "")
                    group_b = r.get("group_b", "")
                    mmd2 = r.get("mmd2", None)
                    coral_norm = r.get("coral_normalized", None)
                    
                    # Only include pairs that have the first bearing
                    if first_bearing_formatted and first_bearing_formatted not in [group_a, group_b]:
                        continue
                    
                    if mmd2 is not None and np.isfinite(mmd2):
                        # Ensure MMD² is non-negative (should already be, but add safety check)
                        mmd2_display = max(0.0, mmd2)
                        group_a_short = group_a.replace("Bearing", "B")
                        group_b_short = group_b.replace("Bearing", "B")
                        metrics_lines.append(f"{group_a_short} vs {group_b_short}:\n  MMD²={mmd2_display:.2e}")
                
                # # Read RMSE and CWC from CSV if available
                # rmse_text = ""
                # cwc_text = ""
                # if metrics_csv_paths and i < len(metrics_csv_paths):
                #     csv_path = metrics_csv_paths[i]
                #     if os.path.exists(csv_path):
                #         try:
                #             df = pd.read_csv(csv_path)
                #             if 'RMSE' in df.columns and 'CWC' in df.columns:
                #                 rmse = df['RMSE'].iloc[0]
                #                 cwc = df['CWC'].iloc[0]
                #                 rmse_text = f"RMSE={rmse:.4f}"
                #                 cwc_text = f"CWC={cwc:.4f}"
                #         except Exception as e:
                #             print(f"[WARN] Failed to read metrics from {csv_path}: {e}")
                
                if metrics_lines:
                    # Combine all pairs into one text block
                    metrics_text = "\n".join(metrics_lines)
                    # Add RMSE and CWC if available
                    # if rmse_text and cwc_text:
                        # metrics_text = f"{metrics_text}\n{rmse_text}, {cwc_text}"
                    # Adjust fontsize based on number of pairs
                    fontsize = max(12, 18 - len(metrics_lines))
                    # For subplot index 1 (second subplot), place metrics at bottom center
                    if i == 1:
                        axes_flat[i].text(
                            0.5, 0.02, metrics_text,
                            transform=axes_flat[i].transAxes,
                            fontsize=fontsize,
                            verticalalignment='bottom',
                            horizontalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1.5),
                            fontfamily='Times New Roman'
                        )
                    else:
                        axes_flat[i].text(
                            0.02, 0.02, metrics_text,
                            transform=axes_flat[i].transAxes,
                            fontsize=fontsize,
                            verticalalignment='bottom',
                            horizontalalignment='left',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1.5),
                            fontfamily='Times New Roman'
                        )
                    # Update legend fontsize to match metrics fontsize
                    legend = axes_flat[i].get_legend()
                    if legend is not None:
                        for text in legend.get_texts():
                            text.set_fontsize(fontsize)
    
    # Handle colorbar subplot if needed
    if has_separate_colorbar:
        # Create colorbar in the third subplot
        cbar_ax = axes_flat[2]
        # Explicitly set the colorbar range based on color_values (RUL labels)
        # Force vmin=0 and vmax=1.0 to ensure colorbar shows the full RUL range (0-1)
        vmin = 0.0
        vmax = 1.0
        print(f"[DEBUG] Creating colorbar with vmin={vmin}, vmax={vmax}, color_values range: [{np.min(color_values):.4f}, {np.max(color_values):.4f}]")
        # Always create a new ScalarMappable with explicit norm to ensure correct range
        norm = Normalize(vmin=vmin, vmax=vmax)
        # Use reversed colormap to match all subplots: dark color for RUL=1.0, light color for RUL=0.0
        cmap_for_colorbar = 'viridis_r'  # Reversed viridis to match all subplots
        sm = ScalarMappable(cmap=cmap_for_colorbar, norm=norm)
        sm.set_array([])  # Empty array, we only need the colormap
        # Create colorbar using the ScalarMappable with explicit norm
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
        # Ensure colorbar uses the correct norm (explicitly set it again)
        cbar.mappable.set_norm(norm)
        # Set ticks to include 0.0 and 1.0
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # Invert y-axis so that 1.0 is at bottom and 0.0 is at top (color gradient remains the same)
        cbar.ax.invert_yaxis()
        print(f"[DEBUG] Colorbar created, norm vmin={cbar.norm.vmin}, vmax={cbar.norm.vmax}, mappable norm vmin={cbar.mappable.norm.vmin}, vmax={cbar.mappable.norm.vmax}")
        cbar.set_label('RUL', rotation=270, labelpad=20, fontsize=22, fontfamily='Times New Roman')
        cbar.ax.tick_params(labelsize=18)
    else:
        # Hide unused subplots
        for i in range(n_models, len(axes_flat)):
            axes_flat[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='svg')
    print(f"[INFO] Multi-model t-SNE visualization saved to {output_path}")
    plt.close()


def visualize_tsne_umap_multi(
    tsne_embeddings_list: List[np.ndarray],
    umap_embeddings_list: List[np.ndarray],
    color_by: str,
    color_values: np.ndarray,
    labels: Optional[np.ndarray] = None,
    output_path: str = "tsne_umap_visualization.svg",
    title: str = "t-SNE and UMAP Visualization",
    figsize: Optional[Tuple[int, int]] = None,
    alpha: float = 0.6,
    s: float = 20.0,
    bearing_names: Optional[List[str]] = None,
):
    """Create visualization with multiple subplots, one for each model's t-SNE and UMAP."""
    n_models = len(tsne_embeddings_list)
    if len(umap_embeddings_list) != n_models:
        raise ValueError(f"tsne_embeddings_list and umap_embeddings_list must have same length")
    
    # Create figure with n_models rows and 2 columns (t-SNE and UMAP)
    if figsize is None:
        figsize = (16, 4 * n_models)
    fig, axes = plt.subplots(n_models, 2, figsize=figsize)
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    # Helper function to plot on an axis
    def plot_on_axis(ax, embeddings, title_suffix, is_last_subplot=False):
        if color_by == "bearing":
            unique_bearings = np.unique(color_values)
            n_bearings = len(unique_bearings)
            
            high_contrast_colors = [
                '#1f77b4',  # Deep Blue
                '#ff7f0e',  # Bright Orange
                '#2ca02c',  # Green
                '#d62728',  # Red
                '#9467bd',  # Purple
                '#8c564b',  # Brown
                '#e377c2',  # Pink
                '#7f7f7f',  # Gray
                '#bcbd22',  # Olive
                '#17becf',  # Cyan
            ]
            
            colors = [high_contrast_colors[i % len(high_contrast_colors)] for i in range(n_bearings)]
            
            for i, bearing_idx in enumerate(unique_bearings):
                mask = color_values == bearing_idx
                if bearing_names is not None and int(bearing_idx) < len(bearing_names):
                    label = format_bearing_name(bearing_names[int(bearing_idx)])
                else:
                    label = f"Bearing {int(bearing_idx) + 1}"
                n_points = np.sum(mask)
                # 第一个轴承（索引为0）使用红色
                if int(bearing_idx) == 0:
                    point_colors = ['#d62728'] * n_points  # Red
                else:
                    point_colors = [colors[i]] * n_points
                ax.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=point_colors,
                    label=label,
                    alpha=alpha,
                    s=s,
                    edgecolors='none',
                )
            # Legend fontsize will be updated after metrics are drawn
            ax.legend(loc='best', fontsize=18, prop={'family': 'Times New Roman'})
            
        elif color_by == "dataset":
            unique_datasets = np.unique(color_values)
            n_datasets = len(unique_datasets)
            
            high_contrast_colors = [
                '#1f77b4',  # Deep Blue
                '#ff7f0e',  # Bright Orange
                '#2ca02c',  # Green
                '#d62728',  # Red
                '#9467bd',  # Purple
            ]
            
            colors = [high_contrast_colors[i % len(high_contrast_colors)] for i in range(n_datasets)]
            
            for i, dataset in enumerate(unique_datasets):
                mask = color_values == dataset
                n_points = np.sum(mask)
                point_colors = [colors[i]] * n_points
                ax.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=point_colors,
                    label=str(dataset),
                    alpha=alpha,
                    s=s,
                    edgecolors='none',
                )
            # Legend fontsize will be updated after metrics are drawn
            ax.legend(loc='best', fontsize=18, prop={'family': 'Times New Roman'})
            
        elif color_by == "label":
            # Set vmin and vmax explicitly to ensure colorbar shows correct RUL range (0-1)
            vmin = float(np.min(color_values))
            vmax = float(np.max(color_values))
            scatter = ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                c=color_values,
                cmap='viridis',
                alpha=alpha,
                s=s,
                vmin=vmin,
                vmax=vmax,
            )
            # Only show colorbar on the last subplot
            if is_last_subplot:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('RUL', rotation=270, labelpad=20, fontsize=22, fontfamily='Times New Roman')
                cbar.ax.tick_params(labelsize=18)
        else:
            ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                alpha=alpha,
                s=s,
            )
        
        ax.grid(True, alpha=0.3)
        # Set title with mixed fonts: Chinese (SimSun) and Western (Times New Roman)
        set_mixed_font_title(ax, title_suffix, fontsize=22, weight='bold')
        # Set tick label fontsize to match legend
        ax.tick_params(labelsize=18)
        # Set border linewidth to 1.5
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    
    # Plot for each model
    for i in range(n_models):
        # Plot t-SNE (subplot index: i*2)
        plot_on_axis(axes[i, 0], tsne_embeddings_list[i], f"({chr(97 + i*2)}) t-SNE", is_last_subplot=False)
        # Plot UMAP (subplot index: i*2 + 1) - only last UMAP shows colorbar
        is_last_umap = (i == n_models - 1)
        plot_on_axis(axes[i, 1], umap_embeddings_list[i], f"({chr(98 + i*2)}) UMAP", is_last_subplot=is_last_umap)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='svg')
    print(f"[INFO] Combined visualization saved to {output_path}")
    plt.close()


def visualize_tsne_umap(
    tsne_embeddings: np.ndarray,
    umap_embeddings: np.ndarray,
    color_by: str,
    color_values: np.ndarray,
    labels: Optional[np.ndarray] = None,
    output_path: str = "tsne_umap_visualization.svg",
    title: str = "t-SNE and UMAP Visualization",
    figsize: Tuple[int, int] = (20, 8),
    alpha: float = 0.6,
    s: float = 20.0,
    bearing_names: Optional[List[str]] = None,
):
    """Create side-by-side visualization of t-SNE and UMAP with different color schemes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Helper function to plot on an axis
    def plot_on_axis(ax, embeddings, title_suffix, is_last_subplot=False):
        if color_by == "bearing":
            unique_bearings = np.unique(color_values)
            n_bearings = len(unique_bearings)
            
            high_contrast_colors = [
                '#1f77b4',  # Deep Blue
                '#ff7f0e',  # Bright Orange
                '#2ca02c',  # Green
                '#d62728',  # Red
                '#9467bd',  # Purple
                '#8c564b',  # Brown
                '#e377c2',  # Pink
                '#7f7f7f',  # Gray
                '#bcbd22',  # Olive
                '#17becf',  # Cyan
            ]
            
            colors = [high_contrast_colors[i % len(high_contrast_colors)] for i in range(n_bearings)]
            
            for i, bearing_idx in enumerate(unique_bearings):
                mask = color_values == bearing_idx
                if bearing_names is not None and int(bearing_idx) < len(bearing_names):
                    label = format_bearing_name(bearing_names[int(bearing_idx)])
                else:
                    label = f"Bearing {int(bearing_idx) + 1}"
                n_points = np.sum(mask)
                # 第一个轴承（索引为0）使用红色
                if int(bearing_idx) == 0:
                    point_colors = ['#d62728'] * n_points  # Red
                else:
                    point_colors = [colors[i]] * n_points
                ax.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=point_colors,
                    label=label,
                    alpha=alpha,
                    s=s,
                    edgecolors='none',
                )
            # Legend fontsize will be updated after metrics are drawn
            ax.legend(loc='best', fontsize=18, prop={'family': 'Times New Roman'})
            
        elif color_by == "dataset":
            unique_datasets = np.unique(color_values)
            n_datasets = len(unique_datasets)
            
            high_contrast_colors = [
                '#1f77b4',  # Deep Blue
                '#ff7f0e',  # Bright Orange
                '#2ca02c',  # Green
                '#d62728',  # Red
                '#9467bd',  # Purple
            ]
            
            colors = [high_contrast_colors[i % len(high_contrast_colors)] for i in range(n_datasets)]
            
            for i, dataset in enumerate(unique_datasets):
                mask = color_values == dataset
                n_points = np.sum(mask)
                point_colors = [colors[i]] * n_points
                ax.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=point_colors,
                    label=str(dataset),
                    alpha=alpha,
                    s=s,
                    edgecolors='none',
                )
            # Legend fontsize will be updated after metrics are drawn
            ax.legend(loc='best', fontsize=18, prop={'family': 'Times New Roman'})
            
        elif color_by == "label":
            # Set vmin and vmax explicitly to ensure colorbar shows correct RUL range (0-1)
            vmin = float(np.min(color_values))
            vmax = float(np.max(color_values))
            scatter = ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                c=color_values,
                cmap='viridis',
                alpha=alpha,
                s=s,
                vmin=vmin,
                vmax=vmax,
            )
            # Only show colorbar on the last subplot
            if is_last_subplot:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('RUL', rotation=270, labelpad=20, fontsize=22, fontfamily='Times New Roman')
                cbar.ax.tick_params(labelsize=18)
        else:
            ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                alpha=alpha,
                s=s,
            )
        
        ax.grid(True, alpha=0.3)
        # Set title with mixed fonts: Chinese (SimSun) and Western (Times New Roman)
        set_mixed_font_title(ax, title_suffix, fontsize=22, weight='bold')
        # Set tick label fontsize to match legend
        ax.tick_params(labelsize=18)
        # Set border linewidth to 1.5
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    
    # Plot t-SNE
    plot_on_axis(ax1, tsne_embeddings, "(a) t-SNE", is_last_subplot=False)
    # ax1.set_xlabel("t-SNE Component 1", fontsize=18)
    # ax1.set_ylabel("t-SNE Component 2", fontsize=18)
    
    # Plot UMAP (last subplot, show colorbar)
    plot_on_axis(ax2, umap_embeddings, "(b) UMAP", is_last_subplot=True)
    # ax2.set_xlabel("UMAP Component 1", fontsize=18)
    # ax2.set_ylabel("UMAP Component 2", fontsize=18)
    
    # Main title
    # fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='svg')
    print(f"[INFO] Combined visualization saved to {output_path}")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE visualization for bearing RUL data.")
    
    parser.add_argument(
        "--mode",
        choices=["raw", "tcn", "attention"],
        nargs="+",
        default=["raw","attention"],
        # default="attention",
        help="Feature extraction mode: raw (flattened input), tcn (after TCN layers), attention (after attention layer). Can specify multiple modes for multiple subplots.",
    )
    
    parser.add_argument(
        "--config",
        nargs="+",
        # default=["../../config/ablation/G_fbtcn_config_ablation_no_fkl.json", "../../config/ablation/A_fbtcn_config_ablation_no_rds_sa_revise.json"],
        default=["../../config/ablation/A_fbtcn_config_ablation_no_rds_sa_revise.json"],
        help="Config JSON (required for tcn/attention modes). Can specify multiple configs.",
    )
    
    parser.add_argument(
        "--model-path",
        nargs="+",
        # default=["/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/G_no_fkl/xjtu_to_xjtu/Bearing3_best_model_fbtcn.pt", "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds_sa_revise/xjtu_to_xjtu/Bearing3_best_model_fbtcn.pt"],
        default=["/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds_sa_revise/xjtu_to_xjtu/Bearing1_best_model_fbtcn.pt"],
        # default=["/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/G_no_fkl/xjtu_to_femto/Bearing_best_model_fbtcn.pt", "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds_sa_revise/xjtu_to_femto/Bearing_best_model_fbtcn.pt"],
        help="Trained model checkpoint (required for tcn/attention modes). Can specify multiple model paths.",
    )
    
    parser.add_argument(
        "--dataset-type",
        default="xjtu_made",
        # default="femto_made",
        help="Dataset folder name under datasetresult/ (e.g., xjtu_made, femto_made).",
    )
    
    parser.add_argument(
        "--bearing-names",
        nargs="+",
        # default=["c3_Bearing3_5_labeled", "c2_Bearing2_2_labeled", "c1_Bearing1_2_labeled"],
        default=["c1_Bearing1_3_labeled"],
        # default=["Bearing1_7_labeled", "Bearing1_2_labeled", "Bearing1_5_labeled"],
        help="Bearing names to visualize (substrings that appear in filenames).",
    )
    
    parser.add_argument(
        "--color-by",
        choices=["bearing", "dataset", "label", "none"],
        default="label",
        help="Color scheme for visualization.",
    )
    
    parser.add_argument(
        "--output-dir",
        default="./tsne_outputs",
        help="Directory to save visualization plots.",
    )
    
    parser.add_argument(
        "--metrics-csv-paths",
        nargs="+",
        default=[
            "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/G_no_fkl/xjtu_to_femto/Bearing1_7_calibrated_metrics.csv",
            "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir/xjtu_to_femto/Bearing1_7_calibrated_metrics.csv"
        ],
        help="Paths to CSV files containing RMSE and CWC metrics. Should have same length as model-path.",
    )
    
    parser.add_argument(
        "--model-titles",
        nargs="+",
        # default=["PSVI","FSVI"],
        default=["输入特征","SA层输出特征"],
        help="Titles for each model subplot. Should have same length as model-path.",
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Maximum number of samples per bearing (None = use all).",
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum total samples for t-SNE computation (for speed).",
    )
    
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity parameter.",
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for t-SNE and UMAP reproducibility.",
    )
    
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter (number of neighbors for UMAP).",
    )
    
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (minimum distance between points in embedding space).",
    )
    
    parser.add_argument(
        "--use-umap",
        action="store_true",
        default=False,
        help="Enable UMAP visualization (t-SNE only by default).",
    )
    
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable discrepancy metrics (MMD/CORAL) computation.",
    )
    parser.add_argument(
        "--metrics-group-by",
        choices=["bearing", "dataset"],
        default=None,
        help="Which grouping to use for MMD/CORAL. Default: follow --color-by if it is bearing/dataset, else bearing.",
    )
    parser.add_argument(
        "--metrics-max-per-group",
        type=int,
        default=1000,
        help="Max samples per group used to compute MMD/CORAL (to control O(n^2) cost). Use 0 to disable subsampling.",
    )
    parser.add_argument(
        "--mmd-kernel",
        choices=["rbf", "linear"],
        default="rbf",
        help="Kernel used for MMD computation.",
    )
    parser.add_argument(
        "--mmd-gamma",
        type=float,
        default=None,
        help="Gamma for RBF MMD kernel. Default: 1 / feature_dim.",
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
    
    print(f"[INFO] Using device: {device}")
    # Ensure mode is a list
    mode_list = args.mode if isinstance(args.mode, list) else [args.mode]
    print(f"[INFO] Modes: {mode_list}")
    print(f"[INFO] Bearings: {args.bearing_names}")
    
    # Load data
    data_dir = get_data_dir(args.dataset_type)
    all_data = []
    all_labels = []
    data_lengths = []
    
    # For label mode, we need to reverse normalize labels using scaler
    # Scaler files are stored in the same directory as data files
    scaler_dir = data_dir
    
    for bearing_name in args.bearing_names:
        data, labels = load_bearing_data([bearing_name], data_dir)
        if len(data) == 0:
            print(f"[WARN] No data found for {bearing_name}, skipping...")
            continue
        
        data = data.to(torch.float32)
        labels = labels.to(torch.float32)
        
        # If color_by == "label", reverse normalize labels to get original RUL (0-1 range)
        if args.color_by == "label":
            # Try to find scaler file for this bearing
            # Scaler file naming: {bearing_name}_labeled_fpt_scaler or {bearing_name}_labeled_scaler
            scaler_found = False
            for scaler_suffix in ["_labeled_fpt_scaler", "_labeled_scaler"]:
                scaler_filename = bearing_name.replace("_labeled", "") + scaler_suffix
                scaler_path = os.path.join(scaler_dir, scaler_filename)
                if os.path.exists(scaler_path):
                    try:
                        scaler = joblib_load(scaler_path)
                        # Reverse normalize labels: labels are (n, 1) shape, need to reshape for inverse_transform
                        labels_np = labels.cpu().numpy()
                        if labels_np.ndim == 1:
                            labels_np = labels_np.reshape(-1, 1)
                        labels_denorm = scaler.inverse_transform(labels_np)
                        labels = torch.tensor(labels_denorm.flatten(), dtype=torch.float32)
                        print(f"[INFO] Reverse normalized labels for {bearing_name} using {scaler_filename}")
                        print(f"[DEBUG] Labels range after denorm: [{labels.min():.4f}, {labels.max():.4f}]")
                        scaler_found = True
                        break
                    except Exception as e:
                        print(f"[WARN] Failed to reverse normalize labels for {bearing_name} using {scaler_filename}: {e}")
                        continue
            
            if not scaler_found:
                print(f"[WARN] No scaler file found for {bearing_name}, labels may not be in 0-1 range")
        
        if args.sample_size is not None:
            sample_size = min(args.sample_size, len(data))
            data = data[:sample_size]
            labels = labels[:sample_size]
        
        all_data.append(data)
        all_labels.append(labels)
        data_lengths.append(len(data))
        print(f"[INFO] Loaded {len(data)} samples from {bearing_name}")
    
    if len(all_data) == 0:
        raise ValueError("No data loaded. Check bearing names and dataset type.")
    
    # Concatenate all data
    data_all = torch.cat(all_data, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    print(f"[INFO] Total samples: {len(data_all)}")
    print(f"[DEBUG] labels_all range: [{labels_all.min():.4f}, {labels_all.max():.4f}], shape: {labels_all.shape}")
    
    # Ensure config and model_path are lists
    config_list = args.config if isinstance(args.config, list) else [args.config]
    model_path_list = args.model_path if isinstance(args.model_path, list) else [args.model_path]
    
    # Ensure they have the same length
    if len(config_list) != len(model_path_list):
        raise ValueError(f"config and model-path must have the same length: {len(config_list)} vs {len(model_path_list)}")
    
    # Extract features for each mode
    n_modes = len(mode_list)
    print(f"[INFO] Processing {n_modes} mode(s)")
    
    all_features_list = []
    for i, mode in enumerate(mode_list):
        print(f"\n[INFO] Processing mode {i+1}/{n_modes}: {mode}")
        
        if mode == "raw":
            features = extract_features_raw(data_all)
        else:
            # For tcn/attention modes, need config and model
            # Use first config/model if multiple modes but single config/model
            config_idx = min(i, len(config_list) - 1)
            model_idx = min(i, len(model_path_list) - 1)
            config_path = config_list[config_idx]
            model_path = model_path_list[model_idx]
            
            print(f"[INFO] Using config: {config_path}")
            print(f"[INFO] Using model: {model_path}")
            
            # Load model
            cfg = load_config(config_path)
            model = build_model(cfg, model_path, device)
            
            if mode == "tcn":
                features = extract_features_tcn(model, data_all, device)
            elif mode == "attention":
                features = extract_features_attention(model, data_all, device)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        
        print(f"[INFO] Features shape: {features.shape}")
        all_features_list.append(features)
    
    # Prepare color values
    os.makedirs(args.output_dir, exist_ok=True)

    # Always build bearing/dataset group labels from original lengths (before sampling)
    bearing_group_values_full = create_bearing_labels(args.bearing_names, data_lengths)
    dataset_group_values_full = create_dataset_labels(args.bearing_names, data_lengths)
    
    # Create color values based on original data length (before sampling)
    if args.color_by == "bearing":
        color_values_full = bearing_group_values_full
        title_suffix = "by Bearing"
    elif args.color_by == "dataset":
        color_values_full = dataset_group_values_full
        title_suffix = "by Dataset"
    elif args.color_by == "label":
        color_values_full = labels_all.cpu().numpy()
        print(f"[DEBUG] Setting color_values_full from labels_all: range [{color_values_full.min():.4f}, {color_values_full.max():.4f}]")
        title_suffix = "by RUL Label"
    else:
        color_values_full = np.zeros(len(data_all))
        title_suffix = ""
    
    # Apply t-SNE and UMAP for each mode
    all_tsne_embeddings = []
    all_umap_embeddings = []
    tsne_sample_indices = None
    
    for i, features in enumerate(all_features_list):
        mode = mode_list[i]
        print(f"\n[INFO] Processing embeddings for mode {i+1}/{n_modes}: {mode}")
        
        # Apply t-SNE
        print("\n" + "="*50)
        print(f"Applying t-SNE for mode {i+1} ({mode})...")
        print("="*50)
        tsne_emb, sample_idx = apply_tsne(
            features,
            perplexity=args.perplexity,
            random_state=args.random_state,
            max_samples=args.max_samples,
        )
        all_tsne_embeddings.append(tsne_emb)
        # Use the same sampling indices for all models (from first model)
        if tsne_sample_indices is None:
            tsne_sample_indices = sample_idx
        
        # Apply UMAP
        use_umap = args.use_umap
        if use_umap:
            print("\n" + "="*50)
            print(f"Applying UMAP for mode {i+1} ({mode})...")
            print("="*50)
            try:
                if tsne_sample_indices is not None:
                    umap_features = features[tsne_sample_indices]
                else:
                    umap_features = features
                umap_emb, _ = apply_umap(
                    umap_features,
                    n_neighbors=args.n_neighbors,
                    min_dist=args.min_dist,
                    random_state=args.random_state,
                    max_samples=None,
                )
                all_umap_embeddings.append(umap_emb)
            except ImportError:
                print("[WARN] UMAP not available. Skipping UMAP visualization.")
                all_umap_embeddings = []
                use_umap = False
        else:
            all_umap_embeddings = []
    
    # Handle sampling indices for color_values (use same indices for all modes)
    if tsne_sample_indices is not None:
        # Update labels_all first if needed
        if args.color_by == "label":
            labels_all = labels_all[tsne_sample_indices]
            # Ensure color_values_full matches labels_all after sampling
            color_values_full = labels_all.cpu().numpy()
            print(f"[DEBUG] After sampling: labels_all range: [{labels_all.min():.4f}, {labels_all.max():.4f}], color_values_full range: [{color_values_full.min():.4f}, {color_values_full.max():.4f}]")
        else:
            color_values_full = color_values_full[tsne_sample_indices]
        bearing_group_values_full = bearing_group_values_full[tsne_sample_indices]
        dataset_group_values_full = dataset_group_values_full[tsne_sample_indices]

    # Final color_values used for plotting
    color_values = color_values_full
    print(f"[DEBUG] Final color_values range: [{color_values.min():.4f}, {color_values.max():.4f}], shape: {color_values.shape}, color_by: {args.color_by}")

    # Compute discrepancy metrics (MMD / CORAL) for each mode
    all_metrics_list = []
    if not args.no_metrics:
        metrics_group_by = args.metrics_group_by
        if metrics_group_by is None:
            metrics_group_by = args.color_by if args.color_by in ("bearing", "dataset") else "bearing"

        if metrics_group_by == "bearing":
            metrics_group_values = bearing_group_values_full
            # Map bearing index -> formatted name
            bearing_name_map: Dict[Any, str] = {
                i: format_bearing_name(name) for i, name in enumerate(args.bearing_names)
            }
            group_name_map = bearing_name_map
        else:
            metrics_group_values = dataset_group_values_full
            group_name_map = None

        metrics_max = args.metrics_max_per_group
        metrics_max_per_group = None if metrics_max == 0 else metrics_max

        # Compute metrics for each mode
        for i, features in enumerate(all_features_list):
            mode = mode_list[i]
            try:
                print(f"\n[INFO] Computing metrics for mode {i+1}/{n_modes} ({mode})...")
                features_for_metrics = features[tsne_sample_indices] if tsne_sample_indices is not None else features
                metric_rows = compute_pairwise_discrepancy_metrics(
                    features=features_for_metrics,
                    group_values=metrics_group_values,
                    group_names=group_name_map,
                    compute_mmd_flag=True,  # Enable MMD computation with biased estimator
                    compute_coral_flag=True,
                    mmd_kernel=args.mmd_kernel,
                    mmd_gamma=args.mmd_gamma,
                    metrics_max_per_group=metrics_max_per_group,
                    random_state=args.random_state,
                    use_median_heuristic=(args.mmd_gamma is None),  # Use median heuristic if gamma not specified
                )
                all_metrics_list.append(metric_rows)

                # Save metrics for each mode
                metrics_base = os.path.join(
                    args.output_dir,
                    f"discrepancy_mode{i+1}_{mode}_{metrics_group_by}_{args.random_state}",
                )
                save_metrics(metric_rows, metrics_base)

                # Print a compact summary for this mode
                if metric_rows:
                    rows_sorted = sorted(
                        metric_rows,
                        key=lambda r: (r.get("mmd2", float("nan")) if r.get("mmd2") is not None else float("nan")),
                        reverse=True,
                    )
                    print("\n" + "=" * 50)
                    print(f"Mode {i+1} ({mode}) - Discrepancy metrics summary (group_by={metrics_group_by})")
                    print("=" * 50)
                    print("Note: MMD^2 < 0 indicates numerical instability (often due to imbalanced sample sizes).")
                    print("      Use CORAL_normalized for more reliable comparison.")
                    print("=" * 50)
                    for r in rows_sorted[:10]:
                        mmd2 = r.get("mmd2", None)
                        # Try both possible field names for mmd2_raw
                        mmd2_raw = r.get("mmd_mmd2_raw", r.get("mmd2_raw", None))
                        coral = r.get("coral", None)
                        coral_norm = r.get("coral_normalized", None)
                        gamma = r.get("mmd_gamma", r.get("gamma", None))
                        
                        mmd2_str = f"{mmd2:.6g}" if isinstance(mmd2, (float, int)) and np.isfinite(mmd2) else "nan"
                        mmd2_raw_str = f"{mmd2_raw:.6g}" if isinstance(mmd2_raw, (float, int)) and np.isfinite(mmd2_raw) else "N/A"
                        coral_str = f"{coral:.6g}" if isinstance(coral, (float, int)) and np.isfinite(coral) else "nan"
                        coral_norm_str = f"{coral_norm:.6g}" if isinstance(coral_norm, (float, int)) and np.isfinite(coral_norm) else "N/A"
                        gamma_str = f"{gamma:.6g}" if isinstance(gamma, (float, int)) and np.isfinite(gamma) else "N/A"
                        
                        # Add warning if MMD^2 was truncated from negative value
                        warning = ""
                        if mmd2_raw is not None and mmd2_raw < 0 and mmd2 == 0:
                            warning = " [⚠️ truncated from negative]"
                        
                        print(
                            f"- {r['group_a']} vs {r['group_b']}: "
                            f"MMD^2={mmd2_str} (raw={mmd2_raw_str}, γ={gamma_str}){warning}, "
                            f"CORAL={coral_str} (norm={coral_norm_str}) "
                            f"(n={r['n_a']}/{r['n_b']}, d={r['feature_dim']})"
                        )
            except Exception as exc:
                print(f"[WARN] Failed to compute discrepancy metrics for mode {i+1} ({mode}): {exc}")
                all_metrics_list.append([])
    
    # Create visualizations
    # Extract first bearing name for filename
    first_bearing_name = args.bearing_names[0] if args.bearing_names else "unknown"
    # Clean bearing name for filename (remove special characters)
    first_bearing_clean = first_bearing_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
    
    use_umap = args.use_umap and len(all_umap_embeddings) > 0
    
    if use_umap:
        # Combined t-SNE and UMAP visualization with multiple modes
        modes_str = "_".join(mode_list)
        output_path = os.path.join(
            args.output_dir,
            f"tsne_umap_{first_bearing_clean}_{modes_str}_{args.color_by}_{args.random_state}.svg"
        )
        
        title = f"t-SNE and UMAP Visualization {title_suffix}"
        
        visualize_tsne_umap_multi(
            tsne_embeddings_list=all_tsne_embeddings,
            umap_embeddings_list=all_umap_embeddings,
            color_by=args.color_by,
            color_values=color_values,
            labels=labels_all.cpu().numpy() if args.color_by != "label" else None,
            output_path=output_path,
            title=title,
            bearing_names=args.bearing_names if args.color_by == "bearing" else None,
        )
        
        # Save UMAP embeddings (from first mode)
        modes_str = "_".join(mode_list)
        np.save(
            os.path.join(args.output_dir, f"umap_embeddings_{modes_str}_{args.random_state}.npy"),
            all_umap_embeddings[0]
        )
    
    # Always create t-SNE visualization
    # If multiple modes, create multi-subplot visualization
    modes_str = "_".join(mode_list)
    if n_modes > 1:
        tsne_output_path = os.path.join(
            args.output_dir,
            f"tsne_{first_bearing_clean}_{modes_str}_{args.color_by}_{args.random_state}.svg"
        )
        
        tsne_title = f"t-SNE Visualization {title_suffix}"
        
        # Map mode names to Chinese titles
        mode_title_map = {
            "raw": "输入特征",
            "tcn": "TCN输出特征",
            "attention": "SA层输出特征"
        }
        mode_titles_list = [mode_title_map.get(mode, mode.upper()) for mode in mode_list]
        # Ensure metrics_csv_paths is a list (repeat for each mode if needed)
        metrics_csv_paths_list = args.metrics_csv_paths if isinstance(args.metrics_csv_paths, list) else [args.metrics_csv_paths]
        # Extend metrics_csv_paths_list to match n_modes if needed
        while len(metrics_csv_paths_list) < n_modes:
            metrics_csv_paths_list.append(metrics_csv_paths_list[-1] if metrics_csv_paths_list else "")
        
        visualize_tsne_multi(
            tsne_embeddings_list=all_tsne_embeddings,
            color_by=args.color_by,
            color_values=color_values,
            labels=labels_all.cpu().numpy() if args.color_by != "label" else None,
            output_path=tsne_output_path,
            title=tsne_title,
            bearing_names=args.bearing_names if args.color_by == "bearing" else None,
            metrics_list=all_metrics_list if not args.no_metrics else None,
            model_titles=mode_titles_list,
            metrics_csv_paths=metrics_csv_paths_list,
        )
    else:
        # Single mode: use original single plot function
        tsne_output_path = os.path.join(
            args.output_dir,
            f"tsne_{first_bearing_clean}_{mode_list[0]}_{args.color_by}_{args.random_state}.svg"
        )
        
        tsne_title = f"t-SNE Visualization ({mode_list[0].upper()} features) {title_suffix}"
        
        visualize_tsne(
            embeddings=all_tsne_embeddings[0],
            color_by=args.color_by,
            color_values=color_values,
            labels=labels_all.cpu().numpy() if args.color_by != "label" else None,
            output_path=tsne_output_path,
            title=tsne_title,
            bearing_names=args.bearing_names,  # Always pass bearing_names for metrics filtering
            metrics=all_metrics_list[0] if not args.no_metrics and len(all_metrics_list) > 0 else None,
        )
    
    # Save embeddings for further analysis (from first mode)
    modes_str = "_".join(mode_list)
    np.save(
        os.path.join(args.output_dir, f"tsne_embeddings_{modes_str}_{args.random_state}.npy"),
        all_tsne_embeddings[0]
    )
    print(f"[INFO] Embeddings saved to {args.output_dir}")
    print(f"[INFO] Visualization complete!")


if __name__ == "__main__":
    main()
