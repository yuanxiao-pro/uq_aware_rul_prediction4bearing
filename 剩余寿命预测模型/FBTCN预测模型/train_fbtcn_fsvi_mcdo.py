#!/usr/bin/env python3
"""
FSVI training for Bayesian TCN with MC-Dropout prediction.

Reference:
- auto_train_fbtcn_sa.py: dataset split / logging / config style
- fbtcn_sa_model.py: BayesianTCN backbone

This script trains with function-space variational inference (function KL)
and performs prediction using MC-Dropout to estimate epistemic uncertainty.
Existing files are left untouched; everything lives here.
"""

import argparse
import copy
import json
import os
import re
import time
from itertools import cycle
from typing import Dict, List, Tuple, Optional
import sys
import types
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Disable custom font selection: stub out matplotlib_chinese_config so training uses defaults
if "matplotlib_chinese_config" not in sys.modules:
    sys.modules["matplotlib_chinese_config"] = types.SimpleNamespace(
        setup_chinese_font=lambda *args, **kwargs: (None, None)
    )
# Local imports from existing modules (do not modify originals)
from loss_function import (
    compute_au_nll,
    compute_au_nll_with_crps,
    compute_au_nll_with_crps_and_pos,
    compute_au_nll_with_pos,
)
from function_kl import calculate_function_kl
from stable_fbtcn_training import StabilizedAUNLL, get_stable_optimizer, get_kl_weight_with_warmup
from test_runner import evaluate_and_save_metrics, save_config
from fbtcn_sa_model import BayesianTCN
from isotonic_calibration import run_isotonic_calibration
from auto_train_fbtcn_sa import (
    setup_logger,
    get_logger,
    set_seed,
    load_bearing_data,
    create_data_loaders,
    get_data_dir,
    list_bearings_in_dir,
    get_context_bearings_by_type,
    build_condition_splits,
    format_time,
    evaluate_model_on_validation,
)


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def select_loss_function(cfg: Dict):
    """Choose loss function consistent with existing options."""
    name = cfg.get("loss_function", "au_nll_with_crps_and_pos")
    if name == "au_nll":
        return compute_au_nll
    if name == "au_nll_with_pos":
        return compute_au_nll_with_pos
    if name == "au_nll_with_crps":
        return compute_au_nll_with_crps
    if name == "au_nll_with_crps_and_pos":
        return compute_au_nll_with_crps_and_pos
    return compute_au_nll_with_crps_and_pos


def enable_mc_dropout(model: nn.Module):
    """Turn on dropout layers while keeping other modules in eval mode."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(
    model: nn.Module,
    test_loader: Data.DataLoader,
    mc_passes: int,
    scaler_name: str,
    results_dir: str,
    scaler_dir: str,
    device: Optional[torch.device] = None,
):
    """
    Run MC-Dropout prediction and return arrays compatible with metric utilities.
    """
    from joblib import load
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    enable_mc_dropout(model)

    targets: List[float] = []
    mu_batches: List[np.ndarray] = []
    sigma_batches: List[np.ndarray] = []

    with torch.no_grad():
        for data, label in test_loader:
            targets += label.tolist()
            data = data.to(device)

            mu_samples = []
            sigma_samples = []
            for _ in range(mc_passes):
                enable_mc_dropout(model)  # ensure dropout stays active
                mu, sigma, _ = model(data)
                mu_samples.append(mu.detach().cpu().numpy())
                sigma_samples.append(sigma.detach().cpu().numpy())

            mu_stack = np.stack(mu_samples, axis=0)  # (mc, batch, 1)
            sigma_stack = np.stack(sigma_samples, axis=0)  # (mc, batch, 1)

            mu_batches.append(mu_stack.squeeze(-1))      # (mc, batch)
            sigma_batches.append(sigma_stack.squeeze(-1))  # (mc, batch)

    if len(mu_batches) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    mu_all = np.concatenate(mu_batches, axis=1)  # (mc, total_samples)
    sigma_all = np.concatenate(sigma_batches, axis=1)  # (mc, total_samples)

    mu_mean = mu_all.mean(axis=0)  # (total_samples,)
    sigma_mean = sigma_all.mean(axis=0)  # aleatoric (mean of sigma)

    targets_arr = np.array(targets).reshape(-1)

    # Inverse-scale
    all_files = os.listdir(scaler_dir)
    matched = [f for f in all_files if scaler_name in f and f.endswith("_scaler")]
    if len(matched) == 0:
        raise FileNotFoundError(f"No scaler matched {scaler_name} in {scaler_dir}")
    scaler = load(os.path.join(scaler_dir, matched[0]))
    if not isinstance(scaler, (StandardScaler, MinMaxScaler)):
        raise TypeError(f"Scaler file is not a scaler object: {matched[0]}")

    targets_arr = scaler.inverse_transform(targets_arr.reshape(-1, 1)).reshape(-1)
    mu_mean = scaler.inverse_transform(mu_mean.reshape(-1, 1)).reshape(-1)
    mu_all_inv = scaler.inverse_transform(mu_all.T).T  # (mc, total_samples)

    os.makedirs(results_dir, exist_ok=True)
    return targets_arr, mu_mean, mu_all_inv, sigma_mean, mu_all_inv


def extract_condition_tag(name: str, same_dataset: bool) -> str:
    """Lightweight condition tag for naming artifacts."""
    if not same_dataset:
        return "cross_domain"
    m = re.search(r"(?:c[123]_)?(Bearing\\d+)", name)
    return m.group(1) if m else name


# --------------------------------------------------------------------------- #
# Training loop (FSVI)
# --------------------------------------------------------------------------- #

def train_fsvi(
    model: nn.Module,
    init_model: nn.Module,
    train_loader: Data.DataLoader,
    context_loader: Data.DataLoader,
    val_loader: Data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    config: Dict,
    device: torch.device,
    logger,
):
    epochs = config.get("epochs", 200)
    patience = config.get("patience", 50)
    max_grad_norm = 5.0
    best_state = None
    best_metric = float("inf")
    patience_counter = 0
    kl_weight_base = config.get("kl_weight", 1e-3)

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        context_iter = cycle(context_loader)

        for batch_data, batch_label in train_loader:
            seq = batch_data.to(device, non_blocking=True)
            labels = batch_label.to(device, non_blocking=True)
            context_seq, _ = next(context_iter)
            context_seq = context_seq.to(device, non_blocking=True)

            optimizer.zero_grad()
            mu, sigma, _ = model(seq)
            nll = loss_fn(labels, mu, sigma)

            try:
                fkl = calculate_function_kl(
                    context_seq, model=model, init_model=init_model, enable_diagnosis=False, debug_nan=False
                )
            except Exception:
                fkl = torch.tensor(0.0, device=device)

            kl_weight = get_kl_weight_with_warmup(epoch, epochs, kl_weight_base)
            loss = nll + kl_weight * fkl / config.get("batch_size", 128)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_train_loss = float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else float("inf")
        val_loss = evaluate_model_on_validation(model, val_loader, device, forward_pass=5)
        logger.info(
            f"Epoch {epoch+1:03d}/{epochs} | Train {avg_train_loss:.6f} | "
            f"Val {val_loss:.6f} | KLw {kl_weight:.4e}"
        )

        if val_loss < best_metric:
            best_metric = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# --------------------------------------------------------------------------- #
# Fold runner
# --------------------------------------------------------------------------- #

def run_single_split(
    split_idx: int,
    train_bearings: List[str],
    test_bearings: List[str],
    context_bearings: List[str],
    validation_bearings: List[str],
    config: Dict,
    device: torch.device,
    train_data_dir: str,
    test_data_dir: str,
    same_dataset: bool,
    logger,
):
    train_set, train_label = load_bearing_data(train_bearings, train_data_dir)
    context_set, context_label = load_bearing_data(context_bearings, train_data_dir)
    test_set, test_label = load_bearing_data(test_bearings, test_data_dir)

    train_loader, context_loader, test_loader, val_loader = create_data_loaders(
        train_set,
        train_label,
        context_set,
        context_label,
        test_set,
        test_label,
        test_set,
        test_label,
        config["batch_size"],
        config["test_batch_size"],
        config["seed"],
    )

    model = BayesianTCN(
        input_dim=config["input_dim"],
        num_channels=config["num_channels"],
        attention_dim=config["attention_dim"],
        kernel_size=config["kernel_size"],
        conv_posterior_rho_init=config["conv_posterior_rho_init"],
        output_posterior_rho_init=config["output_posterior_rho_init"],
        dropout=config["dropout"],
        output_dim=config["output_dim"],
        attention_mode=config.get("attention_mode", "self"),
    ).to(device)
    init_model = copy.deepcopy(model)

    optimizer = get_stable_optimizer(model, config)
    loss_fn = select_loss_function(config)

    logger.info(f"Split {split_idx+1}: train {len(train_bearings)} | test {len(test_bearings)}")
    model = train_fsvi(
        model,
        init_model,
        train_loader,
        context_loader,
        val_loader,
        optimizer,
        loss_fn,
        config,
        device,
        logger,
    )

    condition_tag = extract_condition_tag(test_bearings[0], same_dataset)
    res_dir = os.path.join(
        config["results_dir"],
        f"{config['train_datasets_type'].split('_')[0]}_to_{config['test_datasets_type'].split('_')[0]}",
    )
    os.makedirs(res_dir, exist_ok=True)
    best_path = os.path.join(res_dir, f"{condition_tag}_{config['best_pt_model_base_name']}")
    torch.save(model.state_dict(), best_path)
    logger.info(f"Saved best model to {best_path}")

    # MC-Dropout prediction per test bearing (first test, uncalibrated)
    scaler_dir = os.path.join(config["scaler_dir"], config["test_datasets_type"])
    first_test_results = {}
    
    logger.info(f"\n{'='*80}")
    logger.info("第一次测试（未校准，MC-Dropout）")
    logger.info(f"{'='*80}")
    
    for bearing_name_original in test_bearings:
        single_set, single_label = load_bearing_data([bearing_name_original], test_data_dir)
        if len(single_set) == 0:
            logger.warning(f"Skip empty test bearing: {bearing_name_original}")
            continue

        single_loader = Data.DataLoader(
            dataset=Data.TensorDataset(single_set, single_label),
            batch_size=config["test_batch_size"],
            num_workers=0,
            drop_last=False,
            pin_memory=True,
        )

        if config["test_datasets_type"] in ("xjtu_made", "femto_made"):
            scaler_key = bearing_name_original.replace("_labeled", "_labeled_fpt_scaler")
        else:
            scaler_key = bearing_name_original

        logger.info(f"测试轴承: {bearing_name_original} (scaler查找名: {scaler_key})")
        
        targets, preds_mean, preds_samples, alea_sigma, mu_samples = mc_dropout_predict(
            model,
            single_loader,
            mc_passes=config.get("forward_pass", 50),
            scaler_name=scaler_key,
            results_dir=res_dir,
            scaler_dir=scaler_dir,
            device=device,
        )

        if targets.size == 0:
            logger.warning(f"No predictions produced for {bearing_name_original}")
            continue

        # Clean bearing name for saving (remove suffixes)
        bearing_name = bearing_name_original
        if config["test_datasets_type"] in ("xjtu_made", "femto_made"):
            bearing_name = scaler_key.replace("_labeled_fpt_scaler", "")

        # Convert alea_sigma (mean of sigma after softplus) to log_var
        # sigma is log variance after softplus, so log_var ≈ log(sigma) or use sigma directly
        # For MC-Dropout, we approximate log_var from the mean sigma
        # Handle NaN and negative values
        alea_sigma_clean = np.maximum(alea_sigma, 1e-8)  # Ensure positive
        alea_sigma_clean = np.where(np.isnan(alea_sigma_clean), 1e-8, alea_sigma_clean)  # Replace NaN
        log_var_list = np.log(alea_sigma_clean)  # Convert to log variance
        log_var_list = np.clip(log_var_list, -20, 10)  # Clip to reasonable range to prevent overflow

        # Save first test results (uncalibrated)
        metrics_csv = os.path.join(res_dir, f"{bearing_name}_mcdo.csv")
        fig_png = os.path.join(res_dir, f"{bearing_name}_mcdo.png")
        evaluate_and_save_metrics(
            targets, preds_mean, preds_samples, log_var_list, mu_samples, metrics_csv, fig_png, alpha=0.05
        )
        save_config(config, os.path.join(res_dir, f"{bearing_name}_mcdo.json"))
        logger.info(f"MC-Dropout evaluation saved for {bearing_name}")

        # Store results for calibration
        first_test_results[bearing_name] = {
            'target': targets,
            'prediction': preds_mean,
            'origin_prediction': preds_samples,  # (mc_passes, N)
            'log_var_list': log_var_list,
            'mu_samples': mu_samples,  # Same as preds_samples
            'test_loader': single_loader
        }

    # Isotonic calibration
    if len(first_test_results) == 0:
        logger.warning("⚠️  警告: 没有测试结果，跳过校准")
        return
    
    if validation_bearings and len(validation_bearings) > 0:
        use_same_condition_validation = config.get('use_same_condition_validation', True)
        if use_same_condition_validation:
            logger.info(f"使用与测试集同工况的验证集进行等渗回归校准")
            logger.info(f"全部验证集: {validation_bearings}")
            logger.info(f"测试集: {test_bearings}")
        else:
            logger.info(f"使用全部验证集进行等渗回归校准: {validation_bearings}")
        
        run_isotonic_calibration(
            model=model,
            train_bearings=train_bearings,
            train_data_dir=train_data_dir,
            scaler_dir=scaler_dir,
            test_bearings=test_bearings,
            test_datasets_type=config["test_datasets_type"],
            first_test_results=first_test_results,
            config=config,
            device=device,
            res_dir=res_dir,
            load_bearing_data_func=load_bearing_data,
            calibration_bearings=validation_bearings,
            calibration_mode=config.get('calibration_mode', 'train_first'),
            context_bearings=context_bearings,
            use_same_condition_validation=use_same_condition_validation,
            test_data_dir=test_data_dir
        )
    else:
        logger.warning("⚠️  警告: 未配置验证集，使用训练集的第一个子集进行校准")
        run_isotonic_calibration(
            model=model,
            train_bearings=train_bearings,
            train_data_dir=train_data_dir,
            scaler_dir=scaler_dir,
            test_bearings=test_bearings,
            test_datasets_type=config["test_datasets_type"],
            first_test_results=first_test_results,
            config=config,
            device=device,
            res_dir=res_dir,
            load_bearing_data_func=load_bearing_data,
            calibration_bearings=config.get('calibration_bearings', None),
            calibration_mode=config.get('calibration_mode', 'train_first'),
            context_bearings=context_bearings,
            test_data_dir=test_data_dir
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run_fsvi_mcdo(config: Dict):
    logger = setup_logger(log_dir=config.get("log_dir", None))

    use_deterministic = config.get("use_deterministic", True)
    use_benchmark = config.get("use_benchmark", False)
    set_seed(config["seed"], deterministic=use_deterministic, benchmark=use_benchmark)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_type = config.get("train_datasets_type", "xjtu_made")
    test_type = config.get("test_datasets_type", "xjtu_made")
    train_data_dir = get_data_dir(train_type)
    test_data_dir = get_data_dir(test_type)

    train_bearings_all = list_bearings_in_dir(train_data_dir)
    test_bearings_all = list_bearings_in_dir(test_data_dir)
    if len(train_bearings_all) == 0 or len(test_bearings_all) == 0:
        raise ValueError("No bearings found in train/test directories")

    context_bearings = get_context_bearings_by_type(train_type, train_bearings_all, config.get("context_bearings", []))
    validation_bearings = get_context_bearings_by_type(
        train_type, train_bearings_all, config.get("validation_bearings", [])
    )
    exclude_validation = config.get("exclude_validation_from_training", True)

    same_dataset = train_type == test_type
    if same_dataset:
        splits = build_condition_splits(
            train_bearings_all,
            test_bearings_all,
            context_bearings,
            validation_bearings=validation_bearings,
            same_dataset=True,
            exclude_validation_from_training=exclude_validation,
        )
    else:
        splits = [
            {
                "condition": "cross_domain",
                "train_bearings": [b for b in train_bearings_all if b not in context_bearings],
                "test_bearings": test_bearings_all,
            }
        ]

    logger.info(f"Total splits: {len(splits)}")
    start_time = time.time()
    for idx, split in enumerate(splits):
        run_single_split(
            idx,
            split["train_bearings"],
            split["test_bearings"],
            context_bearings,
            validation_bearings,
            config,
            device,
            train_data_dir,
            test_data_dir,
            same_dataset,
            logger,
        )
    logger.info(f"All splits done. Total time: {format_time(time.time() - start_time)}")


def parse_args():
    parser = argparse.ArgumentParser(description="FSVI training with MC-Dropout inference for Bayesian TCN")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to JSON config; if omitted, uses /mnt/uq_aware_rul_prediction4bearing-main/config/baseline/mcd_config.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config_path or "/mnt/uq_aware_rul_prediction4bearing-main/config/baseline/mcd_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 定义训练任务列表
    tasks = [
        # ['xjtu_made', 'xjtu_made'],
        ['xjtu_made', 'femto_made']
        # ['femto_made', 'xjtu_made']
    ]
    
    # 初始化logger
    logger = setup_logger(log_dir=config.get("log_dir", None))
    logger.info("="*80)
    logger.info("FSVI训练脚本启动（MC-Dropout预测）!!!")
    logger.info("="*80)
    
    total_script_start_time = time.time()
    for task_idx, task in enumerate(tasks):
        logger.info(f"\n🚀 训练任务 {task_idx + 1}/{len(tasks)}: {task[0]} -> {task[1]}")
        config['train_datasets_type'] = task[0]
        config['test_datasets_type'] = task[1]
        run_fsvi_mcdo(config)
        logger.info(f"\n训练任务 {task[0]} -> {task[1]} 完成！")
    
    total_script_end_time = time.time()
    logger.info("="*80)
    logger.info(f"✅ 所有训练任务完成！总训练时间: {format_time(total_script_end_time - total_script_start_time)}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

