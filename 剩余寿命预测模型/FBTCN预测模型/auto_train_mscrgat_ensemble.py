#!/usr/bin/env python3
"""
MSCRGAT Deep Ensemble 训练脚本

使用独立随机初始化的 deep ensemble 策略：训练多个 MSCRGAT 模型（不同 seed），
推理时对预测取平均。单个模型的训练流程与 auto_train_mscrgat 一致。
"""

import json
import os
import sys
import re
import time
import random
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.utils.data as Data
from joblib import load as joblib_load
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auto_train_mscrgat import (
    WINDOW_SIZE,
    _sliding_windows,
    _load_bearing_data_with_windows,
    _align_feature_dim,
    _bearing_id,
    _condition_group,
    _get_target_adapt_by_mapping,
    get_cross_domain_femto_splits,
    train_single_fold_mscrgat,
)
from auto_train_fbtcn_sa import (
    get_data_dir,
    list_bearings_in_dir,
    build_condition_splits,
    normalize_name,
    setup_logger,
    get_logger,
    set_seed,
    format_time,
    get_context_bearings_by_type,
)
from adversarial_mscrgat import (
    run_adversarial_mscrgat,
    load_adversarial_config,
    default_config,
)
# 等渗回归校准（与 auto_train_fbtcn_sa 一致）
from isotonic_calibration import fit_calibrator, calibrate_test_results, filter_validation_by_condition

try:
    from metrics import mae, rmse, picp, nmpiw, cwc
except ImportError:
    def mae(y, p): return np.mean(np.abs(np.asarray(y) - np.asarray(p)))
    def rmse(y, p): return np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2))
    def picp(yt, yl, yu): return np.mean((np.asarray(yt) >= np.asarray(yl)) & (np.asarray(yt) <= np.asarray(yu)))
    def nmpiw(yl, yu, R): return np.sum(np.asarray(yu) - np.asarray(yl)) / (R * len(yl)) if len(yl) > 0 else 0.0
    def cwc(picp_val, nmpiw_val, alpha=0.05, eta=2):
        mu = 1.0 - alpha
        gamma = 1.0 if picp_val < mu else 0.0
        return float(nmpiw_val * (1.0 + gamma * np.exp(-eta * (picp_val - mu))))

DEFAULT_ENSEMBLE_SIZE = 5
ENSEMBLE_STRATEGY_RANDOM_SEED = 'random_seed'
ENSEMBLE_STRATEGY_BAGGING = 'bagging'


def _predict_on_calibration_mscrgat_ensemble(
    models: List[torch.nn.Module],
    calibration_bearings: List[str],
    train_data_dir: str,
    test_data_dir: str,
    scaler_base: str,
    test_datasets_type: str,
    config: Dict,
    device: torch.device,
    window_size: int,
    input_dim: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    在校准集上做 MSCRGAT ensemble 预测，返回 (val_target, val_prediction, val_pred_std) 供等渗回归拟合。
    与 auto_train_fbtcn_sa 中 run_isotonic_calibration 使用的校准数据接口一致（数值为反归一化后）。
    """
    logger = get_logger()
    if not calibration_bearings:
        return None, None, None

    all_te_x, all_te_y = None, None
    for b in calibration_bearings:
        te_x, te_y = _load_bearing_data_with_windows(
            [b], test_data_dir, window_size, load_labels=True
        )
        if te_x is None or te_x.numel() == 0:
            te_x, te_y = _load_bearing_data_with_windows(
                [b], train_data_dir, window_size, load_labels=True
            )
        if te_x is None or te_x.numel() == 0:
            continue
        if all_te_x is None:
            all_te_x, all_te_y = te_x, te_y
        else:
            all_te_x = torch.cat([all_te_x, te_x], dim=0)
            all_te_y = torch.cat([all_te_y, te_y], dim=0)

    if all_te_x is None or all_te_x.numel() == 0:
        logger.warning("校准集无有效数据，跳过等渗回归校准。")
        return None, None, None

    all_te_x = _align_feature_dim(all_te_x.float(), input_dim, device)
    all_te_x = torch.nan_to_num(all_te_x, nan=0.0, posinf=0.0, neginf=0.0)
    scaler_x = config.get('_calibration_scaler_x')  # 由调用方在需要时注入
    if scaler_x is not None and hasattr(scaler_x, 'transform'):
        n_f = all_te_x.shape[2]
        all_te_x = torch.tensor(
            scaler_x.transform(all_te_x.cpu().reshape(-1, n_f).numpy()).reshape(
                all_te_x.shape[0], all_te_x.shape[1], n_f
            ),
            dtype=all_te_x.dtype,
        )
    ty_min = all_te_y.min().item()
    ty_max = all_te_y.max().item()
    span_t = max(ty_max - ty_min, 1e-8)

    test_bs = config.get('test_batch_size', config.get('batch_size', 64))
    alpha = config.get('prediction_interval_alpha', 0.05)
    heteroscedastic = any(getattr(m, 'heteroscedastic', False) for m in models)

    preds_per_model = [[] for _ in range(len(models))]
    sigma_per_model = [[] for _ in range(len(models))]
    with torch.no_grad():
        loader = Data.DataLoader(
            Data.TensorDataset(all_te_x.cpu(), all_te_y.cpu()),
            batch_size=test_bs,
            shuffle=False,
        )
        for bx, _ in loader:
            bx = bx.to(device)
            for mi, m in enumerate(models):
                out = m(bx)
                if getattr(m, 'heteroscedastic', False):
                    p = out[0].cpu().numpy().ravel()
                    sig = out[1].cpu().numpy().ravel()
                    preds_per_model[mi].append(p)
                    sigma_per_model[mi].append(sig)
                else:
                    p = out.cpu().numpy().ravel()
                    preds_per_model[mi].append(p)
    for mi in range(len(models)):
        preds_per_model[mi] = np.concatenate(preds_per_model[mi], axis=0)
    preds_stack = np.stack(preds_per_model, axis=0)
    pred_mean_norm = np.mean(preds_stack, axis=0)
    EU = np.var(preds_stack, axis=0)
    if heteroscedastic:
        for mi in range(len(models)):
            sigma_per_model[mi] = np.concatenate(sigma_per_model[mi], axis=0)
        sigma_stack = np.stack(sigma_per_model, axis=0)
        AU = np.mean(sigma_stack ** 2, axis=0)
    else:
        AU = np.zeros_like(EU)
    total_std_norm = np.sqrt(np.maximum(AU + EU, 1e-12))
    te_y_np = all_te_y.numpy().ravel() if hasattr(all_te_y, 'numpy') else np.asarray(all_te_y).ravel()

    scaler_search_dir = os.path.join(scaler_base, test_datasets_type) if scaler_base else test_data_dir
    if not os.path.isdir(scaler_search_dir):
        scaler_search_dir = test_data_dir
    scaler_obj = None
    first_bearing = calibration_bearings[0]
    if os.path.isdir(scaler_search_dir):
        all_files = os.listdir(scaler_search_dir)
        matched = [f for f in all_files if first_bearing in f and (f.endswith('_scaler') or f.endswith('_fpt_scaler'))]
        if matched:
            try:
                scaler_obj = joblib_load(os.path.join(scaler_search_dir, matched[0]))
            except Exception:
                pass
    if scaler_obj is not None and hasattr(scaler_obj, 'inverse_transform'):
        try:
            val_target = scaler_obj.inverse_transform(te_y_np.reshape(-1, 1)).ravel()
            val_prediction = np.clip(pred_mean_norm, 0.0, 1.0).astype(np.float64)
            val_prediction = scaler_obj.inverse_transform(val_prediction.reshape(-1, 1)).ravel()
            scale_std = np.abs(scaler_obj.data_max_ - scaler_obj.data_min_).max() if hasattr(scaler_obj, 'data_max_') else 1.0
            if scale_std < 1e-8:
                scale_std = 1.0
            val_pred_std = total_std_norm * scale_std
        except Exception:
            val_target = te_y_np
            val_prediction = pred_mean_norm * span_t + ty_min
            val_pred_std = total_std_norm * span_t
    else:
        val_target = te_y_np
        val_prediction = pred_mean_norm * span_t + ty_min
        val_pred_std = total_std_norm * span_t

    logger.info(f"校准集样本数: {len(val_target)}, 预测均值范围: [{val_prediction.min():.4f}, {val_prediction.max():.4f}]")
    return val_target, val_prediction, val_pred_std


def train_and_evaluate_ensemble(
    fold_idx: int,
    train_bearings: List[str],
    target_adapt_bearings: List[str],
    test_eval_bearings: List[str],
    config: Dict,
    device: torch.device,
    train_data_dir: str,
    test_data_dir: str,
    total_folds: int,
    ensemble_size: int,
    total_start_time: Optional[float] = None,
    validation_bearings: Optional[List[str]] = None,
) -> Dict:
    """
    Deep Ensemble：用不同 seed 训练 ensemble_size 个模型，推理时对预测取平均。
    单模型训练流程与 auto_train_mscrgat.train_single_fold_mscrgat 一致。
    """
    logger = get_logger()

    def pretty_names(names: List[str]) -> List[str]:
        keys = set()
        for b in names:
            m = re.search(r'Bearing\d+_\d+', normalize_name(b))
            if m:
                keys.add(m.group(0))
        return sorted(keys)

    ensemble_strategy = config.get('ensemble_strategy', ENSEMBLE_STRATEGY_RANDOM_SEED)
    bagging_sample_ratio = config.get('bagging_sample_ratio', 0.8)

    logger.info(f"\n{'='*80}")
    logger.info(f"📊 MSCRGAT Ensemble 折 {fold_idx + 1}/{total_folds} (ensemble_size={ensemble_size}, strategy={ensemble_strategy})")
    logger.info(f"{'='*80}")
    logger.info(f"源域: {', '.join(pretty_names(train_bearings))}")
    logger.info(f"目标域: {', '.join(pretty_names(target_adapt_bearings))}")
    logger.info(f"测试: {', '.join(pretty_names(test_eval_bearings))}")

    window_size = config.get('window_size', WINDOW_SIZE)
    input_dim = config.get('input_dim', 13)
    base_seed = config.get('seed', 42)

    train_datasets_type = config.get('train_datasets_type', 'xjtu_made_mscrgat')
    test_datasets_type = config.get('test_datasets_type', 'xjtu_made_mscrgat')
    results_base = config.get('results_dir', './mscrgat_results/').rstrip(os.sep)
    subdir_base = train_datasets_type.split('_')[0] + '_to_' + test_datasets_type.split('_')[0] + '_ensemble'
    subdir = subdir_base + f'_{ensemble_strategy}' if ensemble_strategy != ENSEMBLE_STRATEGY_RANDOM_SEED else subdir_base
    res_dir = os.path.join(results_base, subdir)
    os.makedirs(res_dir, exist_ok=True)

    condition = config.get('condition', 'all')
    if isinstance(condition, str) and condition != 'all':
        pt_prefix = f"{condition}_best_mscrgat_ensemble"
    else:
        pt_prefix = "best_mscrgat_ensemble"

    model_paths: List[str] = []
    first_result = None

    for ei in range(ensemble_size):
        seed_i = base_seed + ei
        best_pt_name = f"{pt_prefix}_{ei}.pt"

        set_seed(seed_i, deterministic=config.get('use_deterministic', True), benchmark=config.get('use_benchmark', False))

        if ensemble_strategy == ENSEMBLE_STRATEGY_BAGGING:
            n_sample = max(1, int(len(train_bearings) * bagging_sample_ratio))
            train_bearings_i = np.random.choice(train_bearings, size=n_sample, replace=True).tolist()
            logger.info(f"\n--- 训练 Ensemble 成员 {ei + 1}/{ensemble_size} (seed={seed_i}, Bagging 采样 {n_sample}/{len(train_bearings)} 个轴承) ---")
        else:
            train_bearings_i = train_bearings
            logger.info(f"\n--- 训练 Ensemble 成员 {ei + 1}/{ensemble_size} (seed={seed_i}) ---")

        cfg_i = {
            **config,
            'seed': seed_i,
            'best_pt_model_base_name': best_pt_name,
            'results_subdir': subdir,
        }
        result = train_single_fold_mscrgat(
            fold_idx=fold_idx,
            train_bearings=train_bearings_i,
            target_adapt_bearings=target_adapt_bearings,
            test_eval_bearings=test_eval_bearings,
            config=cfg_i,
            device=device,
            train_data_dir=train_data_dir,
            test_data_dir=test_data_dir,
            total_folds=total_folds,
            total_start_time=total_start_time,
            skip_eval_and_save=True,
        )

        saved_path = result.get('model_path')
        if saved_path and os.path.isfile(saved_path):
            model_paths.append(saved_path)
        else:
            alt_path = os.path.join(res_dir, best_pt_name)
            if os.path.isfile(alt_path):
                model_paths.append(alt_path)
            else:
                logger.warning(f"Ensemble 成员 {ei} 模型未找到，跳过该成员")
        if first_result is None:
            first_result = result

    if len(model_paths) == 0:
        raise RuntimeError("无有效模型，ensemble 失败")

    logger.info(f"\n--- Ensemble 推理 ({len(model_paths)} 个模型) ---")
    from adversarial_mscrgat import build_mscrgat

    models = []
    for p in model_paths:
        m = build_mscrgat(config, device)
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        models.append(m)

    scaler_x = first_result.get('scaler_x') if first_result else None
    y_scale_save = first_result.get('y_scale', 1.0) if first_result else 1.0
    y_offset_save = first_result.get('y_offset', 0.0) if first_result else 0.0
    n_feat = first_result.get('n_feat', input_dim) if first_result else input_dim
    T = first_result.get('T', window_size) if first_result else window_size

    # 用于等渗回归校准：与 auto_train_fbtcn_sa 一致的 first_test_results 结构
    first_test_results: Dict[str, Dict] = {}
    test_results = {}
    config.setdefault('ci', 1.0 - config.get('prediction_interval_alpha', 0.05))

    for bearing_name in test_eval_bearings:
        te_x, te_y = _load_bearing_data_with_windows(
            [bearing_name], test_data_dir, window_size, load_labels=True
        )
        if te_x is None or te_x.numel() == 0:
            te_x, te_y = _load_bearing_data_with_windows(
                [bearing_name], train_data_dir, window_size, load_labels=True
            )
        if te_x is None or te_x.numel() == 0:
            logger.warning(f"测试轴承 {bearing_name} 无数据，跳过。")
            continue

        te_x = _align_feature_dim(te_x.float(), input_dim, device)
        te_x = torch.nan_to_num(te_x, nan=0.0, posinf=0.0, neginf=0.0)

        if scaler_x is not None and hasattr(scaler_x, 'transform'):
            n_f = te_x.shape[2]
            te_x = torch.tensor(
                scaler_x.transform(te_x.cpu().reshape(-1, n_f).numpy()).reshape(
                    te_x.shape[0], te_x.shape[1], n_f
                ),
                dtype=te_x.dtype,
            )
        ty_min = te_y.min().item()
        ty_max = te_y.max().item()
        span_t = ty_max - ty_min
        if span_t < 1e-8:
            span_t = 1.0
        test_bs = config.get('test_batch_size', config.get('batch_size', 64))
        alpha = config.get('prediction_interval_alpha', 0.05)
        heteroscedastic = any(getattr(m, 'heteroscedastic', False) for m in models)

        preds_per_model = [[] for _ in range(len(models))]
        sigma_per_model = [[] for _ in range(len(models))]
        with torch.no_grad():
            loader = Data.DataLoader(
                Data.TensorDataset(te_x.cpu(), te_y.cpu()),
                batch_size=test_bs,
                shuffle=False,
            )
            for bx, _ in loader:
                bx = bx.to(device)
                for mi, m in enumerate(models):
                    out = m(bx)
                    if getattr(m, 'heteroscedastic', False):
                        p = out[0].cpu().numpy().ravel()
                        sig = out[1].cpu().numpy().ravel()
                        preds_per_model[mi].append(p)
                        sigma_per_model[mi].append(sig)
                    else:
                        p = out.cpu().numpy().ravel()
                        preds_per_model[mi].append(p)
        for mi in range(len(models)):
            preds_per_model[mi] = np.concatenate(preds_per_model[mi], axis=0)
        preds_stack = np.stack(preds_per_model, axis=0)
        pred_mean_norm = np.mean(preds_stack, axis=0)

        EU = np.var(preds_stack, axis=0)
        if heteroscedastic:
            for mi in range(len(models)):
                sigma_per_model[mi] = np.concatenate(sigma_per_model[mi], axis=0)
            sigma_stack = np.stack(sigma_per_model, axis=0)
            AU = np.mean(sigma_stack ** 2, axis=0)
        else:
            AU = np.zeros_like(EU)
        total_var = AU + EU
        total_std = np.sqrt(np.maximum(total_var, 1e-12))
        z = norm.ppf(1 - alpha / 2)
        y_lower_norm = pred_mean_norm - z * total_std
        y_upper_norm = pred_mean_norm + z * total_std
        te_y_np = te_y.numpy().ravel() if hasattr(te_y, 'numpy') else np.asarray(te_y).ravel()

        scaler_base = (config.get('scaler_dir') or '').rstrip('/')
        scaler_search_dir = os.path.join(scaler_base, test_datasets_type) if scaler_base else test_data_dir
        if not os.path.isdir(scaler_search_dir):
            scaler_search_dir = test_data_dir
        scaler_obj = None
        if os.path.isdir(scaler_search_dir):
            all_files = os.listdir(scaler_search_dir)
            matched = [f for f in all_files if bearing_name in f and (f.endswith('_scaler') or f.endswith('_fpt_scaler'))]
            if matched:
                try:
                    scaler_obj = joblib_load(os.path.join(scaler_search_dir, matched[0]))
                except Exception:
                    pass
        if scaler_obj is not None and hasattr(scaler_obj, 'inverse_transform'):
            try:
                y_true_orig = scaler_obj.inverse_transform(te_y_np.reshape(-1, 1)).ravel()
                pred_orig = np.clip(pred_mean_norm, 0.0, 1.0).astype(np.float64)
                y_lower_orig = np.clip(y_lower_norm, 0.0, 1.0).astype(np.float64)
                y_upper_orig = np.clip(y_upper_norm, 0.0, 1.0).astype(np.float64)
            except Exception:
                pred_orig = pred_mean_norm * span_t + ty_min
                y_lower_orig = y_lower_norm * span_t + ty_min
                y_upper_orig = y_upper_norm * span_t + ty_min
                y_true_orig = te_y_np
        else:
            pred_orig = pred_mean_norm * span_t + ty_min
            y_lower_orig = y_lower_norm * span_t + ty_min
            y_upper_orig = y_upper_norm * span_t + ty_min
            y_true_orig = te_y_np

        if scaler_obj is None:
            preds_orig_per_model = [preds_stack[i] * span_t + ty_min for i in range(preds_stack.shape[0])]
        else:
            preds_orig_per_model = [np.clip(preds_stack[i], 0.0, 1.0).astype(np.float64) for i in range(preds_stack.shape[0])]

        mean_au = float(np.mean(AU))
        R = float(np.max(y_true_orig) - np.min(y_true_orig)) if y_true_orig.size > 0 and np.max(y_true_orig) != np.min(y_true_orig) else 1.0
        picp_val = picp(y_true_orig, y_lower_orig, y_upper_orig)
        nmpiw_val = nmpiw(y_lower_orig, y_upper_orig, R)
        cwc_val = cwc(picp_val, nmpiw_val, alpha=alpha)
        metrics = {
            "MAE": mae(y_true_orig, pred_orig),
            "RMSE": rmse(y_true_orig, pred_orig),
            "PICP": picp_val,
            "NMPIW": nmpiw_val,
            "CWC": cwc_val,
            "mean_AU": mean_au,
        }

        bearing_short = re.sub(r'[^Bearing0-9_]', '_', normalize_name(bearing_name))
        bearing_short = bearing_short.strip('_') or bearing_name

        # 供等渗回归校准使用（与 isotonic_calibration.calibrate_test_results 接口一致）
        std_orig = np.maximum((y_upper_orig - y_lower_orig) / (2.0 * z), 1e-12)
        first_test_results[bearing_short] = {
            'target': y_true_orig,
            'prediction': pred_orig,
            'origin_prediction': pred_orig.reshape(1, -1),
            'log_var_list': np.log(np.maximum(std_orig ** 2, 1e-12)),
            'mu_samples': pred_orig.reshape(1, -1),
        }

        csv_path = os.path.join(res_dir, f"{bearing_short}_mscrgat_ensemble_metrics.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            cols = list(metrics.keys())
            vals = [str(metrics[k]) for k in cols]
            f.write(",".join(cols) + "\n")
            f.write(",".join(vals) + "\n")

        pred_csv_path = os.path.join(res_dir, f"{bearing_short}_mscrgat_ensemble_predictions.csv")
        model_cols = [f"pred_model_{i}" for i in range(len(models))]
        header = "true_rul,pred_rul," + ",".join(model_cols) + ",y_lower,y_upper,au"
        with open(pred_csv_path, 'w', encoding='utf-8') as f:
            f.write(header + "\n")
            for i in range(len(y_true_orig)):
                model_vals = ",".join(str(preds_orig_per_model[m][i]) for m in range(len(models)))
                f.write(f"{y_true_orig[i]},{pred_orig[i]},{model_vals},{y_lower_orig[i]},{y_upper_orig[i]},{AU[i]}\n")

        json_path = os.path.join(res_dir, f"{bearing_short}_mscrgat_ensemble_config.json")
        save_cfg = {**config, 'ensemble_size': ensemble_size, 'y_scale': y_scale_save, 'y_offset': y_offset_save}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(save_cfg, f, indent=2, ensure_ascii=False)
        test_results[bearing_name] = metrics

    # 等渗回归校准（与 auto_train_fbtcn_sa 一致）
    if first_test_results:
        use_same_condition_validation = config.get('use_same_condition_validation', True)
        if validation_bearings and len(validation_bearings) > 0:
            if use_same_condition_validation:
                selected_calibration = filter_validation_by_condition(validation_bearings, test_eval_bearings)
                calibration_bearings = selected_calibration if selected_calibration else validation_bearings
                logger.info(f"等渗回归校准：使用与测试集同工况的验证集 {calibration_bearings}")
            else:
                calibration_bearings = validation_bearings
                logger.info(f"等渗回归校准：使用全部验证集 {calibration_bearings}")
        else:
            calibration_bearings = [train_bearings[0]] if train_bearings else []
            logger.warning("未配置 validation_bearings，使用训练集第一个轴承进行校准")
        if calibration_bearings:
            cfg_cal = {**config, '_calibration_scaler_x': scaler_x}
            scaler_base = (config.get('scaler_dir') or '').rstrip('/')
            val_target, val_prediction, val_pred_std = _predict_on_calibration_mscrgat_ensemble(
                models=models,
                calibration_bearings=calibration_bearings,
                train_data_dir=train_data_dir,
                test_data_dir=test_data_dir,
                scaler_base=scaler_base,
                test_datasets_type=test_datasets_type,
                config=cfg_cal,
                device=device,
                window_size=window_size,
                input_dim=input_dim,
            )
            if val_target is not None and val_prediction is not None and val_pred_std is not None:
                logger.info("拟合等渗回归校准器并对测试集校准...")
                calibrator = fit_calibrator(val_target, val_prediction, val_pred_std, config)
                calibrate_test_results(calibrator, first_test_results, config, res_dir)
            else:
                logger.warning("校准集预测失败，跳过等渗回归校准。")

    return {
        'fold_idx': fold_idx,
        'train_bearings': train_bearings,
        'target_adapt_bearings': target_adapt_bearings,
        'test_eval_bearings': test_eval_bearings,
        'model_paths': model_paths,
        'config': config,
        'test_results': test_results,
        'ensemble_size': len(model_paths),
    }


def main(config: Dict) -> Dict:
    logger = setup_logger(log_dir=config.get('log_dir'))
    ensemble_size = config.get('ensemble_size', DEFAULT_ENSEMBLE_SIZE)
    ensemble_strategy = config.get('ensemble_strategy', ENSEMBLE_STRATEGY_RANDOM_SEED)
    set_seed(
        config.get('seed', 42),
        deterministic=config.get('use_deterministic', True),
        benchmark=config.get('use_benchmark', False),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}, ensemble_size: {ensemble_size}, ensemble_strategy: {ensemble_strategy}")

    train_datasets_type = config.get('train_datasets_type', 'xjtu_made_mscrgat')
    test_datasets_type = config.get('test_datasets_type', 'xjtu_made_mscrgat')
    train_data_dir = get_data_dir(train_datasets_type)
    test_data_dir = get_data_dir(test_datasets_type)

    train_bearings_all = list_bearings_in_dir(train_data_dir)
    test_bearings_all = list_bearings_in_dir(test_data_dir)
    if not train_bearings_all:
        raise ValueError(f"训练数据目录无轴承: {train_data_dir}")
    if not test_bearings_all:
        raise ValueError(f"测试数据目录无轴承: {test_data_dir}")

    logger.info(f"训练数据集: {train_datasets_type}, 测试数据集: {test_datasets_type}")

    # 验证集轴承（用于等渗回归校准，与 auto_train_fbtcn_sa 一致）
    validation_bearings = get_context_bearings_by_type(
        train_datasets_type,
        train_bearings_all,
        config.get('validation_bearings', []),
    )
    if validation_bearings:
        logger.info(f"验证集轴承(用于校准): {validation_bearings}")

    same_dataset = (train_datasets_type == test_datasets_type)

    if same_dataset:
        splits = build_condition_splits(
            train_bearings_all,
            test_bearings_all,
            context_bearings=config.get('context_bearings', []),
            validation_bearings=config.get('validation_bearings', []),
            same_dataset=True,
            exclude_validation_from_training=config.get('exclude_validation_from_training', True),
        )
        if not splits:
            raise ValueError("同域工况划分为空，请检查数据与 context/validation 配置。")
        for s in splits:
            test_list = s['test_bearings']
            cond_id = s.get('condition', '?')
            target_adapt, test_eval = _get_target_adapt_by_mapping(
                cond_id, test_list, test_bearings_all, test_datasets_type
            )
            s['target_adapt_bearings'] = target_adapt
            s['test_eval_bearings'] = test_eval
            s['condition'] = cond_id
    else:
        target_condition = config.get('cross_domain_target_condition', '1')
        target_adapt_ids = tuple(config.get('cross_domain_target_adapt', ['Bearing1_1', 'Bearing1_6']))
        add_target_to_train = config.get('cross_domain_add_target_condition_to_train', True)
        split_dict = get_cross_domain_femto_splits(
            train_bearings_all,
            test_bearings_all,
            target_adapt_ids,
            target_condition=target_condition,
            add_target_condition_to_train=add_target_to_train,
        )
        splits = [split_dict]

    logger.info(f"总轮数: {len(splits)}")
    total_start = time.time()
    all_results = []

    for loop_idx, split in enumerate(splits):
        cfg = {**config, 'condition': split.get('condition', 'all')}
        result = train_and_evaluate_ensemble(
            loop_idx,
            split['train_bearings'],
            split['target_adapt_bearings'],
            split['test_eval_bearings'],
            cfg,
            device,
            train_data_dir,
            test_data_dir,
            total_folds=len(splits),
            ensemble_size=ensemble_size,
            total_start_time=total_start,
            validation_bearings=validation_bearings,
        )
        all_results.append(result)

    total_time = time.time() - total_start
    logger.info(f"\n{'='*80}\n✅ Ensemble 全部完成\n{'='*80}")
    logger.info(f"总时间: {format_time(total_time)}")
    return {'all_results': all_results, 'config': config, 'total_time': total_time}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MSCRGAT Deep Ensemble 训练")
    parser.add_argument('--config_path', type=str, default=None, help='MSCRGAT 配置 JSON')
    parser.add_argument('--ensemble_size', type=int, default=DEFAULT_ENSEMBLE_SIZE, help='Ensemble 模型数量')
    parser.add_argument(
        '--ensemble_strategy',
        type=str,
        default=None,
        choices=[ENSEMBLE_STRATEGY_RANDOM_SEED, ENSEMBLE_STRATEGY_BAGGING],
        help=f'集成策略: {ENSEMBLE_STRATEGY_RANDOM_SEED}=随机种子, {ENSEMBLE_STRATEGY_BAGGING}=Bagging (未指定则用 config)'
    )
    parser.add_argument('--bagging_sample_ratio', type=float, default=None, help='Bagging 采样比例 (未指定则用 config)')
    args = parser.parse_args()

    config_path = args.config_path or os.path.join(
        os.path.dirname(__file__), '../../config/baseline/ens_mscrgat_config_seed0.json'
    )
    if os.path.isfile(config_path):
        config = load_adversarial_config(config_path)
    else:
        config = default_config()

    config.setdefault('results_dir', os.path.join(os.path.dirname(__file__), '../../auto_baselines_result/ens_mscrgat/'))
    config.setdefault('scaler_dir', os.path.join(os.path.dirname(__file__), '../../datasetresult/'))
    config.setdefault('log_dir', os.path.join(os.path.dirname(__file__), '../../auto_baselines_result/logs/ens_mscrgat/'))
    config.setdefault('best_pt_model_base_name', 'best_model_ens_mscrgat.pt')
    config.setdefault('window_size', WINDOW_SIZE)
    config['ensemble_size'] = args.ensemble_size
    if args.ensemble_strategy is not None:
        config['ensemble_strategy'] = args.ensemble_strategy
    else:
        config.setdefault('ensemble_strategy', ENSEMBLE_STRATEGY_RANDOM_SEED)
    if args.bagging_sample_ratio is not None:
        config['bagging_sample_ratio'] = args.bagging_sample_ratio
    else:
        config.setdefault('bagging_sample_ratio', 0.8)
    print(config['heteroscedastic'])
    print(config['ensemble_strategy'])
    print(config['epochs'])
    tasks = [
        ['xjtu_made_mscrgat', 'xjtu_made_mscrgat'],
        ['xjtu_made_mscrgat', 'femto_made_mscrgat'],
    ]
    for task in tasks:
        config['train_datasets_type'] = task[0]
        config['test_datasets_type'] = task[1]
        get_logger().info(f"\n🚀 任务: {task[0]} -> {task[1]} (Ensemble)")
        main(config)
