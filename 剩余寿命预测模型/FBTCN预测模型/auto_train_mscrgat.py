#!/usr/bin/env python3
"""
MSCRGAT 自动训练脚本

参考 auto_train_fbtcn_sa 的训练框架与结果保存逻辑，训练 MSCRGAT（域对抗）。
跨工况/跨域逻辑：
- 同域（xjtu->xjtu）：按工况 2 训 1 测；每次用 2 个工况做源域（有标签），
  用目标工况的 2 个未标注子集（按映射选）做域适应，其余轴承做测试。
- 跨域（xjtu->femto）：源域用 XJTU 全部数据，目标域训练集 = femto 工况1 全部，
  测试集 = femto 全部（工况1+2+3）。
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auto_train_fbtcn_sa import (
    get_data_dir,
    list_bearings_in_dir,
    build_condition_splits,
    normalize_name,
    setup_logger,
    get_logger,
    set_seed,
    format_time,
)
from adversarial_mscrgat import (
    run_adversarial_mscrgat,
    load_adversarial_config,
    default_config,
)

try:
    from metrics import mae, rmse
except ImportError:
    def mae(y, p): return np.mean(np.abs(np.asarray(y) - np.asarray(p)))
    def rmse(y, p): return np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2))

WINDOW_SIZE = 1


def _sliding_windows(
    data: torch.Tensor,
    labels: torch.Tensor,
    window_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对单轴承时序数据做滑窗。返回 X: (N, window_size, F), y: (N,)。"""
    if data.dim() == 3:
        data = data.squeeze(1)
    if labels.dim() > 1:
        labels = labels.squeeze(-1)
    T, F = data.shape
    if T < window_size:
        pad_len = window_size - T
        data_padded = torch.cat([data, data[-1:].expand(pad_len, F)], dim=0)
        last_label = labels[-1:].clone()
        X = data_padded.unsqueeze(0)
        y = last_label
        return X, y
    X_list = []
    y_list = []
    for i in range(0, T - window_size + 1):
        X_list.append(data[i : i + window_size])
        y_list.append(labels[i + window_size - 1])
    X = torch.stack(X_list, dim=0)
    y = torch.stack(y_list, dim=0)
    return X, y


def _load_bearing_data_with_windows(
    bearing_list: List[str],
    data_dir: str,
    window_size: int,
    load_labels: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """按轴承加载数据并做滑窗。load_labels=True 时返回 (X, y)，否则 (X, None)。"""
    x_list, y_list = [], []
    for bearing in bearing_list:
        data_files = [
            f for f in os.listdir(data_dir)
            if bearing in f and (f.endswith("_fpt_data") or f.endswith("_data"))
        ]
        label_files = [
            f for f in os.listdir(data_dir)
            if bearing in f and (f.endswith("_fpt_label") or f.endswith("_label"))
        ] if load_labels else []
        data_files.sort()
        label_files.sort()
        for di, data_file in enumerate(data_files):
            data = joblib_load(os.path.join(data_dir, data_file))
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            if load_labels and di < len(label_files):
                label = joblib_load(os.path.join(data_dir, label_files[di]))
                if not isinstance(label, torch.Tensor):
                    label = torch.tensor(label, dtype=torch.float32)
                X_w, y_w = _sliding_windows(data, label, window_size)
            else:
                label_ph = torch.zeros(data.size(0), dtype=torch.float32)
                X_w, y_w = _sliding_windows(data, label_ph, window_size)
            if X_w.size(0) > 0:
                x_list.append(X_w)
                if load_labels:
                    y_list.append(y_w)
    if not x_list:
        return torch.empty(0), (None if load_labels else torch.empty(0))
    X = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0) if load_labels and y_list else None
    return X, y


TARGET_ADAPT_SECOND_BEARING_MAP = {
    ("xjtu", "1"): "Bearing1_2",
    ("xjtu", "2"): "Bearing2_2",
    ("xjtu", "3"): "Bearing3_4",
    ("femto", "1"): "Bearing1_6",
    ("femto", "2"): "Bearing2_2",
    ("femto", "3"): "Bearing3_2",
}


def _bearing_id(name: str) -> Optional[str]:
    m = re.search(r'Bearing\d+_\d+', normalize_name(name))
    return m.group(0) if m else None


def _condition_group(name: str) -> str:
    key = _bearing_id(name) or normalize_name(name)
    m = re.match(r'Bearing(\d+)_\d+', key)
    return m.group(1) if m else key


def _get_target_adapt_by_mapping(
    condition_id: str,
    test_bearings: List[str],
    test_bearings_all: List[str],
    dataset_type: str,
) -> Tuple[List[str], List[str]]:
    """按映射选目标域训练集；返回 (target_adapt, test_eval)。"""
    prefix = "xjtu" if "xjtu" in dataset_type.lower() else "femto"
    first_id = f"Bearing{condition_id}_1"
    second_id = TARGET_ADAPT_SECOND_BEARING_MAP.get(
        (prefix, condition_id), f"Bearing{condition_id}_2"
    )
    condition_bearings = [b for b in test_bearings_all if _condition_group(b) == condition_id]
    id_to_bearing = {_bearing_id(b): b for b in condition_bearings if _bearing_id(b)}
    target_adapt = []
    for bid in [first_id, second_id]:
        if bid in id_to_bearing and id_to_bearing[bid] not in target_adapt:
            target_adapt.append(id_to_bearing[bid])
    if len(target_adapt) < 2 and len(condition_bearings) > len(target_adapt):
        sorted_rest = sorted(
            [b for b in condition_bearings if b not in target_adapt],
            key=lambda x: (normalize_name(x), x),
        )
        for b in sorted_rest:
            if len(target_adapt) >= 2:
                break
            target_adapt.append(b)
    target_adapt_set = set(target_adapt)
    test_eval = [b for b in test_bearings if b not in target_adapt_set]
    return target_adapt, test_eval


def get_cross_domain_femto_splits(
    train_bearings_all: List[str],
    test_bearings_all: List[str],
    target_adapt_ids: Tuple[str, ...] = ("Bearing1_1", "Bearing1_6"),
    target_condition: Optional[str] = None,
    add_target_condition_to_train: bool = True,
) -> Dict:
    """
    xjtu->femto 跨域划分：
    - 目标域训练集 = femto 工况1 的全部子集（用于域适应，无标签）
    - 测试集 = femto 的全部子集（工况1+2+3，有标签评估）
    - 若 add_target_condition_to_train=True：源域训练集 = XJTU 全部 + femto 工况1 全部（有标签）
    target_condition 为 "1" 时按上述规则；为 None 时按 target_adapt_ids 指定轴承。
    """
    if target_condition is not None:
        target_adapt = [b for b in test_bearings_all if _condition_group(b) == target_condition]
        test_eval = list(test_bearings_all)
        if add_target_condition_to_train:
            train_bearings = list(train_bearings_all) + list(target_adapt)
        else:
            train_bearings = list(train_bearings_all)
    else:
        target_ids = set(target_adapt_ids)
        target_adapt = [b for b in test_bearings_all if _bearing_id(b) in target_ids]
        test_eval = [b for b in test_bearings_all if _bearing_id(b) not in target_ids]
        train_bearings = list(train_bearings_all)
    return {
        'condition': 'cross_domain',
        'train_bearings': train_bearings,
        'target_adapt_bearings': target_adapt,
        'test_eval_bearings': test_eval,
    }


def _align_feature_dim(
    x: torch.Tensor,
    target_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if x.dim() != 3:
        return x
    _, seq, f = x.shape
    if f >= target_dim:
        return x
    pad = torch.zeros(x.size(0), seq, target_dim - f, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-1)


def train_single_fold_mscrgat(
    fold_idx: int,
    train_bearings: List[str],
    target_adapt_bearings: List[str],
    test_eval_bearings: List[str],
    config: Dict,
    device: torch.device,
    train_data_dir: str,
    test_data_dir: str,
    total_folds: int = 1,
    total_start_time: Optional[float] = None,
    skip_eval_and_save: bool = False,
) -> Dict:
    logger = get_logger()

    def pretty_names(names: List[str]) -> List[str]:
        keys = set()
        for b in names:
            m = re.search(r'Bearing\d+_\d+', normalize_name(b))
            if m:
                keys.add(m.group(0))
        return sorted(keys)

    logger.info(f"\n{'='*80}")
    logger.info(f"📊 MSCRGAT 折 {fold_idx + 1}/{total_folds} - 训练开始前数据划分")
    logger.info(f"{'='*80}")
    logger.info(f"源域训练集 ({len(train_bearings)} 个): {train_bearings}")
    logger.info(f"目标域训练集 ({len(target_adapt_bearings)} 个): {target_adapt_bearings}")
    logger.info(f"测试集 ({len(test_eval_bearings)} 个): {test_eval_bearings}")
    logger.info(f"源域(精简): {', '.join(pretty_names(train_bearings))}")
    logger.info(f"目标域(精简): {', '.join(pretty_names(target_adapt_bearings))}")
    logger.info(f"测试(精简): {', '.join(pretty_names(test_eval_bearings))}")

    window_size = config.get('window_size', WINDOW_SIZE)
    input_dim = config.get('input_dim', 13)

    source_x, source_y = _load_bearing_data_with_windows(
        train_bearings, train_data_dir, window_size, load_labels=True
    )
    target_x, _ = _load_bearing_data_with_windows(
        target_adapt_bearings, test_data_dir, window_size, load_labels=False
    )
    if target_x.numel() == 0:
        target_x, _ = _load_bearing_data_with_windows(
            target_adapt_bearings, train_data_dir, window_size, load_labels=False
        )

    if source_x.numel() == 0:
        raise ValueError("源域数据为空，请检查 train_bearings 与数据目录。")
    if target_x.numel() == 0:
        raise ValueError("目标域适应数据为空，请检查 target_adapt_bearings 与数据目录。")

    if source_x.shape[2] != input_dim:
        config = {**config, 'input_dim': source_x.shape[2]}
        input_dim = source_x.shape[2]
    source_x = _align_feature_dim(source_x.to(device), input_dim, device)
    target_x = _align_feature_dim(target_x.to(device), input_dim, device)
    source_x = source_x.cpu().float()
    target_x = target_x.cpu().float()
    source_y = source_y.cpu().float()

    source_x = torch.nan_to_num(source_x, nan=0.0, posinf=0.0, neginf=0.0)
    source_y = torch.nan_to_num(source_y, nan=0.0, posinf=0.0, neginf=0.0)
    target_x = torch.nan_to_num(target_x, nan=0.0, posinf=0.0, neginf=0.0)

    N_s, T, n_feat = source_x.shape
    scaler_x = StandardScaler()
    scaler_x.fit(source_x.reshape(-1, n_feat).numpy())
    source_x = torch.tensor(
        scaler_x.transform(source_x.reshape(-1, n_feat).numpy()).reshape(N_s, T, n_feat),
        dtype=source_x.dtype,
    )
    target_x = torch.tensor(
        scaler_x.transform(target_x.reshape(-1, n_feat).numpy()).reshape(target_x.shape[0], T, n_feat),
        dtype=target_x.dtype,
    )

    sy_min, sy_max = source_y.min().item(), source_y.max().item()
    span_s = sy_max - sy_min
    if span_s < 1e-8:
        span_s = 1.0
    source_y_norm = ((source_y - sy_min) / span_s).float().clamp(0.0, 1.0)
    y_scale_save = span_s
    y_offset_save = sy_min

    fold_start = time.time()
    train_datasets_type = config.get('train_datasets_type', 'xjtu_made_mscrgat')
    test_datasets_type = config.get('test_datasets_type', 'xjtu_made_mscrgat')
    results_base = config.get('results_dir', './mscrgat_results/').rstrip(os.sep)
    subdir = config.get(
        'results_subdir',
        train_datasets_type.split('_')[0] + '_to_' + test_datasets_type.split('_')[0],
    )
    res_dir = os.path.join(results_base, subdir)
    os.makedirs(res_dir, exist_ok=True)

    condition = config.get('condition', 'all')
    if isinstance(condition, str) and condition != 'all':
        best_pt_name = f"{condition}_{config.get('best_pt_model_base_name', 'best_mscrgat.pt')}"
    else:
        best_pt_name = config.get('best_pt_model_base_name', 'best_mscrgat.pt')
    best_pt_path = os.path.join(res_dir, best_pt_name)

    model, discriminator, info = run_adversarial_mscrgat(
        source_x, source_y_norm, target_x, config=config, device=device, best_model_path=best_pt_path
    )
    info['y_scale'] = y_scale_save
    info['y_offset'] = y_offset_save

    logger.info(f"最佳模型（训练 loss 最低）已保存: {best_pt_path}")
    training_time = time.time() - fold_start

    if skip_eval_and_save:
        return {
            'fold_idx': fold_idx,
            'model_path': best_pt_path,
            'scaler_x': scaler_x,
            'n_feat': n_feat,
            'T': T,
            'input_dim': input_dim,
            'y_scale': y_scale_save,
            'y_offset': y_offset_save,
            'config': config,
            'test_eval_bearings': test_eval_bearings,
        }

    model.load_state_dict(torch.load(best_pt_path, map_location=device))
    logger.info(f"已加载最佳模型用于测试: {best_pt_path}")

    test_results = {}
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
        te_x = torch.tensor(
            scaler_x.transform(te_x.cpu().reshape(-1, n_feat).numpy()).reshape(te_x.shape[0], T, n_feat),
            dtype=te_x.dtype,
        )
        ty_min = te_y.min().item()
        ty_max = te_y.max().item()
        span_t = ty_max - ty_min
        if span_t < 1e-8:
            span_t = 1.0
        te_y_norm = ((te_y - ty_min) / span_t).float().clamp(0.0, 1.0)
        test_bs = config.get('test_batch_size', config.get('batch_size', 64))

        model.eval()
        pred_list, y_list = [], []
        with torch.no_grad():
            loader = Data.DataLoader(
                Data.TensorDataset(te_x.cpu(), te_y_norm.cpu()),
                batch_size=test_bs, shuffle=False,
            )
            for bx, by in loader:
                bx = bx.to(device)
                out = model(bx)
                pred = out[0] if getattr(model, 'heteroscedastic', False) else out
                pred_list.append(pred.cpu().numpy().ravel())
                y_list.append(by.numpy().ravel())
        pred_np = np.concatenate(pred_list, axis=0)
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
                pred_orig = np.clip(pred_np, 0.0, 1.0).astype(np.float64)
            except Exception:
                pred_orig = pred_np * span_t + ty_min
                y_true_orig = te_y_np
        else:
            pred_orig = pred_np * span_t + ty_min
            y_true_orig = te_y_np

        metrics = {"MAE": mae(y_true_orig, pred_orig), "RMSE": rmse(y_true_orig, pred_orig)}

        bearing_short = re.sub(r'[^Bearing0-9_]', '_', normalize_name(bearing_name))
        bearing_short = bearing_short.strip('_') or bearing_name

        csv_path = os.path.join(res_dir, f"{bearing_short}_mscrgat_metrics.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            cols = list(metrics.keys())
            vals = [str(metrics[k]) for k in cols]
            f.write(",".join(cols) + "\n")
            f.write(",".join(vals) + "\n")

        pred_csv_path = os.path.join(res_dir, f"{bearing_short}_mscrgat_predictions.csv")
        with open(pred_csv_path, 'w', encoding='utf-8') as f:
            f.write("true_rul,pred_rul\n")
            for i in range(len(y_true_orig)):
                f.write(f"{y_true_orig[i]},{pred_orig[i]}\n")

        json_path = os.path.join(res_dir, f"{bearing_short}_mscrgat_config.json")
        save_cfg = {**config, 'y_scale': y_scale_save, 'y_offset': y_offset_save}
        if config.get('scaler_dir') is not None:
            save_cfg['scaler_dir'] = config['scaler_dir']
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(save_cfg, f, indent=2, ensure_ascii=False)
        test_results[bearing_name] = metrics

    return {
        'fold_idx': fold_idx,
        'train_bearings': train_bearings,
        'target_adapt_bearings': target_adapt_bearings,
        'test_eval_bearings': test_eval_bearings,
        'model_path': best_pt_path,
        'config': config,
        'training_time': training_time,
        'test_results': test_results,
        'y_scale': y_scale_save,
        'y_offset': y_offset_save,
        'scaler_x': scaler_x,
        'n_feat': n_feat,
        'T': T,
        'input_dim': input_dim,
    }


def main(config: Dict) -> Dict:
    logger = setup_logger(log_dir=config.get('log_dir'))
    set_seed(config.get('seed', 42), deterministic=config.get('use_deterministic', True), benchmark=config.get('use_benchmark', False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

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
    logger.info(f"训练轴承数: {len(train_bearings_all)}, 测试轴承数: {len(test_bearings_all)}")

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
        # xjtu->femto：目标域训练集 = femto 工况1 全部，测试集 = femto 全部（工况1+2+3）
        target_condition = config.get('cross_domain_target_condition', '1')
        target_adapt_ids = tuple(config.get('cross_domain_target_adapt', ['Bearing1_1', 'Bearing1_6']))
        split_dict = get_cross_domain_femto_splits(
            train_bearings_all,
            test_bearings_all,
            target_adapt_ids,
            target_condition=target_condition,
        )
        splits = [split_dict]

    logger.info(f"总轮数: {len(splits)}")
    total_start = time.time()
    all_results = []

    for loop_idx, split in enumerate(splits):
        cfg = {**config, 'condition': split.get('condition', 'all')}
        result = train_single_fold_mscrgat(
            loop_idx,
            split['train_bearings'],
            split['target_adapt_bearings'],
            split['test_eval_bearings'],
            cfg,
            device,
            train_data_dir,
            test_data_dir,
            total_folds=len(splits),
            total_start_time=total_start,
        )
        all_results.append(result)

    total_time = time.time() - total_start
    logger.info(f"\n{'='*80}\n✅ 全部完成\n{'='*80}")
    logger.info(f"总时间: {format_time(total_time)}")
    return {'all_results': all_results, 'config': config, 'total_time': total_time}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MSCRGAT 自动训练（域对抗）")
    parser.add_argument('--config_path', type=str, default=None, help='MSCRGAT 配置 JSON')
    args = parser.parse_args()

    config_path = args.config_path or os.path.join(
        os.path.dirname(__file__), '../../config/baseline/adversarial_mscrgat_config.json'
    )
    if os.path.isfile(config_path):
        config = load_adversarial_config(config_path)
    else:
        config = default_config()

    config.setdefault('results_dir', os.path.join(os.path.dirname(__file__), '../../auto_baselines_result/mscrgat/'))
    config.setdefault('scaler_dir', os.path.join(os.path.dirname(__file__), '../../datasetresult/'))
    config.setdefault('log_dir', os.path.join(os.path.dirname(__file__), '../../auto_baselines_result/logs/mscrgat/'))
    config.setdefault('best_pt_model_base_name', 'best_model_mscrgat.pt')
    config.setdefault('num_target_adapt', 2)
    config.setdefault('window_size', WINDOW_SIZE)

    tasks = [
        # ['xjtu_made_mscrgat', 'xjtu_made_mscrgat'],
        ['xjtu_made_mscrgat', 'femto_made_mscrgat'],
    ]
    for task in tasks:
        config['train_datasets_type'] = task[0]
        config['test_datasets_type'] = task[1]
        get_logger().info(f"\n🚀 任务: {task[0]} -> {task[1]}")
        main(config)
