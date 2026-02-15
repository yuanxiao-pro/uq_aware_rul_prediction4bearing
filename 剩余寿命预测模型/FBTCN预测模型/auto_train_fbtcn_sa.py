#!/usr/bin/env python3
"""
FBTCN自动训练脚本（新版本）
支持K折交叉验证、超参数搜索和详细进度显示
"""

import json
import os
import sys
import time
import copy
import random
import re
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import argparse
from joblib import load
from itertools import product
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import KFold
import re
# 添加路径以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from loss_function import compute_au_nll_with_crps_and_pos, compute_au_nll, compute_au_nll_with_crps, compute_au_nll_with_pos
from stable_fbtcn_training import model_train_stable, get_stable_optimizer
# ==================== 模型定义 ====================
from fbtcn_sa_model import BayesianTCN
# ==================== 测试函数 ====================
from test_runner import run_test_and_save, save_config, save_model_and_config, evaluate_and_save_metrics
# ==================== 等渗回归校准 ====================
from isotonic_calibration import run_isotonic_calibration
# ==================== 日志设置 ====================
_logger_initialized = False

def setup_logger(log_dir: str = None, log_filename: str = None, force_new: bool = False) -> logging.Logger:
    """
    设置日志记录器，同时输出到控制台和文件
    
    Args:
        log_dir: 日志文件目录，如果为None则使用当前目录下的logs文件夹
        log_filename: 日志文件名，如果为None则使用时间戳生成
        force_new: 是否强制创建新的logger（即使已存在）
    
    Returns:
        logger: 配置好的日志记录器
    """
    global _logger_initialized
    
    # 创建logger
    logger = logging.getLogger('FBTCN_Training')
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler（除非强制新建）
    if logger.handlers and not force_new:
        return logger
    
    # 如果强制新建，清除现有handlers
    if force_new and logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
    
    # 创建日志目录
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名
    if log_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'fbtcn_training_{timestamp}.log'
    
    log_path = os.path.join(log_dir, log_filename)
    
    # 文件handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    if not _logger_initialized or force_new:
        logger.info(f"日志文件已创建: {log_path}")
        _logger_initialized = True
    
    return logger

def get_logger() -> logging.Logger:
    """
    获取已存在的logger，如果不存在则创建新的
    
    Returns:
        logger: 日志记录器
    """
    logger = logging.getLogger('FBTCN_Training')
    if not logger.handlers:
        return setup_logger()
    return logger

# ==================== 工具函数 ====================
def set_seed(seed_value, deterministic=True, benchmark=False):
    """设置随机种子"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
    os.environ['PYTHONHASHSEED'] = str(seed_value)


def load_bearing_data(bearing_list: List[str], data_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    加载轴承数据
    
    Args:
        bearing_list: 轴承名称列表
        data_dir: 数据目录
    
    Returns:
        data: 数据张量
        labels: 标签张量
    """
    data_list = []
    label_list = []
    
    for bearing in bearing_list:
        # 尝试两种文件格式
        data_files = [f for f in os.listdir(data_dir) if bearing in f and (f.endswith('_fpt_data') or f.endswith('_data'))]
        label_files = [f for f in os.listdir(data_dir) if bearing in f and (f.endswith('_fpt_label') or f.endswith('_label'))]
        
        data_files.sort()
        label_files.sort()
        
        for data_file, label_file in zip(data_files, label_files):
            data = load(os.path.join(data_dir, data_file))
            label = load(os.path.join(data_dir, label_file))
            data_list.append(data)
            label_list.append(label)
    
    if len(data_list) > 0:
        data_all = torch.cat([torch.tensor(d, dtype=torch.float32) if not isinstance(d, torch.Tensor) else d 
                             for d in data_list], dim=0)
        label_all = torch.cat([torch.tensor(l, dtype=torch.float32) if not isinstance(l, torch.Tensor) else l 
                              for l in label_list], dim=0)
    else:
        data_all = torch.empty(0)
        label_all = torch.empty(0)
    
    return data_all, label_all


def create_data_loaders(train_set, train_label, context_set, context_label, 
                       test_set, test_label, validation_set, validation_label,
                       batch_size, test_batch_size, seed, workers=0):
    """创建数据加载器"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(train_set, train_label),
        batch_size=batch_size, num_workers=workers, drop_last=False, shuffle=True,
        pin_memory=True, persistent_workers=True if workers > 0 else False,
        generator=generator
    )
    context_loader = Data.DataLoader(
        dataset=Data.TensorDataset(context_set, context_label),
        batch_size=batch_size, num_workers=workers, drop_last=False, shuffle=True,
        pin_memory=True, persistent_workers=True if workers > 0 else False,
        generator=generator
    )
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(test_set, test_label),
        batch_size=test_batch_size, num_workers=workers, drop_last=False,
        pin_memory=True, persistent_workers=True if workers > 0 else False
    )
    validation_loader = Data.DataLoader(
        dataset=Data.TensorDataset(validation_set, validation_label),
        batch_size=test_batch_size, num_workers=workers, drop_last=False,
        pin_memory=True, persistent_workers=True if workers > 0 else False
    )
    
    return train_loader, context_loader, test_loader, validation_loader


def evaluate_model_on_validation(model, validation_loader, device, forward_pass=10):
    """
    在验证集上评估模型
    
    Returns:
        validation_loss: 验证集损失
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for data, label in validation_loader:
            data, label = data.to(device), label.to(device)
            mu_list = []
            for _ in range(forward_pass):
                mu, sigma, kl = model(data)
                mu_list.append(mu.cpu().numpy())
            
            mu_mean = np.mean(np.stack(mu_list, axis=0), axis=0)
            loss = np.mean((mu_mean - label.cpu().numpy()) ** 2)
            total_loss += loss * len(label)
            n_samples += len(label)
    
    return total_loss / n_samples if n_samples > 0 else float('inf')


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}小时{minutes}分钟"

# ==================== 数据集与划分辅助函数 ====================

def get_data_dir(dataset_type: str) -> str:
    """根据数据集类型获取数据目录"""
    base_paths = [
        os.path.join(os.path.dirname(__file__), '../../datasetresult'),
        os.path.join(os.path.dirname(__file__), 'datasetresult'),
        'datasetresult'
    ]
    for base_path in base_paths:
        data_dir = os.path.join(base_path, dataset_type)
        if os.path.exists(data_dir):
            return data_dir
    return os.path.join('datasetresult', dataset_type)

def list_bearings_in_dir(data_dir: str) -> List[str]:
    """
    使用正则表达式扫描数据目录，提取轴承名称（去掉 *_fpt_data/_fpt_label/_data/_label 这些后缀）
    """
    if not os.path.exists(data_dir):
        return []
    files = os.listdir(data_dir)
    names = set()
    pattern = re.compile(r'^(.+?)(?:_fpt_data|_fpt_label|_data|_label)$')
    for f in files:
        m = pattern.match(f)
        if m:
            names.add(m.group(1))
    # print("list_bearings_in_dir_regex", names)
    return sorted(list(names))

def normalize_name(name: str) -> str:
    """去除数据集前缀，只保留从 'Bearing' 开始的部分，用于重叠检查"""
    if 'Bearing' in name:
        return name[name.index('Bearing'):]
    return name


def build_splits(train_bearings_all: List[str], test_bearings_all: List[str], context_bearings: List[str]) -> List[Dict[str, List[str]]]:
    """
    构建逐轴承 2训1测 划分：
    - 每次选择一个测试轴承
    - 训练集 = train_bearings_all - context - 当前测试轴承
    - 确保三者互不重叠
    """
    context_norm = {normalize_name(b) for b in context_bearings}
    splits = []
    for test_bearing in test_bearings_all:
        # 跳过在上下文中的测试轴承
        if normalize_name(test_bearing) in context_norm:
            continue
        test_bearings = [test_bearing]
        train_bearings = [
            b for b in train_bearings_all
            if normalize_name(b) not in context_norm
            and b not in test_bearings
        ]
        if len(train_bearings) == 0:
            continue
        # 重叠检查
        assert set(train_bearings).isdisjoint(test_bearings)
        assert all(normalize_name(b) not in context_norm for b in train_bearings)
        splits.append({
            'train_bearings': train_bearings,
            'test_bearings': test_bearings
        })
    return splits


def get_context_bearings_by_type(train_datasets_type: str, train_bearings_all: List[str], context_bearings: List[str]) -> List[str]:
    """
    根据数据集类型生成固定的 context_bearings
    当前策略：
      - 对 xjtu*：优先选择训练集中以 '_1' 结尾的轴承作为上下文
      - 若未找到，则回退到配置中的 context_bearings
      - 其他类型：直接使用配置中的 context_bearings
    """
    if len(context_bearings) == 0:
        return [b for b in train_bearings_all if '_1' in b]
    # print("train_bearings_all", train_bearings_all)
    if train_datasets_type.startswith('xjtu'):
        ctx_opt = [opt.replace('xjtu_', '') for opt in context_bearings if 'xjtu' in opt]
        print("ctx_opt", ctx_opt)
        # ctx = [b for b in train_bearings_all if b.startswith('c') and '_1' in b]
        ctx = [b for opt in ctx_opt for b in train_bearings_all if opt in b]
        print("get_context_bearings_by_type xjtu", ctx)
        if ctx:
            return ctx
    elif train_datasets_type.startswith('femto'):
        # ctx = [b for b in train_bearings_all if b.startswith('Bearing') and '_1' in b]
        ctx_opt = [opt.replace('femto_', '') for opt in context_bearings if 'femto' in opt]
        print("ctx_opt", ctx_opt)
        # ctx = [b for b in train_bearings_all if 'Bearing1_1' in b or 'Bearing2_3' in b or 'Bearing3_1' in b]
        ctx = [b for opt in ctx_opt for b in train_bearings_all if opt in b]
        print("get_context_bearings_by_type femto", ctx)
        if ctx:
            return ctx
    return [b for b in train_bearings_all if '_1' in b]


def build_condition_splits(
    train_bearings_all: List[str],
    test_bearings_all: List[str],
    context_bearings: List[str],
    validation_bearings: List[str] = None,
    same_dataset: bool = True,
    exclude_validation_from_training: bool = True,
) -> List[Dict[str, List[str]]]:
    """
    按工况分组的 2训1测：
    - "工况"按 Bearing 前面的数字来划分（例如 Bearing3_1, Bearing3_3, Bearing3_5 都属于工况3）
    - 每次选择一个工况作为测试，测试集为该工况下的所有轴承（排除上下文，根据配置决定是否排除验证集）
    - 当 train_datasets_type == test_datasets_type（same_dataset=True）时：
        * 训练集为其余工况的轴承，排除上下文，根据配置决定是否排除验证集（经典 2 训 1 测按工况）
      当 train_datasets_type != test_datasets_type（same_dataset=False）时：
        * 训练集可以使用全部工况数据（排除上下文，根据配置决定是否排除验证集），无需再排除与测试工况相同的条件
    - 这样 femto 同域情况下应为 3 轮：工况1 / 工况2 / 工况3
    - validation_bearings: 验证集轴承列表，用于等渗回归校准
    - exclude_validation_from_training: 是否从训练中排除验证集（默认True，即验证集不参与训练）
    """
    def condition_key(name: str) -> str:
        """
        将各种文件名/前缀还原为“物理轴承名”的工况键，例如：
        - c1_Bearing1_1_labeled   -> Bearing1_1
        - c1_Bearing1_1ed         -> Bearing1_1
        - Bearing1_1_labeled      -> Bearing1_1
        - Bearing1_1ed            -> Bearing1_1
        - femto_Bearing2_3_xxx    -> Bearing2_3
        """
        base = normalize_name(name)
        # 优先用正则直接抽取 BearingX_Y
        m = re.search(r'Bearing\d+_\d+', base)
        if m:
            return m.group(0)
        parts = base.split('_')
        return '_'.join(parts[:2]) if len(parts) >= 2 else base

    def condition_group(name: str) -> str:
        """
        将物理轴承名进一步映射为“工况编号”，例如：
        - Bearing1_1 / Bearing1_5 -> '1'
        - Bearing3_2 / Bearing3_5 -> '3'
        这样就可以按工况 1/2/3 来划分 3 轮训练。
        """
        key = condition_key(name)  # e.g. Bearing3_5
        # print("condition_group", key)
        m = re.match(r'Bearing(\d+)_\d+', key)
        if m:
            return m.group(1)
        # 兜底：若不符合模式，则直接返回 key
        return key

    context_norm = {normalize_name(b) for b in context_bearings}
    validation_norm = {normalize_name(b) for b in (validation_bearings or [])} if exclude_validation_from_training else set()

    # 先按"工况编号"把测试集轴承分组（比如 femto 下应该得到 3 个工况：'1','2','3'）
    group_to_test: Dict[str, List[str]] = {}
    for b in test_bearings_all:
        g = condition_group(b)
        group_to_test.setdefault(g, []).append(b)

    # 同样按工况把训练集轴承分组
    group_to_train: Dict[str, List[str]] = {}
    for b in train_bearings_all:
        g = condition_group(b)
        group_to_train.setdefault(g, []).append(b)

    splits: List[Dict[str, List[str]]] = []
    for g, test_list_all in group_to_test.items():
        # 当前工况 g 的测试轴承（排除上下文，根据配置决定是否排除验证集）
        test_list = [b for b in test_list_all 
                     if normalize_name(b) not in context_norm 
                     and (not exclude_validation_from_training or normalize_name(b) not in validation_norm)]
        if not test_list:
            continue

        # 训练轴承：
        # - 同数据集类型：其余工况（!= g）的全部训练轴承，排除上下文，根据配置决定是否排除验证集
        # - 异数据集类型：可以使用所有工况的训练轴承，排除上下文，根据配置决定是否排除验证集（跨域场景下不需要排除与测试相同工况）
        train_list: List[str] = []
        for other_g, train_bs in group_to_train.items():
            if same_dataset and other_g == g:
                continue
            train_list.extend(
                b for b in train_bs
                if normalize_name(b) not in context_norm
                and (not exclude_validation_from_training or normalize_name(b) not in validation_norm)
            )
        if not train_list:
            continue

        # 重叠与上下文检查
        assert set(train_list).isdisjoint(test_list)
        assert all(normalize_name(b) not in context_norm for b in train_list)
        assert all(normalize_name(b) not in context_norm for b in test_list)
        # 只有当排除验证集时才检查验证集重叠
        if exclude_validation_from_training:
            assert all(normalize_name(b) not in validation_norm for b in train_list)
            assert all(normalize_name(b) not in validation_norm for b in test_list)

        splits.append({
            'condition': g,
            'train_bearings': train_list,
            'test_bearings': test_list,
        })

    return splits

# ==================== 超参数搜索 ====================

def hyperparameter_search(config: Dict, train_bearings: List[str], context_bearings: List[str],
                         validation_bearings: List[str], device: torch.device, 
                         train_data_dir: str, test_data_dir: str, n_trials: int = 20) -> Dict:
    """
    超参数搜索（在验证集上进行）
    
    Args:
        config: 配置字典
        train_bearings: 训练集轴承列表
        context_bearings: 上下文集轴承列表（固定）
        validation_bearings: 验证集轴承列表
        device: 设备
        train_data_dir: 训练数据目录
        test_data_dir: 测试数据目录
        n_trials: 搜索次数
    
    Returns:
        best_params: 最佳超参数
    """
    # print("\n" + "="*80)
    # print("🔍 开始超参数搜索（在验证集上进行）")
    # print("="*80)
    
    # 定义超参数搜索空间
    param_grid = {
        'num_channels': [
            [32, 64, 32],
            # [64, 128, 64],
            # [32, 64, 128, 64],
            # [16, 32, 64, 32],
            # [64, 128, 128, 64],
        ],
        'kernel_size': [3, 5, 7, 9],
        # 'dropout': [0.1, 0.2, 0.3, 0.4],
        'learn_rate': [0.001, 0.01, 0.03, 0.05],
        # 'kl_weight': [1e-6, 1e-5, 1e-4, 1e-3],
        # 'output_posterior_rho_init': [-3, -2, -1, 0],
    }
    
    # 加载数据（训练集和上下文集使用训练数据目录，验证集使用测试数据目录）
    train_set, train_label = load_bearing_data(train_bearings, train_data_dir)
    context_set, context_label = load_bearing_data(context_bearings, train_data_dir)
    validation_set, validation_label = load_bearing_data(validation_bearings, test_data_dir)
    
    if len(train_set) == 0 or len(validation_set) == 0:
        # print("⚠️  警告：训练集或验证集为空，跳过超参数搜索，使用默认参数")
        return {
            'num_channels': config['num_channels'],
            'kernel_size': config['kernel_size'],
            'learn_rate': config['learn_rate'],
        }
    
    # 创建数据加载器
    batch_size = config['batch_size']
    test_batch_size = config['test_batch_size']
    seed = config['seed']
    
    train_loader, context_loader, _, validation_loader = create_data_loaders(
        train_set, train_label, context_set, context_label,
        torch.empty(0), torch.empty(0), validation_set, validation_label,
        batch_size, test_batch_size, seed
    )
    
    # 随机搜索
    best_score = float('inf')
    best_params = None
    
    # 生成随机参数组合
    param_combinations = []
    for _ in range(n_trials):
        params = {
            'num_channels': random.choice(param_grid['num_channels']),
            'kernel_size': random.choice(param_grid['kernel_size']),
            'learn_rate': random.choice(param_grid['learn_rate']),
        }
        param_combinations.append(params)
    
    # 搜索进度条
    search_pbar = tqdm(param_combinations, desc="超参数搜索", ncols=120, 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for trial_idx, params in enumerate(search_pbar):
        # 更新配置
        trial_config = config.copy()
        trial_config.update(params)
        
        # 创建模型
        # 注意：dropout、kl_weight 和 output_posterior_rho_init 使用配置文件中的值，不参与搜索
        model = BayesianTCN(
            input_dim=config['input_dim'],
            num_channels=params['num_channels'],
            attention_dim=config.get('attention_dim', 1),
            kernel_size=params['kernel_size'],
            dropout=config['dropout'],  # 使用配置文件中的值
            output_dim=config['output_dim'],
            output_posterior_rho_init=config['output_posterior_rho_init'],  # 使用配置文件中的值
            conv_posterior_rho_init=config.get('conv_posterior_rho_init', -2),
            attention_mode=config.get('attention_mode', 'self'),
        ).to(device)
        
        init_model = copy.deepcopy(model)
        optimizer = get_stable_optimizer(model, trial_config)
        
        # 快速训练（只训练几个epoch进行评估）
        search_epochs = min(10, config.get('epochs', 1000) // 10)
        
        try:
            train_losses, _, _ = model_train_stable(
                search_epochs, model, init_model, optimizer, compute_au_nll_with_crps_and_pos,
                train_loader, context_loader, validation_loader, device, trial_config,
                skip_validation=True
            )
            
            # 在验证集上评估
            val_score = evaluate_model_on_validation(model, validation_loader, device, forward_pass=5)
            
            search_pbar.set_postfix({
                'trial': f"{trial_idx+1}/{n_trials}",
                'val_loss': f"{val_score:.6f}",
                'best': f"{best_score:.6f}" if best_score != float('inf') else "N/A"
            })
            
            if val_score < best_score:
                best_score = val_score
                best_params = params.copy()
                search_pbar.set_description(f"超参数搜索 [最佳: {best_score:.6f}]")
        
        except Exception as e:
            # print(f"\n⚠️  试验 {trial_idx+1} 失败: {e}")
            continue
    
    search_pbar.close()
    
    # print(f"\n✓ 超参数搜索完成")
    # print(f"最佳验证损失: {best_score:.6f}")
    # print(f"最佳超参数:")
    # for key, value in best_params.items():
    #     print(f"  {key}: {value}")
    
    return best_params


# ==================== K折交叉验证 ====================

def k_fold_split(bearings: List[str], k: int = 5, seed: int = 42) -> List[Tuple[List[str], List[str]]]:
    """
    K折交叉验证分割
    
    Args:
        bearings: 所有轴承列表
        k: 折数
        seed: 随机种子
    
    Returns:
        folds: [(train_bearings, test_bearings), ...] 列表
    """
    set_seed(seed, deterministic=False, benchmark=False)
    
    bearings_shuffled = bearings.copy()
    random.shuffle(bearings_shuffled)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    
    for train_idx, test_idx in kf.split(bearings_shuffled):
        train_bearings = [bearings_shuffled[i] for i in train_idx]
        test_bearings = [bearings_shuffled[i] for i in test_idx]
        folds.append((train_bearings, test_bearings))
    return folds

# ==================== 单折训练 ====================
def train_single_fold(fold_idx: int, train_bearings: List[str], test_bearings: List[str],
                      context_bearings: List[str], config: Dict, device: torch.device,
                      train_data_dir: str, test_data_dir: str, total_folds: int = 1, 
                      total_start_time: float = None) -> Dict:
    """
    训练单个折
    
    Args:
        fold_idx: 折索引
        train_bearings: 训练集轴承列表
        test_bearings: 测试集轴承列表
        context_bearings: 上下文集轴承列表
        config: 配置字典
        device: 设备
        train_data_dir: 训练数据目录
        test_data_dir: 测试数据目录
        total_folds: 总折数
        total_start_time: 总开始时间
    
    Returns:
        results: 训练结果字典
    """
    # 获取logger（如果已存在）
    logger = get_logger()
    
    # 显示当前折的信息（按物理工况简洁展示）
    def pretty_names(names: List[str]) -> List[str]:
        keys = set()
        for b in names:
            m = re.search(r'Bearing\d+_\d+', normalize_name(b))
            if m:
                keys.add(m.group(0))
        return sorted(keys)

    logger.info(f"\n{'='*80}")
    logger.info(f"📊 折 {fold_idx + 1}/{total_folds} - 训练信息")
    logger.info(f"{'='*80}")
    logger.info(f"训练集轴承(工况): {', '.join(pretty_names(train_bearings))}")
    logger.info(f"测试集轴承(工况): {', '.join(pretty_names(test_bearings))}")
    if context_bearings:
        logger.info(f"上下文集轴承(工况): {', '.join(pretty_names(context_bearings))}")
    
    # 加载数据（训练集和上下文集使用训练数据目录，测试集使用测试数据目录）
    train_set, train_label = load_bearing_data(train_bearings, train_data_dir)
    context_set, context_label = load_bearing_data(context_bearings, train_data_dir)
    test_set, test_label = load_bearing_data(test_bearings, test_data_dir)
    
    # 验证不重叠
    train_set_bearings = set(train_bearings)
    test_set_bearings = set(test_bearings)
    context_set_bearings = set(context_bearings)
    
    overlap_train_test = train_set_bearings & test_set_bearings
    overlap_train_context = train_set_bearings & context_set_bearings
    overlap_test_context = test_set_bearings & context_set_bearings
    
    if overlap_train_test or overlap_train_context or overlap_test_context:
        logger.warning(f"\n⚠️  警告: 发现数据重叠!")
        if overlap_train_test:
            logger.warning(f"  训练集与测试集重叠: {overlap_train_test}")
        if overlap_train_context:
            logger.warning(f"  训练集与上下文集重叠: {overlap_train_context}")
        if overlap_test_context:
            logger.warning(f"  测试集与上下文集重叠: {overlap_test_context}")
    
    # 创建数据加载器
    train_loader, context_loader, test_loader, validation_loader = create_data_loaders(
        train_set, train_label, context_set, context_label,
        test_set, test_label, test_set, test_label,  # 使用测试集作为验证集
        config['batch_size'], config['test_batch_size'], config['seed']
    )
    
    # 创建模型
    model = BayesianTCN(
        input_dim=config['input_dim'],        
        num_channels=config['num_channels'],
        attention_dim=config['attention_dim'],
        kernel_size=config['kernel_size'],
        conv_posterior_rho_init=config['conv_posterior_rho_init'],
        output_posterior_rho_init=config['output_posterior_rho_init'],
        dropout=config['dropout'],
        output_dim=config['output_dim'],
        attention_mode=config.get('attention_mode', 'self')
    ).to(device)
    
    init_model = copy.deepcopy(model)
    optimizer = get_stable_optimizer(model, config)
    
    # 获取logger（如果已存在）
    logger = get_logger()
    
    # 计算预计时间
    if total_start_time is not None and fold_idx > 0:
        elapsed = time.time() - total_start_time
        avg_time_per_fold = elapsed / fold_idx
        remaining_folds = total_folds - fold_idx
        estimated_remaining = avg_time_per_fold * remaining_folds
        logger.info(f"\n预计剩余时间: {format_time(estimated_remaining)}")
    
    # 训练
    epochs = config['epochs']
    logger.info(f"测试集轴承: {test_bearings}")
    text = test_bearings[0]
    if config['train_datasets_type'] == config['test_datasets_type']:
        match = re.search(r'(?:c[123]_)?(Bearing\d+)', text)
        if match:
            condiction = match.group(1)  # 提取第一个捕获组
            logger.info(f"提取的工况标识: {condiction}")
        else:
            condiction = "Unknown"
            logger.warning(f"未能从 {text} 中提取工况标识")
    else:
        condiction = "Bearing"
    
    fold_start_time = time.time()
    res_dir = config['results_dir']+config['train_datasets_type'].split('_')[0]+'_to_'+config['test_datasets_type'].split('_')[0]+'/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    best_pt_model_name = condiction + '_' + config['best_pt_model_base_name']
    where_best_pt_model_name = res_dir + best_pt_model_name
    logger.info(f"最佳模型保存路径: {where_best_pt_model_name}")
    if config['loss_function'] == 'au_nll':
        loss_function = compute_au_nll
    elif config['loss_function'] == 'au_nll_with_pos':
        loss_function = compute_au_nll_with_pos
    elif config['loss_function'] == 'au_nll_with_crps':
        loss_function = compute_au_nll_with_crps
    elif config['loss_function'] == 'au_nll_with_crps_and_pos':
        loss_function = compute_au_nll_with_crps_and_pos
    else:
        raise ValueError(f"不支持的损失函数: {config['loss_function']}")
    train_losses, val_losses, best_epoch = model_train_stable(
        epochs, model, init_model, optimizer, loss_function,
        train_loader, context_loader, validation_loader, device, config,
        where_best_pt_model_name,
        skip_validation=True
    )
    
    training_time = time.time() - fold_start_time
    
    return {
        'fold_idx': fold_idx,
        'train_bearings': train_bearings,
        'test_bearings': test_bearings,
        'context_bearings': context_bearings,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'training_time': training_time,
        'model': init_model,
        'model_path': where_best_pt_model_name,
        'config': config,
        'test_loader': test_loader
    }


# ==================== 主函数 ====================

def main(config):
    """主函数"""
    # 初始化日志记录器
    log_dir = config.get('log_dir', None)
    logger = setup_logger(log_dir=log_dir)
       
    # 设置随机种子
    use_deterministic = config.get('use_deterministic', True)
    use_benchmark = config.get('use_benchmark', False)
    set_seed(config['seed'], deterministic=use_deterministic, benchmark=use_benchmark)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 根据配置中的数据集类型选择数据目录（不再从 json 提供轴承列表）
    train_datasets_type = config.get('train_datasets_type', 'xjtu')
    test_datasets_type = config.get('test_datasets_type', 'xjtu')
    train_data_dir = get_data_dir(train_datasets_type)
    test_data_dir = get_data_dir(test_datasets_type)
    # 自动扫描轴承列表
    train_bearings_all = list_bearings_in_dir(train_data_dir)
    test_bearings_all = list_bearings_in_dir(test_data_dir)
    if len(train_bearings_all) == 0:
        logger.error(f"训练数据目录 {train_data_dir} 未找到任何轴承文件")
        raise ValueError(f"训练数据目录 {train_data_dir} 未找到任何轴承文件")
    if len(test_bearings_all) == 0:
        logger.error(f"测试数据目录 {test_data_dir} 未找到任何轴承文件")
        raise ValueError(f"测试数据目录 {test_data_dir} 未找到任何轴承文件")
    
    logger.info(f"训练数据集类型: {train_datasets_type}, 测试数据集类型: {test_datasets_type}")
    logger.info(f"训练数据目录: {train_data_dir}, 测试数据目录: {test_data_dir}")
    logger.info(f"训练集轴承数量: {len(train_bearings_all)}, 测试集轴承数量: {len(test_bearings_all)}")

    # 生成固定的上下文轴承（基于数据集类型，若无则回退到配置）
    context_bearings = get_context_bearings_by_type(
        train_datasets_type,
        train_bearings_all,
        config.get('context_bearings', [])
    )
    
    # 生成验证集轴承（用于等渗回归校准）
    validation_bearings = get_context_bearings_by_type(
        train_datasets_type,
        train_bearings_all,
        config.get('validation_bearings', [])
    )
    
    # 是否从训练中排除验证集（默认True，即验证集不参与训练）
    exclude_validation_from_training = config.get('exclude_validation_from_training', True)
    
    if validation_bearings:
        if exclude_validation_from_training:
            logger.info(f"验证集轴承(用于校准，不参与训练): {validation_bearings}")
        else:
            logger.info(f"验证集轴承(用于校准，同时参与训练): {validation_bearings}")

    # 构建按工况的划分
    # same_dataset=True 表示 train/test 来自同一数据集；否则为跨数据集场景
    same_dataset = (train_datasets_type == test_datasets_type)

    if same_dataset:
        # 同域：经典 2 训 1 测，按工况划分，多轮训练
        splits = build_condition_splits(
            train_bearings_all,
            test_bearings_all,
            context_bearings,
            validation_bearings=validation_bearings,
            same_dataset=True,
            exclude_validation_from_training=exclude_validation_from_training,
        )
        if len(splits) == 0:
            raise ValueError("构建工况划分失败：请检查训练/测试目录与上下文配置是否导致空划分")
    else:
        # 异域：只训练 1 轮，用全部训练集工况（排除上下文，根据配置决定是否排除验证集），
        # 后续在另一个数据集上可以挨个测试
        context_norm = {normalize_name(b) for b in context_bearings}
        validation_norm = {normalize_name(b) for b in validation_bearings} if exclude_validation_from_training else set()
        train_used = [
            b for b in train_bearings_all
            if normalize_name(b) not in context_norm
            and (not exclude_validation_from_training or normalize_name(b) not in validation_norm)
        ]
        if not train_used:
            raise ValueError("跨数据集场景下，排除上下文和验证集后训练集为空，请检查配置")
        splits = [{
            'condition': 'all',
            'train_bearings': train_used,
            # 这里先把全部测试轴承传入，用于创建 DataLoader 和训练阶段的评估；
            # 真正"挨个测试"的细粒度结果可以在训练完成后再调用 test_runner 单独完成。
            'test_bearings': test_bearings_all,
        }]

    logger.info(f"\n{'='*80}")
    logger.info(f"📋 开始训练 总轮数: {len(splits)}")
    logger.info(f"{'='*80}")

    all_results = []
    best_hparams = None
    total_start_time = time.time()

    for loop_idx, split in enumerate(splits):
        test_bearings = split['test_bearings']
        train_bearings = split['train_bearings']

        loop_config = config.copy()
        logger.info(f"训练轴承: {train_bearings}")
        # 训练
        result = train_single_fold(
            loop_idx, train_bearings, test_bearings, context_bearings,
            loop_config, device, train_data_dir, test_data_dir,
            total_folds=len(splits), total_start_time=total_start_time
        )
        res_dir = config['results_dir']+task[0].split('_')[0]+'_to_'+task[1].split('_')[0]+'/'
        scaler_dir = config['scaler_dir']+task[1]+'/'
        logger.info(f"结果保存目录: {res_dir}")
        logger.info(f"Scaler目录: {scaler_dir}")
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        # 测试代码
        # 首先进行第一次测试（未校准）
        logger.info(f"\n{'='*80}")
        logger.info("第一次测试（未校准）")
        logger.info(f"{'='*80}")
        
        first_test_results = {}
        for bearing_name_original in test_bearings:
            # 为每个轴承创建单独的 test_loader
            single_bearing_test_set, single_bearing_test_label = load_bearing_data([bearing_name_original], test_data_dir)
            if len(single_bearing_test_set) == 0:
                logger.warning(f"⚠️  警告: 轴承 {bearing_name_original} 没有数据，跳过测试")
                continue
            
            # 创建单个轴承的 test_loader
            single_bearing_test_loader = Data.DataLoader(
                dataset=Data.TensorDataset(single_bearing_test_set, single_bearing_test_label),
                batch_size=config['test_batch_size'], num_workers=0, drop_last=False,
                pin_memory=True, persistent_workers=False
            )
            
            # 准备用于 scaler 查找的轴承名（不能改后缀，应该截取前缀直接拼接）
            bearing_name_for_scaler = bearing_name_original
            if test_datasets_type == 'xjtu_made':
                bearing_name_for_scaler = bearing_name_original.replace("_labeled", "_labeled_fpt_scaler")
            elif test_datasets_type == 'femto_made':
                bearing_name_for_scaler = bearing_name_original.replace("_labeled", "_labeled_fpt_scaler")
            logger.info(f"测试轴承: {bearing_name_original} (scaler查找名: {bearing_name_for_scaler})")
            model_path = result['model_path']
            model = result['model']
            model.load_state_dict(torch.load(model_path))
            logger.info(f"测试加载的权重文件地址: {model_path}")
            # 使用单个轴承的 test_loader 进行测试
            target, prediction, origin_prediction, log_var_list, mu_samples = run_test_and_save(
                model, 
                single_bearing_test_loader, 
                config['forward_pass'], 
                bearing_name_for_scaler, 
                res_dir, 
                scaler_dir, 
                device
            )
            logger.info(f"target shape: {target.shape}")
            if test_datasets_type == 'xjtu_made':
                bearing_name = bearing_name_for_scaler.replace("_labeled_fpt_scaler", "")
            elif test_datasets_type == 'femto_made':
                bearing_name = bearing_name_for_scaler.replace("_labeled_fpt_scaler", "")
            else:
                # 其他类型（如 xjtu_made_mscrgat、femto_made_mscrgat 等）统一用相同规则
                bearing_name = bearing_name_for_scaler.replace("_labeled_fpt_scaler", "") if "_labeled_fpt_scaler" in bearing_name_for_scaler else bearing_name_original
            
            # 保存第一次测试结果
            evaluate_and_save_metrics(target, prediction, origin_prediction, log_var_list, mu_samples, 
                                     res_dir+bearing_name+'.csv', res_dir+bearing_name+'.png', 0.05)
            save_config(config, res_dir+bearing_name+'.json')
            
            # 保存第一次测试结果用于后续校准
            first_test_results[bearing_name] = {
                'target': target,
                'prediction': prediction,
                'origin_prediction': origin_prediction,
                'log_var_list': log_var_list,
                'mu_samples': mu_samples,
                'test_loader': single_bearing_test_loader
            }
            all_results.append(result)
        
        # 等渗回归校准
        # 使用验证集进行校准（按照 notebook 中的方法）
        # 如果配置了 validation_bearings，则使用验证集；否则回退到训练集的第一个子集
        if validation_bearings and len(validation_bearings) > 0:
            # 是否只使用与测试集同工况的验证集（从配置中读取，默认为True）
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
                test_datasets_type=test_datasets_type,
                first_test_results=first_test_results,
                config=config,
                device=device,
                res_dir=res_dir,
                load_bearing_data_func=load_bearing_data,
                calibration_bearings=validation_bearings,  # 使用验证集
                calibration_mode=config.get('calibration_mode', 'train_first'),  # 使用自定义验证集
                context_bearings=context_bearings,
                use_same_condition_validation=use_same_condition_validation,  # 是否只使用同工况的验证集
                test_data_dir=test_data_dir  # 传递测试数据目录
            )
        else:
            logger.warning("⚠️  警告: 未配置验证集，使用训练集的第一个子集进行校准")
            run_isotonic_calibration(
                model=model,
                train_bearings=train_bearings,
                train_data_dir=train_data_dir,
                scaler_dir=scaler_dir,
                test_bearings=test_bearings,
                test_datasets_type=test_datasets_type,
                first_test_results=first_test_results,
                config=config,
                device=device,
                res_dir=res_dir,
                load_bearing_data_func=load_bearing_data,
                calibration_bearings=config.get('calibration_bearings', None),
                calibration_mode=config.get('calibration_mode', 'train_first'),
                context_bearings=context_bearings,
                test_data_dir=test_data_dir  # 传递测试数据目录
            )

    total_time = time.time() - total_start_time

    logger.info(f"\n{'='*80}")
    logger.info(f"✅ 所有测试循环完成")
    logger.info(f"{'='*80}")
    logger.info(f"总训练时间: {format_time(total_time)}")
    logger.info(f"平均每轮时间: {format_time(total_time / len(splits))}")
    return {
        'all_results': all_results,
        'best_hparams': best_hparams if len(all_results) > 0 else None,
        'config': config,
        'total_time': total_time,
    }


if __name__ == "__main__":
    # 初始化全局logger
    logger = setup_logger()
    logger.info("="*80)
    logger.info("自动训练脚本启动!!!")
    logger.info("="*80)

    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None, help='配置文件路径')
    args, unknown = parser.parse_known_args()
    config_path = args.config_path
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../../config/ablation/A_fbtcn_config_ablation_no_rds_all_data.json')
    if not os.path.exists(config_path):
        raise ValueError(f"配置文件 {config_path} 不存在")
    logger.info(f"使用的配置参数json为：{config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        logger.info(f"配置参数: {config}")

    # tasks = [['femto_made', 'femto_made'], ['xjtu_made', 'xjtu_made'], ['xjtu_made', 'femto_made'], ['femto_made', 'xjtu_made']]
    # tasks = [['xjtu_made', 'xjtu_made']]
    tasks = [['xjtu_made_v3', 'xjtu_made_v3']]

    total_script_start_time = time.time()
    for task_idx, task in enumerate(tasks):
        logger.info(f"\n🚀 训练任务 {task_idx + 1}/{len(tasks)}: {task[0]} -> {task[1]}")
        config['train_datasets_type'] = task[0]
        config['test_datasets_type'] = task[1]
        results = main(config)
        logger.info(f"\n训练任务 {task[0]} -> {task[1]} 完成！结果已返回，可以自行保存。")
    
    total_script_end_time = time.time()
    logger.info("="*80)
    logger.info(f"✅ 所有训练任务完成！总训练时间: {format_time(total_script_end_time - total_script_start_time)}")
    logger.info("="*80)
