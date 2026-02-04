#!/usr/bin/env python3
"""
FBTCN 迁移学习框架（最小实现）

流程：源域预训练 →（可选）加载预训练权重 → 目标域微调（可选冻结 backbone）。
模型使用 fbtcn_sa_model.BayesianTCN，数据格式 [N, seq_len, input_dim] 与 [N]。
"""

import os
import sys
import copy
import torch
import torch.utils.data as Data
import numpy as np
from typing import Dict, Optional, Tuple

# 保证可导入上级模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fbtcn_sa_model import BayesianTCN
from stable_fbtcn_training import model_train_stable, get_stable_optimizer
from loss_function import compute_au_nll_with_crps_and_pos


def default_config() -> Dict:
    """与现有 FBTCN 训练兼容的默认配置（最小集）。"""
    return {
        "input_dim": 1,
        "num_channels": [32, 64, 32],
        "attention_dim": 1,
        "kernel_size": 5,
        "dropout": 0.2,
        "output_dim": 1,
        "conv_posterior_rho_init": -2,
        "output_posterior_rho_init": -2,
        "attention_mode": "self",
        "batch_size": 64,
        "test_batch_size": 256,
        "epochs": 100,
        "finetune_epochs": 50,
        "learn_rate": 0.01,
        "finetune_lr": 0.001,
        "seed": 42,
        "opt": "Adam",
        "kl_weight": 0.01,
        "scheduler": "cosine",
        "patience": 100,
    }


def build_model(config: Dict, device: Optional[torch.device] = None) -> BayesianTCN:
    """根据 config 构建 BayesianTCN。"""
    model = BayesianTCN(
        input_dim=config["input_dim"],
        num_channels=config["num_channels"],
        attention_dim=config["attention_dim"],
        kernel_size=config["kernel_size"],
        dropout=config["dropout"],
        output_dim=config.get("output_dim", 1),
        conv_posterior_rho_init=config.get("conv_posterior_rho_init", -2),
        output_posterior_rho_init=config.get("output_posterior_rho_init", -2),
        attention_mode=config.get("attention_mode", "self"),
    )
    if device is not None:
        model = model.to(device)
    return model


def load_pretrained(model: BayesianTCN, path: str, device: Optional[torch.device] = None) -> BayesianTCN:
    """从 checkpoint 加载权重到 model（严格匹配 state_dict）。"""
    state = torch.load(path, map_location=device or next(model.parameters()).device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model


def freeze_backbone(model: BayesianTCN) -> None:
    """冻结 TCN 与注意力，只保留 mu/sigma 可训练。"""
    for name, p in model.named_parameters():
        if "network" in name or "attention" in name:
            p.requires_grad = False
        else:
            p.requires_grad = True


def unfreeze_all(model: BayesianTCN) -> None:
    """全部参数可训练。"""
    for p in model.parameters():
        p.requires_grad = True


def _make_loaders(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    batch_size: int,
    test_batch_size: int,
    seed: int,
) -> Tuple[Data.DataLoader, Data.DataLoader, Data.DataLoader, Data.DataLoader]:
    """构造 train / context / test / validation 四个 DataLoader。context 与 train 相同（最小实现）。"""
    g = torch.Generator().manual_seed(seed)
    train_ds = Data.TensorDataset(train_x, train_y)
    val_ds = Data.TensorDataset(val_x, val_y)
    train_loader = Data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=g
    )
    context_loader = Data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=g
    )
    val_loader = Data.DataLoader(val_ds, batch_size=test_batch_size, shuffle=False)
    test_loader = Data.DataLoader(val_ds, batch_size=test_batch_size, shuffle=False)
    return train_loader, context_loader, test_loader, val_loader


def pretrain(
    model: BayesianTCN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    config: Dict,
    device: torch.device,
    save_path: Optional[str] = None,
) -> Tuple[BayesianTCN, Dict]:
    """在源域上预训练模型。"""
    train_loader, context_loader, _, val_loader = _make_loaders(
        train_x, train_y, val_x, val_y,
        config["batch_size"], config["test_batch_size"], config["seed"],
    )
    init_model = copy.deepcopy(model)
    optimizer = get_stable_optimizer(model, config)
    epochs = config.get("epochs", 100)
    best_pt = save_path or os.path.join(os.path.dirname(__file__), "transfer_pretrain_best.pt")
    train_losses, val_losses, best_epoch = model_train_stable(
        epochs, model, init_model, optimizer, compute_au_nll_with_crps_and_pos,
        train_loader, context_loader, val_loader, device, config,
        best_pt, skip_validation=True,
    )
    return model, {"train_losses": train_losses, "val_losses": val_losses, "best_epoch": best_epoch}


def finetune(
    model: BayesianTCN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    config: Dict,
    device: torch.device,
    freeze_backbone_layers: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[BayesianTCN, Dict]:
    """在目标域上微调。可选冻结 backbone，仅训练 mu/sigma。"""
    if freeze_backbone_layers:
        freeze_backbone(model)
        lr = config.get("finetune_lr", config["learn_rate"] * 0.1)
    else:
        unfreeze_all(model)
        lr = config.get("finetune_lr", config["learn_rate"])
    # 微调阶段使用单独的学习率
    finetune_config = copy.deepcopy(config)
    finetune_config["learn_rate"] = lr
    finetune_config["epochs"] = config.get("finetune_epochs", 50)
    train_loader, context_loader, _, val_loader = _make_loaders(
        train_x, train_y, val_x, val_y,
        finetune_config["batch_size"], finetune_config["test_batch_size"], finetune_config["seed"],
    )
    init_model = copy.deepcopy(model)
    optimizer = get_stable_optimizer(model, finetune_config)
    best_pt = save_path or os.path.join(os.path.dirname(__file__), "transfer_finetune_best.pt")
    train_losses, val_losses, best_epoch = model_train_stable(
        finetune_config["epochs"], model, init_model, optimizer, compute_au_nll_with_crps_and_pos,
        train_loader, context_loader, val_loader, device, finetune_config,
        best_pt, skip_validation=True,
    )
    return model, {"train_losses": train_losses, "val_losses": val_losses, "best_epoch": best_epoch}


def run_transfer(
    source_train_x: torch.Tensor,
    source_train_y: torch.Tensor,
    source_val_x: torch.Tensor,
    source_val_y: torch.Tensor,
    target_train_x: torch.Tensor,
    target_train_y: torch.Tensor,
    target_val_x: torch.Tensor,
    target_val_y: torch.Tensor,
    config: Optional[Dict] = None,
    pretrained_path: Optional[str] = None,
    freeze_backbone_layers: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[BayesianTCN, Dict]:
    """
    最小迁移流程：若无 pretrained_path 则先在源域预训练，再在目标域微调；
    若有 pretrained_path 则加载后直接在目标域微调。
    freeze_backbone_layers：微调时是否冻结 TCN+注意力，只训练 mu/sigma。
    """
    cfg = config or default_config()
    dev = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg, dev)
    pretrain_info = {}

    if pretrained_path and os.path.isfile(pretrained_path):
        load_pretrained(model, pretrained_path, dev)
        print(f"[Transfer] 已加载预训练: {pretrained_path}")
    else:
        print("[Transfer] 源域预训练...")
        model, pretrain_info = pretrain(
            model, source_train_x, source_train_y, source_val_x, source_val_y, cfg, dev,
        )
        print(f"[Transfer] 预训练 best_epoch: {pretrain_info['best_epoch']}")

    print("[Transfer] 目标域微调...")
    model, finetune_info = finetune(
        model, target_train_x, target_train_y, target_val_x, target_val_y,
        cfg, dev, freeze_backbone_layers=freeze_backbone_layers,
    )
    print(f"[Transfer] 微调 best_epoch: {finetune_info['best_epoch']}")
    return model, {"pretrain": pretrain_info, "finetune": finetune_info}


# ==================== 最小可运行示例 ====================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = default_config()
    config["epochs"] = 1000
    config["finetune_epochs"] = 100
    # 构造伪数据： [N, seq_len, input_dim], [N]
    seq_len, input_dim = 64, config["input_dim"]
    n_source, n_target = 200, 80
    source_x = torch.randn(n_source, seq_len, input_dim, device=device)
    source_y = torch.rand(n_source, device=device)
    source_val_x = torch.randn(50, seq_len, input_dim, device=device)
    source_val_y = torch.rand(50, device=device)
    target_x = torch.randn(n_target, seq_len, input_dim, device=device)
    target_y = torch.rand(n_target, device=device)
    target_val_x = torch.randn(30, seq_len, input_dim, device=device)
    target_val_y = torch.rand(30, device=device)

    model, info = run_transfer(
        source_train_x=source_x, source_train_y=source_y,
        source_val_x=source_val_x, source_val_y=source_val_y,
        target_train_x=target_x, target_train_y=target_y,
        target_val_x=target_val_x, target_val_y=target_val_y,
        config=config, pretrained_path=None, freeze_backbone_layers=True, device=device,
    )
    print("迁移学习最小示例运行完成。")
