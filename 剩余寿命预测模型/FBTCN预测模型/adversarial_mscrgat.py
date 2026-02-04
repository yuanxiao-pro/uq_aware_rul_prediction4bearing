#!/usr/bin/env python3
"""
FBTCN 域对抗学习最小实现（DANN 风格）

特征提取器 F（FBTCN 至 attention 池化后的特征）+ 任务头（mu/sigma）+ 域判别器 D。
通过梯度反转层（GRL）使 F 学习域不变特征，同时最小化源域任务损失。

最大最小博弈（min-max）：
  - D：最小化域分类误差（min_D domain_loss），即尽量区分源/目标域；
  - F：最小化任务损失并最大化域混淆（min_F task_loss - λ·domain_loss），即骗过 D。
等价于 min_F max_D [ task_loss(F) - λ·domain_loss(F,D) ]。默认实现为「同时更新」
（一次 backward 后 F、D 一起 step）；可选 alternating=True 做「交替」min-max 更新。
数据：源域 (X_s, y_s)，目标域 X_t（无标签）；格式 [N, seq_len, input_dim]。
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fbtcn_sa_model import BayesianTCN
from loss_function import compute_smooth_l1_task_loss, compute_mmd_loss
from joblib import load as joblib_load
from auto_train_fbtcn_sa import get_data_dir, load_bearing_data, list_bearings_in_dir
from metrics import mae, rmse, picp, nmpiw, cwc, ece, sharpness
from scipy.stats import norm
try:
    from MSCRGAT import MSCRGAT
except ImportError:
    MSCRGAT = None


def gaussian_nll_loss(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """高斯负对数似然：-log p(y|μ,σ) = log(σ) + (y-μ)²/(2σ²)。用于异方差回归（偶然不确定性）。"""
    sigma = sigma.clamp(min=eps)
    return (torch.log(sigma) + (y - mu) ** 2 / (2 * sigma ** 2)).mean()


class GradientReversalFn(torch.autograd.Function):
    """前向不变，反向时梯度乘以 -lambda（用于域对抗）。"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFn.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_

class DomainDiscriminator(nn.Module):
    """域判别器：MLP，特征 -> 2 个 logit（类别 0=源域，1=目标域），全连接层 64, 64, 32 -> 2，使用 CrossEntropyLoss。"""
    def __init__(self, feature_dim: int, hidden_dims: tuple = (64, 64, 32), num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_f = feature_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_f, h),
                nn.ReLU(inplace=False),
                nn.Dropout(dropout),
            ])
            in_f = h
        layers.append(nn.Linear(in_f, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def default_config() -> Dict:
    return {
        "input_dim": 1,
        "num_channels": [32, 64, 32],
        "attention_dim": 32,
        "kernel_size": 5,
        "dropout": 0.5,
        "output_dim": 1,
        "conv_posterior_rho_init": -2,
        "output_posterior_rho_init": -2,
        "attention_mode": "self",
        "batch_size": 1024,
        "epochs": 1000,
        "learn_rate": 0.0032,          # 略提高，便于 task 损失下降3e-2
        "discriminator_lr": 1e-3,
        "domain_loss_weight": 0.01,   # 降低，让 task 损失主导，避免域对齐压制 RUL 学习
        "grl_lambda": 1.0,
        "mmd_loss_weight": 0.01,
        "mmd_gamma": 1.0,  # 式 (19) 高斯核带宽 γ
        "alternating": True,
        "grl_schedule": True,  # True：λ 从 0 线性增至 grl_lambda，先巩固任务再加强域对齐
        "task_warmup_epochs": 80,  # 前 N 个 epoch 仅优化 task 损失，再开启域对抗/MMD，便于 task 先下降
        "decay_period": 32,
        "decay_rate": 0.8,    # MSCRGAT：衰减率（每 decay_period 个 epoch lr *= decay_rate）
        "min_lr": 1e-5,       # MSCRGAT：学习率下限，不再低于此值
        "discriminator_hidden": (64, 64, 32),  # 域判别器：3 个全连接层
        "discriminator_output": 2,              # 域判别器：输出层维度
        "mscrgat_feature_dim": 128,             # MSCRGAT 拼接特征维度（64+64），域判别器输入
        "heteroscedastic": False,               # True 时预测头输出 (μ, σ)，用高斯 NLL 训练，输出偶然不确定性
        "seed": 0,
    }


def load_adversarial_config(config_path: str) -> Dict:
    """
    从 JSON 文件加载超参，与 default_config() 合并，缺失项用默认值。
    JSON 中数组（如 discriminator_hidden）在代码中会转为 tuple 使用。
    """
    base = default_config()
    if not os.path.isfile(config_path):
        return base
    with open(config_path, "r", encoding="utf-8") as f:
        from_file = json.load(f)
    base.update(from_file)
    return base


def build_fbtcn(config: Dict, device: Optional[torch.device] = None) -> BayesianTCN:
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


def build_mscrgat(config: Dict, device: Optional[torch.device] = None):
    """
    从 config 构建 MSCRGAT（Multi-Scale CNN + Res_GRU + 3 层全连接 + Sigmoid）。
    config 需含 input_dim；可选 predictor_hidden，默认 (64, 32)；heteroscedastic=False 时仅输出 μ。
    """
    if MSCRGAT is None:
        raise ImportError("MSCRGAT 未安装：请确保 MSCRGAT.py 与 adversarial_fbtcn.py 同目录。")
    model = MSCRGAT(
        input_dim=config["input_dim"],
        predictor_hidden=tuple(config.get("predictor_hidden", (64, 32))),
        dropout=config.get("dropout", 0.5),
        heteroscedastic=config.get("heteroscedastic", False),
    )
    if device is not None:
        model = model.to(device)
    return model


def train_adversarial(
    model: BayesianTCN,
    discriminator: DomainDiscriminator,
    grl: GradientReversalLayer,
    source_loader: Data.DataLoader,
    target_loader: Data.DataLoader,
    config: Dict,
    device: torch.device,
    alternating: bool = False,
) -> Tuple[list, list, list]:
    """
    域对抗训练。整体损失：L_total = L_smooth_l1 + λ_d·L_domain + λ_m·L_mmd。
    source_loader 产出 (X, y)，target_loader 产出 (X,) 或 (X, _)。
    alternating=True 时做显式 min-max 交替更新：先 max_D（更新 D 最小化 domain_loss），
    再 min_F（更新 F 最小化 task_loss - λ·domain_loss + λ_m·L_mmd）；否则为同时更新（一次 backward）。
    """
    model.train()
    discriminator.train()
    opt_f = torch.optim.Adam(model.parameters(), lr=config["learn_rate"], weight_decay=1e-5)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config["discriminator_lr"], weight_decay=1e-5)
    domain_weight = config.get("domain_loss_weight", 0.1)
    mmd_weight = config.get("mmd_loss_weight", 0.1)
    mmd_gamma = config.get("mmd_gamma", 1.0)
    grl.set_lambda(config.get("grl_lambda", 1.0))

    # 原域损失：CrossEntropyLoss（二分类：0=源域，1=目标域）
    # domain_criterion = nn.CrossEntropyLoss()
    task_losses, domain_losses, mmd_losses = [], [], []

    target_iter = iter(target_loader)
    for batch_idx, (s_x, s_y) in enumerate(source_loader):
        s_x, s_y = s_x.to(device), s_y.to(device)
        try:
            t_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            t_batch = next(target_iter)
        t_x = t_batch[0].to(device)

        # 源域前向：任务损失（Smooth L1）
        mu, sigma, kl, feat_s = model(s_x, feature=True)
        task_loss = compute_smooth_l1_task_loss(s_y, mu)
        # 目标域前向：取特征（需保留梯度，供域损失经 GRL 反传）
        _, _, _, feat_t = model(t_x, feature=True)

        # 域判别：拼接特征，域标签 y_i：0=源 1=目标（long）
        feat = torch.cat([feat_s, feat_t], dim=0)
        domain_labels = torch.cat([
            torch.zeros(feat_s.size(0), dtype=torch.long, device=device),
            torch.ones(feat_t.size(0), dtype=torch.long, device=device),
        ], dim=0)
        rev_feat = grl(feat)
        domain_logits = discriminator(rev_feat)
        # 式 (17)：L_domain = -(1/N)*sum[ y_i*log(y_hat_i) + (1-y_i)*log(1-y_hat_i) ]，二分类交叉熵
        y_hat = torch.softmax(domain_logits, dim=1)[:, 1].clamp(1e-7, 1.0 - 1e-7)
        y_true = domain_labels.float()
        domain_loss = F.binary_cross_entropy(y_hat, y_true, reduction="mean")
        # domain_loss = domain_criterion(domain_logits, domain_labels)  # 原 CrossEntropyLoss

        # MMD 损失：式 (18) + 式 (19)，源/目标特征分布对齐（用 GRL 前的 feat_s, feat_t）
        mmd_loss = compute_mmd_loss(feat_s, feat_t, gamma=mmd_gamma)

        # 整体损失：L_total = L_smooth_l1 + λ_d·L_domain + λ_m·L_mmd
        # min-max 博弈：F 最小化 task_loss 并最大化域混淆，D 最小化 domain_loss。
        # total = L_smooth_l1 + λ_d·L_domain + λ_m·L_mmd，经 GRL 反传后 F 实际收到 L_smooth_l1 - λ_d·L_domain + λ_m·L_mmd 的梯度。
        if alternating:
            # 显式 min-max 交替：1) 更新 D 最小化 domain_loss（max_D 等价于 min_D -domain_loss）
            opt_d.zero_grad()
            domain_loss.backward(retain_graph=True)
            opt_d.step()
            # 2) 更新 F：最小化 task_loss - λ·domain_loss + λ_mmd·L_MMD（GRL 已使 F 收到 -λ·d(domain_loss)/dF）
            opt_f.zero_grad()
            (task_loss - domain_weight * domain_loss + mmd_weight * mmd_loss).backward()
            opt_f.step()
        else:
            # 同时更新（一次 backward）：F 收到 task_loss + λ·(-domain_loss)|_GRL + λ_mmd·L_MMD，D 收到 λ·domain_loss
            total = task_loss + domain_weight * domain_loss + mmd_weight * mmd_loss
            opt_f.zero_grad()
            opt_d.zero_grad()
            total.backward()
            opt_f.step()
            opt_d.step()

        task_losses.append(task_loss.item())
        domain_losses.append(domain_loss.item())
        mmd_losses.append(mmd_loss.item())
    return task_losses, domain_losses, mmd_losses


def train_adversarial_mscrgat(
    model,
    discriminator: DomainDiscriminator,
    grl: GradientReversalLayer,
    source_loader: Data.DataLoader,
    target_loader: Data.DataLoader,
    config: Dict,
    device: torch.device,
    alternating: bool = False,
) -> Tuple[list, list, list]:
    """
    MSCRGAT 域对抗训练。整体损失 L_total = L_smooth_l1 + λ_d·L_domain + λ_m·L_mmd。
    model 为 MSCRGAT，forward(·, feature=True) 返回 (rul, feat)，feat 供 D 与 MMD 使用。
    """
    model.train()
    discriminator.train()
    heteroscedastic = getattr(model, "heteroscedastic", False)
    opt_f = torch.optim.Adam(model.parameters(), lr=config["learn_rate"], weight_decay=1e-5)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config["discriminator_lr"], weight_decay=1e-5)
    domain_weight = config.get("domain_loss_weight", 0.1)
    mmd_weight = config.get("mmd_loss_weight", 0.1)
    mmd_gamma = config.get("mmd_gamma", 1.0)
    grl.set_lambda(config.get("grl_lambda", 1.0))
    task_losses, domain_losses, mmd_losses, au_losses = [], [], [], []
    target_iter = iter(target_loader)
    for batch_idx, (s_x, s_y) in enumerate(source_loader):
        s_x, s_y = s_x.to(device), s_y.to(device)
        if torch.isnan(s_x).any() or torch.isinf(s_x).any():
            s_x = torch.nan_to_num(s_x, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(s_y).any() or torch.isinf(s_y).any():
            s_y = torch.nan_to_num(s_y, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            t_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            t_batch = next(target_iter)
        t_x = t_batch[0].to(device)
        if torch.isnan(t_x).any() or torch.isinf(t_x).any():
            t_x = torch.nan_to_num(t_x, nan=0.0, posinf=0.0, neginf=0.0)

        sy_1d = s_y.squeeze(-1)
        if heteroscedastic:
            rul_s, sigma_s, feat_s = model(s_x, feature=True)
            task_loss = gaussian_nll_loss(sy_1d, rul_s, sigma_s)
            _, _, feat_t = model(t_x, feature=True)
        else:
            rul_s, feat_s = model(s_x, feature=True)
            task_loss = compute_smooth_l1_task_loss(sy_1d, rul_s)
            _, feat_t = model(t_x, feature=True)

        feat = torch.cat([feat_s, feat_t], dim=0)
        domain_labels = torch.cat([
            torch.zeros(feat_s.size(0), dtype=torch.long, device=device),
            torch.ones(feat_t.size(0), dtype=torch.long, device=device),
        ], dim=0)
        rev_feat = grl(feat)
        domain_logits = discriminator(rev_feat)
        # 式 (17)：L_domain = -(1/N)*sum[ y_i*log(y_hat_i) + (1-y_i)*log(1-y_hat_i) ]，二分类交叉熵
        y_hat = torch.softmax(domain_logits, dim=1)[:, 1].clamp(1e-7, 1.0 - 1e-7)  # 目标域概率
        y_true = domain_labels.float()
        domain_loss = F.binary_cross_entropy(y_hat, y_true, reduction="mean")
        # domain_loss = domain_criterion(domain_logits, domain_labels)  # 原 CrossEntropyLoss
        mmd_loss = compute_mmd_loss(feat_s, feat_t, gamma=mmd_gamma)

        if alternating:
            opt_d.zero_grad()
            domain_loss.backward(retain_graph=True)
            opt_d.step()
            opt_f.zero_grad()
            (task_loss - domain_weight * domain_loss + mmd_weight * mmd_loss).backward()
            opt_f.step()
        else:
            total = task_loss + domain_weight * domain_loss + mmd_weight * mmd_loss
            opt_f.zero_grad()
            opt_d.zero_grad()
            total.backward()
            # 梯度裁剪，防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            opt_f.step()
            opt_d.step()

        # NaN 检测（首次出现时打印调试信息）
        if torch.isnan(task_loss) or torch.isnan(domain_loss) or torch.isnan(mmd_loss):
            if batch_idx == 0:
                print(f"[DEBUG] NaN detected at batch {batch_idx}: task={task_loss.item()}, domain={domain_loss.item()}, mmd={mmd_loss.item()}")
                print(f"        rul_s range: [{rul_s.min().item():.4f}, {rul_s.max().item():.4f}], NaN: {torch.isnan(rul_s).any().item()}")
                print(f"        feat_s range: [{feat_s.min().item():.4f}, {feat_s.max().item():.4f}], NaN: {torch.isnan(feat_s).any().item()}")
                print(f"        feat_t range: [{feat_t.min().item():.4f}, {feat_t.max().item():.4f}], NaN: {torch.isnan(feat_t).any().item()}")
        task_losses.append(task_loss.item())
        domain_losses.append(domain_loss.item())
        mmd_losses.append(mmd_loss.item())
        if heteroscedastic:
            au_losses.append(sigma_s.pow(2).mean().item())
    return task_losses, domain_losses, mmd_losses, au_losses


def run_adversarial_mscrgat(
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    target_x: torch.Tensor,
    config: Optional[Dict] = None,
    device: Optional[torch.device] = None,
    best_model_path: Optional[str] = None,
):
    """
    MSCRGAT（Res-GRU + MSCNN + 预测器）+ 域判别器，整体损失 = Smooth L1 + domain loss + MMD。
    使用 run_adversarial 风格的域对抗训练；RUL 为原始尺度，不做归一化。
    """
    if MSCRGAT is None:
        raise ImportError("MSCRGAT 未安装：请确保 MSCRGAT.py 与 adversarial_fbtcn.py 同目录。")
    cfg = config or default_config()
    print(cfg)
    cfg["input_dim"] = cfg.get("input_dim", 11)
    dev = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    model = build_mscrgat(cfg, dev)
    feat_dim = cfg.get("mscrgat_feature_dim", 128)
    discriminator = DomainDiscriminator(
        feature_dim=feat_dim,
        hidden_dims=tuple(cfg.get("discriminator_hidden", (64, 64, 32))),
        num_classes=cfg.get("discriminator_output", 2),
        dropout=cfg["dropout"],
    ).to(dev)
    grl = GradientReversalLayer(lambda_=cfg.get("grl_lambda", 1.0))

    # 训练前打印模型结构
    # print("\n--- MSCRGAT 模型结构 ---")
    # print(model)
    # n_params = sum(p.numel() for p in model.parameters())
    # print(f"MSCRGAT 参数量: {n_params:,}\n")
    # print("--- 域判别器结构 ---")
    # print(discriminator)
    # n_disc = sum(p.numel() for p in discriminator.parameters())
    # print(f"域判别器参数量: {n_disc:,}\n")

    src_ds = Data.TensorDataset(source_x, source_y)
    tgt_ds = Data.TensorDataset(target_x, torch.zeros(target_x.size(0)))
    bs = min(cfg["batch_size"], len(src_ds), len(tgt_ds))
    bs = max(bs, 1)
    use_cuda = dev.type == "cuda"
    num_workers = cfg.get("num_workers", 4 if use_cuda else 0)
    if use_cuda and cfg.get("use_benchmark", False):
        torch.backends.cudnn.benchmark = True
    loader_kw = dict(batch_size=bs, shuffle=True, drop_last=True, pin_memory=use_cuda, num_workers=num_workers)
    if num_workers > 0:
        loader_kw["persistent_workers"] = True
    source_loader = Data.DataLoader(src_ds, **loader_kw)
    target_loader = Data.DataLoader(tgt_ds, **loader_kw)
    if len(source_loader) == 0 or len(target_loader) == 0:
        print(f"[WARNING] DataLoader empty: source={len(src_ds)}, target={len(tgt_ds)}, batch_size={bs}. Using drop_last=False.")
        loader_kw["drop_last"] = False
        source_loader = Data.DataLoader(src_ds, **loader_kw)
        target_loader = Data.DataLoader(tgt_ds, **loader_kw)

    epochs = cfg.get("epochs", 50)
    alternating = False
    grl_schedule = cfg.get("grl_schedule", True)   # 默认 True：前期小 λ、小 domain 权重，先学好 task
    grl_lambda_max = cfg.get("grl_lambda", 1.0)
    domain_weight_max = cfg.get("domain_loss_weight", 0.01)
    mmd_weight_max = cfg.get("mmd_loss_weight", 0.01)
    task_warmup_epochs = cfg.get("task_warmup_epochs", 0)  # 前 N 个 epoch 仅 task 损失，再开域对齐
    all_task, all_domain, all_mmd = [], [], []
    best_task_loss = float("inf")
    # 说明：domain_loss 不下降且稳定在 ~0.693 是预期现象。二分类 CrossEntropy 随机猜测时 = ln(2)≈0.693，
    # 即判别器无法区分源/目标域，特征已域不变，域对齐成功。若 domain_loss 持续下降则说明判别器占优、域未对齐。
    # print("训练开始（domain_loss 稳定在 ~0.693 表示域判别器处于随机猜测，即域对齐成功，属正常）\n")
    for ep in range(epochs):
        if task_warmup_epochs > 0 and ep < task_warmup_epochs:
            cfg_ep = {**cfg, "grl_lambda": 0.0, "domain_loss_weight": 0.0, "mmd_loss_weight": 0.0}
        elif grl_schedule:
            p = (ep - task_warmup_epochs + 1) / max(1, epochs - task_warmup_epochs)
            p = min(1.0, p)
            cfg_ep = {
                **cfg,
                "grl_lambda": p * grl_lambda_max,
                "domain_loss_weight": p * domain_weight_max,
                "mmd_loss_weight": p * mmd_weight_max,
            }
        else:
            cfg_ep = cfg
        tl, dl, ml, au_list = train_adversarial_mscrgat(
            model, discriminator, grl, source_loader, target_loader, cfg_ep, dev, alternating=alternating,
        )
        all_task.extend(tl)
        all_domain.extend(dl)
        all_mmd.extend(ml)
        mean_task_loss = float(np.mean(tl))
        if best_model_path and mean_task_loss < best_task_loss:
            best_task_loss = mean_task_loss
            torch.save(model.state_dict(), best_model_path)
        msg = f"MSCRGAT 对抗 Epoch {ep+1}/{epochs} | task: {np.mean(tl):.4f} | domain: {np.mean(dl):.4f} | mmd: {np.mean(ml):.4f}"
        if au_list:
            msg += f" | AU_mean: {np.mean(au_list):.6f}"
        print(msg)

    return model, discriminator, {
        "task_losses": all_task,
        "domain_losses": all_domain,
        "mmd_losses": all_mmd,
        "y_scale": 1.0,
        "y_offset": 0.0,  # 由 __main__ 在训练前对 source_y 平移后写入，评估时 pred_orig = pred + y_offset
    }


def run_adversarial(
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    target_x: torch.Tensor,
    config: Optional[Dict] = None,
    device: Optional[torch.device] = None,
) -> Tuple[BayesianTCN, DomainDiscriminator, Dict]:
    """
    最小流程：构建 F + D + GRL，用源域与目标域数据做域对抗训练。
    source_x [N_s, seq_len, dim], source_y [N_s], target_x [N_t, seq_len, dim]。
    """
    cfg = config or default_config()
    dev = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    model = build_fbtcn(cfg, dev)
    discriminator = DomainDiscriminator(
        feature_dim=cfg["attention_dim"],
        hidden_dims=tuple(cfg.get("discriminator_hidden", (64, 64, 32))),
        num_classes=cfg.get("discriminator_output", 2),
        dropout=cfg["dropout"],
    ).to(dev)
    grl = GradientReversalLayer(lambda_=cfg.get("grl_lambda", 1.0))

    src_ds = Data.TensorDataset(source_x, source_y)
    tgt_ds = Data.TensorDataset(target_x, torch.zeros(target_x.size(0)))  # 占位
    source_loader = Data.DataLoader(src_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    target_loader = Data.DataLoader(tgt_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    epochs = cfg.get("epochs", 50)
    alternating = cfg.get("alternating", False)
    grl_schedule = cfg.get("grl_schedule", False)
    grl_lambda_max = cfg.get("grl_lambda", 1.0)
    domain_weight_max = cfg.get("domain_loss_weight", 0.1)
    all_task, all_domain, all_mmd = [], [], []
    for ep in range(epochs):
        if grl_schedule:
            # λ 从 0 线性增至最大值，前期以任务为主、后期再加强域对齐，减轻 task_loss 被破坏
            p = (ep + 1) / float(epochs)
            cfg = {**cfg, "grl_lambda": p * grl_lambda_max, "domain_loss_weight": p * domain_weight_max}
        tl, dl, ml = train_adversarial(
            model, discriminator, grl, source_loader, target_loader, cfg, dev,
            alternating=alternating,
        )
        all_task.extend(tl)
        all_domain.extend(dl)
        all_mmd.extend(ml)
        if epochs <= 10 or (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep+1}/{epochs} | task_loss: {np.mean(tl):.4f} | domain_loss: {np.mean(dl):.4f} | mmd_loss: {np.mean(ml):.4f}")

    return model, discriminator, {"task_losses": all_task, "domain_losses": all_domain, "mmd_losses": all_mmd}


def run_mscrgat(
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    config: Optional[Dict] = None,
    device: Optional[torch.device] = None,
):
    """
    使用 MSCRGAT 在源域上做监督训练。输入维度由 config["input_dim"] 指定（默认 11）。
    标签归一化到 [0,1] 后使用 BCE 损失；返回 model 与 info（含 loss 列表与 y_scale 用于反归一化）。
    """
    if MSCRGAT is None:
        raise ImportError("MSCRGAT 未安装：请确保 MSCRGAT.py 与 adversarial_fbtcn.py 同目录。")
    cfg = config or default_config()
    cfg["input_dim"] = cfg.get("input_dim", 11)
    dev = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    model = build_mscrgat(cfg, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("learn_rate", 1e-2), weight_decay=1e-5)
    decay_period = cfg.get("decay_period", 200)
    decay_rate = cfg.get("decay_rate", 0.8)
    min_lr = cfg.get("min_lr", 1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_period, gamma=decay_rate)
    bce = nn.BCELoss()
    heteroscedastic = getattr(model, "heteroscedastic", False)

    y_scale = float(source_y.max()) + 1e-8
    source_y_norm = (source_y / y_scale).float().clamp(0.0, 1.0)
    src_ds = Data.TensorDataset(source_x, source_y_norm)
    loader = Data.DataLoader(
        src_ds,
        batch_size=cfg.get("batch_size", 64),
        shuffle=True,
        drop_last=True,
    )
    epochs = cfg.get("epochs", 50)
    all_loss = []
    for ep in range(epochs):
        model.train()
        ep_loss = []
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            opt.zero_grad()
            out = model(bx)
            by_1d = by.squeeze(-1)
            if heteroscedastic:
                mu, sigma = out[0], out[1]
                loss = gaussian_nll_loss(by_1d, mu, sigma)
            else:
                loss = bce(out, by_1d)
            loss.backward()
            opt.step()
            ep_loss.append(loss.item())
        all_loss.extend(ep_loss)
        scheduler.step()
        for g in opt.param_groups:
            g["lr"] = max(g["lr"], min_lr)
        if epochs <= 10 or (ep + 1) % 10 == 0 or ep == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"MSCRGAT Epoch {ep+1}/{epochs} | loss: {np.mean(ep_loss):.4f} | lr: {lr:.2e}")
    return model, {"losses": all_loss, "y_scale": y_scale}


def evaluate_mscrgat_model(
    model,
    x: torch.Tensor,
    y_true: torch.Tensor,
    device: torch.device,
    y_scale: float = 1.0,
    y_offset: float = 0.0,
    batch_size: int = 64,
) -> Dict[str, float]:
    """MSCRGAT 点估计，计算 MAE、RMSE。heteroscedastic 时取 μ 为预测值。预测值 = pred * y_scale + y_offset。"""
    model.eval()
    heteroscedastic = getattr(model, "heteroscedastic", False)
    ds = Data.TensorDataset(x, y_true)
    loader = Data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    pred_list, y_list = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            out = model(bx)
            pred = (out[0] if heteroscedastic else out).cpu().numpy().ravel()
            pred_list.append(pred * y_scale + y_offset)
            y_list.append(by.numpy().ravel())
    y_true_np = np.concatenate(y_list, axis=0)
    pred_np = np.concatenate(pred_list, axis=0)
    return {"MAE": mae(y_true_np, pred_np), "RMSE": rmse(y_true_np, pred_np)}


def evaluate_adversarial_model(
    model: BayesianTCN,
    x: torch.Tensor,
    y_true: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    使用 剩余寿命预测模型/metrics.py 计算各项指标。
    在源域或任意有标签数据上评估：MAE, RMSE, PICP, NMPIW, CWC, ECE, Sharpness。
    """
    model.eval()
    ds = Data.TensorDataset(x, y_true)
    loader = Data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    mu_list, sigma_list, y_list = [], [], []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            mu, sigma, _ = model(bx, feature=False)
            mu_list.append(mu.cpu().numpy().ravel())
            sigma_list.append(sigma.cpu().numpy().ravel())
            y_list.append(by.cpu().numpy().ravel())
    y_true_np = np.concatenate(y_list, axis=0)
    mu_np = np.concatenate(mu_list, axis=0)
    sigma_np = np.concatenate(sigma_list, axis=0)
    var_np = np.maximum(sigma_np ** 2, 1e-12)

    z = norm.ppf(1.0 - alpha / 2.0)
    y_lower = mu_np - z * sigma_np
    y_upper = mu_np + z * sigma_np
    R = float(np.ptp(y_true_np))
    if R < 1e-8:
        R = 1.0

    metrics_dict = {
        "MAE": mae(y_true_np, mu_np),
        "RMSE": rmse(y_true_np, mu_np),
        "PICP": picp(y_true_np, y_lower, y_upper),
        "NMPIW": nmpiw(y_lower, y_upper, R),
        "CWC": cwc(picp(y_true_np, y_lower, y_upper), nmpiw(y_lower, y_upper, R), alpha=alpha),
        "ECE": ece(y_true_np, mu_np, var_np),
        "Sharpness": sharpness(sigma_np, alpha=alpha),
    }
    return metrics_dict


# 滑窗长度，设为 1（得到 [N, 1, input_dim] 的时序输入）
WINDOW_SIZE = 1


def _sliding_windows(data: torch.Tensor, labels: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对单轴承时序数据做滑窗。参考 notebook 中的滑窗思路。
    data: (T, 1, F) 或 (T, F)；labels: (T,) 或 (T, 1)。
    返回 X: (N, window_size, F), y: (N,)，N = T - window_size + 1；每个窗口的标签取窗口末时刻的 RUL。
    若 T < window_size：末尾填充至 window_size（重复最后一帧），返回 1 个窗口，标签取末时刻（避免测试集如工况3轴承3 T=30 时滑窗为 0）。
    """
    if data.dim() == 3:
        data = data.squeeze(1)
    if labels.dim() > 1:
        labels = labels.squeeze(-1)
    T, F = data.shape
    if T < window_size:
        # 末尾填充：重复最后一帧，得到 1 个窗口，标签取末时刻
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
    bearing_list: list,
    data_dir: str,
    window_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按轴承加载 *_fpt_data / *_fpt_label，对每个文件做滑窗后拼接。返回 (X, y)，X 形状 [N, window_size, F]。"""
    x_list, y_list = [], []
    for bearing in bearing_list:
        data_files = [f for f in os.listdir(data_dir) if bearing in f and (f.endswith("_fpt_data") or f.endswith("_data"))]
        label_files = [f for f in os.listdir(data_dir) if bearing in f and (f.endswith("_fpt_label") or f.endswith("_label"))]
        data_files.sort()
        label_files.sort()
        for data_file, label_file in zip(data_files, label_files):
            data = joblib_load(os.path.join(data_dir, data_file))
            label = joblib_load(os.path.join(data_dir, label_file))
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.float32)
            X_w, y_w = _sliding_windows(data, label, window_size)
            if X_w.size(0) > 0:
                x_list.append(X_w)
                y_list.append(y_w)
    if not x_list:
        return torch.empty(0), torch.empty(0)
    return torch.cat(x_list, dim=0), torch.cat(y_list, dim=0)


def _load_xjtu_xjtu_for_adversarial(device: torch.device, window_size: int = WINDOW_SIZE):
    """源域：XJTU 工况1 + 工况2；目标域：XJTU 工况3 的轴承1和2。滑窗后形状 [N, window_size, 13]。
    使用 MSCRGAT 特征集（13 个特征）。xjtu_made_mscrgat 目录下文件名为 Bearing*_mscrgat_labeled。
    """
    data_type = "xjtu_made_mscrgat"
    data_dir = get_data_dir(data_type)
    if not os.path.isdir(data_dir):
        return None, None, None, None
    files = os.listdir(data_dir)
    # xjtu_made_mscrgat 命名：Bearing1_1_mscrgat_labeled, Bearing2_1_mscrgat_labeled, Bearing3_1_mscrgat_labeled 等
    src_bearings = [
        "Bearing3_1_mscrgat_labeled", "Bearing3_2_mscrgat_labeled", "Bearing3_3_mscrgat_labeled", "Bearing3_4_mscrgat_labeled", "Bearing3_5_mscrgat_labeled",
        "Bearing2_1_mscrgat_labeled", "Bearing2_2_mscrgat_labeled", "Bearing2_3_mscrgat_labeled", "Bearing2_4_mscrgat_labeled", "Bearing2_5_mscrgat_labeled",
    ]
    tgt_bearings = ["Bearing1_1_mscrgat_labeled", "Bearing1_2_mscrgat_labeled"]
    src_bearings = [b for b in src_bearings if any(b in f for f in files)]
    tgt_bearings = [b for b in tgt_bearings if any(b in f for f in files)]
    if not src_bearings or not tgt_bearings:
        return None, None, None, None
    source_x, source_y = _load_bearing_data_with_windows(src_bearings, data_dir, window_size)
    target_x, _ = _load_bearing_data_with_windows(tgt_bearings, data_dir, window_size)
    if source_x.numel() == 0 or target_x.numel() == 0:
        return None, None, None, None
    source_x = source_x.float()
    source_y = source_y.float()
    target_x = target_x.float()
    return source_x, source_y, target_x, source_x.shape[2]


def _load_xjtu_test_set(device: torch.device, window_size: int = WINDOW_SIZE):
    """测试集：XJTU 工况3 轴承3（Bearing3_3），数据来自 xjtu_made_mscrgat。滑窗后返回 (test_x, test_y)，形状 [N, window_size, 13] 与 [N]。"""
    data_type = "xjtu_made_mscrgat"
    data_dir = get_data_dir(data_type)
    if not os.path.isdir(data_dir):
        return None, None
    # 测试集用 Bearing3_3（工况3 轴承3），与目标域 3_1/3_4 不重叠，便于评估泛化
    test_bearings = ["Bearing1_3_mscrgat_labeled"]
    test_bearings = [b for b in test_bearings if any(b in f for f in os.listdir(data_dir))]
    if not test_bearings:
        return None, None
    test_x, test_y = _load_bearing_data_with_windows(test_bearings, data_dir, window_size)
    if test_x.numel() == 0:
        return None, None
    test_x = test_x.float()
    test_y = test_y.float()
    return test_x, test_y


def _load_xjtu_femto_for_adversarial(device: torch.device):
    """源域：xjtu_made，目标域：femto_made。若目录或数据不存在则返回 None，调用方用随机数据兜底。"""
    src_type, tgt_type = "xjtu_made", "femto_made"
    src_dir = get_data_dir(src_type)
    tgt_dir = get_data_dir(tgt_type)
    if not os.path.isdir(src_dir) or not os.path.isdir(tgt_dir):
        return None, None, None, None
    src_bearings = list_bearings_in_dir(src_dir)
    tgt_bearings = list_bearings_in_dir(tgt_dir)
    if not src_bearings or not tgt_bearings:
        return None, None, None, None
    source_x, source_y = load_bearing_data(src_bearings, src_dir)
    target_x, _ = load_bearing_data(tgt_bearings, tgt_dir)
    if source_x.numel() == 0 or target_x.numel() == 0:
        return None, None, None, None
    # 统一为 [N, seq_len, input_dim]，保证在 CPU 上以便 DataLoader
    source_x = source_x.float()
    source_y = source_y.float()
    target_x = target_x.float()
    if source_x.dim() == 2:
        source_x = source_x.unsqueeze(-1)
    if target_x.dim() == 2:
        target_x = target_x.unsqueeze(-1)
    return source_x, source_y, target_x, source_x.shape[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSCRGAT 域对抗训练")
    default_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../config/adversarial_mscrgat_config.json"
    )
    parser.add_argument("--config_path", type=str, default=default_config_path, help="超参 JSON 路径")
    args = parser.parse_args()
    cfg = load_adversarial_config(args.config_path)
    if not os.path.isfile(args.config_path):
        print(f"[WARNING] 未找到配置文件 {args.config_path}，使用 default_config()。")
    else:
        print(f"已加载超参: {args.config_path}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 源域：工况1+工况2；目标域：工况3 轴承1和2；测试集：工况3 轴承3
    loaded = _load_xjtu_xjtu_for_adversarial(dev)
    source_x, source_y, target_x, data_dim = loaded[0], loaded[1], loaded[2], loaded[3]
    test_x, test_y = _load_xjtu_test_set(dev)
    y_offset = 0.0  # 使用 [0,1] 归一化后固定为 0
    if source_x is not None and target_x is not None:
        # xjtu_made_mscrgat 数据集有 13 个特征维度，与 config["input_dim"]=13 一致，不需要截取
        actual_dim = source_x.shape[2]
        if actual_dim != cfg.get("input_dim", 13):
            print(f"  警告：数据实际维度 {actual_dim} 与配置 input_dim {cfg.get('input_dim', 13)} 不一致，更新配置为 {actual_dim}")
            cfg["input_dim"] = actual_dim
        # 数据安全：替换 NaN/Inf
        source_x = torch.nan_to_num(source_x, nan=0.0, posinf=0.0, neginf=0.0)
        source_y = torch.nan_to_num(source_y, nan=0.0, posinf=0.0, neginf=0.0)
        target_x = torch.nan_to_num(target_x, nan=0.0, posinf=0.0, neginf=0.0)
        # 特征统一标准化：仅在源域 fit，对 target/test 做 transform，使输入尺度一致（勿用变量名 F，避免遮蔽 torch.nn.functional）
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
        print(f"  特征已用源域 StandardScaler 统一标准化（源/目标/测试同尺度）")
        # 源域 RUL 归一化到 [0,1]，与 ReLU 输出尺度一致，评估时测试集也归一化到 [0,1] 再比较
        sy_min, sy_max = source_y.min().item(), source_y.max().item()
        span_s = sy_max - sy_min
        if span_s < 1e-8:
            span_s = 1.0
        source_y = ((source_y - sy_min) / span_s).float().clamp(0.0, 1.0)
        print(f"  源域 RUL 归一化至 [0,1]: 原范围 [{sy_min:.4f}, {sy_max:.4f}]")
        print(f"数据目录: xjtu_made_mscrgat | 源域 XJTU 工况1+工况2，目标域 XJTU 工况3 轴承1和2，滑窗 window_size={WINDOW_SIZE}，输入 [N, {source_x.shape[1]}, {source_x.shape[2]}]: source {source_x.shape}, target {target_x.shape}")
        print(f"  source_x range: [{source_x.min().item():.4f}, {source_x.max().item():.4f}], NaN: {torch.isnan(source_x).any().item()}")
        print(f"  source_y range: [{source_y.min().item():.4f}, {source_y.max().item():.4f}], NaN: {torch.isnan(source_y).any().item()}")
        print(f"  target_x range: [{target_x.min().item():.4f}, {target_x.max().item():.4f}], NaN: {torch.isnan(target_x).any().item()}")
        if test_x is not None:
            test_x = torch.nan_to_num(test_x, nan=0.0, posinf=0.0, neginf=0.0)
            test_y = torch.nan_to_num(test_y, nan=0.0, posinf=0.0, neginf=0.0)
            # 测试集特征用同一 scaler 变换，与源/目标同尺度
            test_x = torch.tensor(
                scaler_x.transform(test_x.reshape(-1, n_feat).numpy()).reshape(test_x.shape[0], T, n_feat),
                dtype=test_x.dtype,
            )
            # 确保测试集维度与配置一致（xjtu_made_mscrgat 为 13 维）
            if test_x.shape[2] != cfg.get("input_dim", 13):
                print(f"  警告：测试集维度 {test_x.shape[2]} 与配置 input_dim {cfg.get('input_dim', 13)} 不一致")
            print(f"  测试集 XJTU 工况3 轴承3: {test_x.shape}, {test_y.shape}")
    else:
        seq_len, dim = 64, 11
        source_x = torch.randn(200, seq_len, dim)
        source_y = torch.rand(200)
        target_x = torch.randn(160, seq_len, dim)
        test_x, test_y = None, None
        print("未找到 xjtu_made_mscrgat 数据目录，使用随机数据，输入维度 11。")

    # MSCRGAT（Res-GRU + MSCNN + 预测器 + 域判别器）对抗训练，损失 = Smooth L1 + domain + MMD
    model, disc, info = run_adversarial_mscrgat(source_x, source_y, target_x, config=cfg, device=dev)
    info["y_offset"] = y_offset
    print("MSCRGAT 对抗训练完成。")

    # 测试集评估：XJTU 工况3 轴承3（仅点估计：MAE、RMSE），测试 RUL 归一化到 [0,1] 与预测尺度一致
    if test_x is not None and test_y is not None:
        ty_min, ty_max = test_y.min().item(), test_y.max().item()
        span_t = ty_max - ty_min
        if span_t < 1e-8:
            span_t = 1.0
            print("\n⚠️  测试集 RUL 为常数（无变化），归一化后真实值全为 0，MAE/RMSE 会退化为 0 或无意义。")
        test_y_norm = ((test_y - ty_min) / span_t).float().clamp(0.0, 1.0)
        print("\n--- MSCRGAT 测试集（XJTU 工况3 轴承3）评估 (MAE, RMSE)，尺度 [0,1] ---")
        print(f"测试集输入维度: test_x.shape = {test_x.shape} (样本数={test_x.shape[0]}, 窗口大小={test_x.shape[1]}, 特征维度={test_x.shape[2]})")
        print(f"测试集 RUL 原范围 [{ty_min:.4f}, {ty_max:.4f}]，已归一化至 [0,1] 与源域一致")
        if test_x.shape[2] != cfg.get("input_dim", 13):
            print(f"⚠️  警告：测试集特征维度 {test_x.shape[2]} 与模型配置 input_dim {cfg.get('input_dim', 13)} 不一致！")
        test_metrics = evaluate_mscrgat_model(
            model, test_x, test_y_norm, dev,
            y_scale=1.0,
            y_offset=0.0,
            batch_size=cfg.get("batch_size", 64),
        )
        for name, value in test_metrics.items():
            print(f"  {name}: {value:.4f}")
        print(f"  (指标为 [0,1] 尺度，可直接与论文归一化指标对比)")

        # 预测结果绘图：预测与真实均为 [0,1]，纵轴固定 0-1
        model.eval()
        pred_list, y_list = [], []
        with torch.no_grad():
            loader = Data.DataLoader(
                Data.TensorDataset(test_x, test_y_norm),
                batch_size=cfg.get("batch_size", 64),
                shuffle=False,
            )
            for bx, by in loader:
                out = model(bx.to(dev))
                pred = out[0] if getattr(model, "heteroscedastic", False) else out
                pred_list.append(pred.cpu().numpy().ravel())
                y_list.append(by.numpy().ravel())
        y_true_np = np.concatenate(y_list, axis=0)
        pred_np = np.concatenate(pred_list, axis=0)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        n = len(y_true_np)
        axes[0].plot(np.arange(n), y_true_np, label="True RUL", color="C0")
        axes[0].plot(np.arange(n), pred_np, label="Pred RUL", color="C1", alpha=0.8)
        axes[0].set_xlabel("Sample index")
        axes[0].set_ylabel("RUL [0, 1]")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].legend()
        axes[0].set_title("RUL: True vs Pred (by sample)")
        axes[0].grid(True, alpha=0.3)
        axes[1].scatter(y_true_np, pred_np, alpha=0.6, s=20)
        axes[1].plot([0, 1], [0, 1], "r--", label="y=x")
        axes[1].set_xlabel("True RUL [0, 1]")
        axes[1].set_ylabel("Pred RUL [0, 1]")
        axes[1].set_xlim(0.0, 1.0)
        axes[1].set_ylim(0.0, 1.0)
        axes[1].legend()
        axes[1].set_title("Pred vs True")
        axes[1].grid(True, alpha=0.3)
        print(f"  绘图：True/Pred 均在 [0,1] 尺度，范围 True [{y_true_np.min():.4f}, {y_true_np.max():.4f}], Pred [{pred_np.min():.4f}, {pred_np.max():.4f}]")
        if np.allclose(y_true_np, 0) and np.allclose(pred_np, 0):
            print("  ⚠️  True 与 Pred 均为全 0：测试集 RUL 常量化后为 0，且模型预测也为 0，故 MAE/RMSE=0 无参考意义。可检查：测试轴承样本数/标签是否正常、模型是否在测试分布上输出塌缩。")
        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mscrgat_test_rul_plot.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  预测结果图已保存: {plot_path}")

        # 测试集效果可能不佳的原因分析（供排查与改进参考）
        print("\n--- 测试集效果可能不佳的原因分析 ---")
        print("  1. 标签尺度不一致：源域 RUL 由多轴承混合且每轴承单独 StandardScaler 归一化，训练时仅做整体平移(y_offset)；测试集为单轴承、其 RUL 为该轴承自己的 StandardScaler 尺度。pred+y_offset 与 test_y 不在同一物理尺度，MAE/RMSE 易偏大。")
        print("  2. 域偏移：源域=工况1+2，目标域=工况3 轴承1/4，测试=工况3 轴承3。测试轴承未参与训练，若域对齐不足或过拟合源域，目标域/测试集表现会下降。")
        print("  3. 输入特征尺度：dataset_maker 对每个轴承单独做 StandardScaler，源域与测试轴承特征分布不同，模型在测试集上输入分布与训练不一致。")
        print("  4. 目标域无标签：域对抗只对齐特征分布，RUL 仅从源域学习；若工况3 与工况1/2 的 RUL–特征关系差异大，泛化会变差。")
        print("  改进方向：统一 RUL 尺度(如全数据集 0–1 归一化)；源域/测试用同一特征 scaler(如仅在源域 fit、对 test 做 transform)；加强域对齐或增加 task_warmup；选用 run_supervised_mscrgat 在 0–1 标签上训练并评估。")

        # 点估计偏高时的改进措施（与文章不符时可参考）
        print("\n--- 点估计指标偏高时的改进措施 ---")
        print("  1. 指标口径：确认论文是否使用归一化 MAE/RMSE（如 MAE/R），当前已输出 MAE_norm、RMSE_norm 供对比。")
        print("  2. 任务优先：增大 task_warmup_epochs（如 120~200），让模型先学好 RUL 再做强域对齐。")
        print("  3. 域损失权重：适当减小 domain_loss_weight、mmd_loss_weight（如 0.005），避免域对齐压制 RUL 学习。")
        print("  4. 学习率：可尝试略提高 learn_rate（如 0.002~0.003）或对 predictor 单独设更大 lr。")
        print("  5. 正则：dropout 从 0.5 降至 0.3 以增强拟合能力；或增加 predictor_hidden 如 [128, 64, 32]。")
        print("  6. 标签尺度：若论文在 [0,1] 归一化 RUL 上训练，可改用 run_supervised_mscrgat 的归一化训练再反归一化评估。")
        print("  7. 特征标准化：对输入特征做逐维标准化（零均值单位方差），与论文预处理一致。")
        print("  8. 训练轮数：增加 epochs 或基于验证集 early stopping，确保收敛。")
    else:
        print("\n--- 无测试集数据，跳过测试集评估 ---")
