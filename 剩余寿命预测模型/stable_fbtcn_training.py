#!/usr/bin/env python3
"""
稳定版FBTCN训练代码
修复原始训练中的不稳定问题
"""

import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import matplotlib.pyplot as plt
import time
import sys
import os
from joblib import dump, load
import torch.utils.data as Data
# 移除混合精度训练相关导入
# 添加路径
sys.path.append('剩余寿命预测模型')
try:
    from loss_function import compute_au_nll_with_pos
    from function_kl import get_bayesian_model_parameters, get_bayesian_model_mu_rho, calculate_function_kl
    from metrics import picp, nmpiw, ece, aleatoric_uncertainty, epistemic_uncertainty, sharpness, mae, rmse
except ImportError:
    print("无法导入必要模块，请检查路径设置")

# 稳定化的损失函数
class StabilizedAUNLL(nn.Module):
    """
    稳定化的Aleatoric Uncertainty NLL损失
    
    重要说明：
    - 模型输出的sigma是对数方差（log variance）
    - 损失函数直接使用对数方差计算NLL
    - NLL = 0.5 * (log_var + exp(-log_var) * (y_true - y_pred)^2)
    """
    
    def __init__(self, lambda_pos=0.05, sigma_min=-8.0, sigma_max=2.0):
        super().__init__()
        self.lambda_pos = lambda_pos
        self.sigma_min = sigma_min  # 对数方差的最小值
        self.sigma_max = sigma_max  # 对数方差的最大值
    
    def forward(self, targets, mean, log_var):
        """
        Args:
            targets: 真实值
            mean: 预测均值  
            log_var: 预测的对数方差（模型直接输出）
        """
        # 裁剪log_var以避免数值不稳定
        log_var = torch.clamp(log_var, min=self.sigma_min, max=self.sigma_max)
        
        # 计算NLL损失
        # precision = 1/variance = exp(-log_var)
        precision = torch.exp(-log_var)
        squared_error = (targets - mean) ** 2
        # NLL = 0.5 * (log_var + precision * squared_error)
        nll = 0.5 * (log_var + precision * squared_error)
        
        # 恒正约束（更温和）
        # positive_penalty = self.lambda_pos * torch.mean(torch.relu(-mean))
        positive_penalty = 0.0
        # 方差正则化 - 防止方差过小或过大
        # var_reg = 0.001 * torch.mean((log_var + 3.0) ** 2)
        var_reg = 0.0
        return nll.mean() + positive_penalty + var_reg

def model_train_stable(epochs, model, init_model, optimizer, 
        loss_function, train_loader, context_loader, validation_loader, 
        device, config, best_pt_model_name, skip_validation=False):
    """稳定化的训练函数"""
    model = model.to(device)
    init_model = init_model.to(device)
    # 学习率调度器-修复兼容性问题
    if config.get('scheduler', 'plateau') == 'plateau' and not skip_validation:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15, min_lr=1e-7)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 训练记录
    minimum_train_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = [] if not skip_validation else None
    
    # 梯度裁剪参数
    max_grad_norm = 5.0  # 放宽梯度裁剪，让模型能够有效学习
    
    # 损失稳定性检查
    loss_explosion_threshold = 1e8
    consecutive_bad_batches = 0
    max_consecutive_bad = 5
    
    start_time = time.time()
    patience_counter = 0
    patience = config.get('patience', 50) if config.get('patience', 50) != "inf" else epochs
    
    print(f"开始稳定化训练... {'(跳过验证集)' if skip_validation else ''}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = []
        kl_losses = []
        nll_losses = []
        # 性能优化：只在需要时收集统计信息（每10个epoch或最后一个epoch）
        collect_stats = (epoch % 10 == 0 or epoch == epochs - 1)
        mus = np.array([]) if collect_stats else None
        sigmas = np.array([]) if collect_stats else None
        valid_batches = 0
        # 固定KL权重
        # kl_weight = config.get('kl_weight', 0.01)

        # KL权重warm-up
        kl_weight = get_kl_weight_with_warmup(epoch, epochs, config.get('kl_weight', 0.01))
        
        # 分阶段KL权重
        # kl_weight = get_training_phase_kl_weight(epoch, epochs)

        # 无KL散度
        # init_kl_weight = 0

        # 指数衰减KL权重，但最小值为init_kl_weight * 0.01
        # kl_weight = init_kl_weight * np.exp(-2 * epoch / epochs)
        # kl_weight = max(kl_weight, init_kl_weight * 0.01)

        # 线性衰减KL权重
        # kl_weight = min(epoch / epochs, init_kl_weight)

        
        # 创建循环的context_loader迭代器
        import itertools
        context_iter = itertools.cycle(context_loader)
        
        for i, train_batch in enumerate(train_loader):
            seq, labels = train_batch
            context_batch = next(context_iter)
            context_seq, _ = context_batch
            seq, labels = seq.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            context_seq = context_seq.to(device, non_blocking=True)
            optimizer.zero_grad()
            try:
                mu, sigma, _ = model(seq)
                # 性能优化：只在需要时收集统计信息，避免频繁的GPU→CPU传输
                if collect_stats:
                    mus = np.append(mus, mu.detach().cpu().numpy().flatten())
                    sigmas = np.append(sigmas, sigma.detach().cpu().numpy().flatten())
                # Function KL计算
                if kl_weight != 0:
                    try:
                        function_kl = calculate_function_kl(context_seq, model=model, init_model=init_model, 
                                        enable_diagnosis=False, diagnosis_save_path='kl_diagnosis.png', 
                                        diagnosis_threshold=1e5, debug_nan=False)
                        # 性能优化：减少print频率（每10个batch且每10个epoch才打印）
                        # if (i % 10 == 0 and epoch % 10 == 0) or (i == 0 and epoch < 3):
                            # print("function_kl", function_kl.item())
                    except Exception as e:
                        # if epoch < 10:  # 只在前10个epoch打印错误信息
                            # print(f"Function KL计算失败: {str(e)}")
                        function_kl = torch.tensor(0.0, device=device)
                else:
                    function_kl = torch.tensor(0.0, device=device)
                # 损失计算
                nll_loss = loss_function(labels, mu, sigma)
                total_loss = (nll_loss + kl_weight * function_kl / config.get('batch_size', 128))
                # 检查损失是否合理
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    consecutive_bad_batches += 1
                    # print(f"跳过无效损失 - Epoch {epoch+1}, Batch {i+1}")
                    if consecutive_bad_batches >= max_consecutive_bad:
                        print("连续无效批次过多，停止训练")
                        return train_losses, val_losses, epoch
                    continue
                
                # 损失爆炸检查
                if total_loss.item() > loss_explosion_threshold:
                    consecutive_bad_batches += 1
                    # print(f"损失爆炸 - Epoch {epoch+1}, Loss: {total_loss.item():.2e}")
                    if consecutive_bad_batches >= max_consecutive_bad:
                        print("损失持续爆炸，停止训练")
                        return train_losses, val_losses, epoch
                    continue
                
                # 重置坏批次计数器
                consecutive_bad_batches = 0
                
                # 标准反向传播
                total_loss.backward()
                
                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # 检查梯度是否过大
                if grad_norm > 10.0:
                    # print(f"梯度过大，跳过更新 - Grad norm: {grad_norm:.2f}")
                    continue
                
                # 参数更新
                optimizer.step()
                
                # 记录损失
                train_loss.append(total_loss.item())
                kl_losses.append(function_kl.item())
                nll_losses.append(nll_loss.item())
                valid_batches += 1
                
            except Exception as e:
                # print(f"批次处理出错 - Epoch {epoch+1}, Batch {i+1}: {str(e)}")
                consecutive_bad_batches += 1
                if consecutive_bad_batches >= max_consecutive_bad:
                    print("错误过多，停止训练")
                    return train_losses, val_losses, epoch
                continue
        # 性能优化：只在收集了数据时打印统计信息
        # if collect_stats and mus is not None and len(mus) > 0:
            # print(f"mu_mean: {np.mean(mus):.6f}, sigma_mean: {np.mean(sigmas):.6f}")
        
        # 更新学习率
        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        
        # 计算平均损失
        if len(train_loss) > 0:
            train_avg_loss = np.average(train_loss)
            train_avg_kl = np.average(kl_losses) if len(kl_losses) > 0 else 0.0
            train_avg_nll = np.average(nll_losses) if len(nll_losses) > 0 else 0.0
            train_losses.append(train_avg_loss)
            
            # 条件性计算验证损失
            if not skip_validation:
                val_loss = evaluate_model_stable(model, validation_loader, device)
                val_losses.append(val_loss)
                
                # 学习率调度（对于ReduceLROnPlateau）
                # if isinstance(scheduler, ReduceLROnPlateau):
                #     scheduler.step(val_loss)
            else:
                val_loss = None
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # 训练集sigma（sigma是对数方差，显示实际标准差）
            if 'sigma' in locals():
                # sigma是对数方差，exp(sigma)是方差，sqrt(exp(sigma))是标准差
                train_sigma_mean = sigma.mean().item()
            else:
                train_sigma_mean = float('nan')
            
            # 动态打印格式
            if skip_validation:
                print(f'Epoch {epoch+1:3d}/{epochs} | '
                      f'Train: {train_avg_loss:.6f} | '
                      f'KL: {train_avg_kl:.4f} | '
                      f'NLL: {train_avg_nll:.6f} | '
                      f'Sigma: {train_sigma_mean:.4f} | '
                      f'LR: {current_lr:.2e} | '
                      f'Valid: {valid_batches}/{len(train_loader)}')
            else:
                print(f'Epoch {epoch+1:3d}/{epochs} | '
                      f'Train: {train_avg_loss:.6f} | '
                      f'Val: {val_loss:.6f} | '
                      f'KL: {train_avg_kl:.4f} | '
                      f'NLL: {train_avg_nll:.6f} | '
                      f'Sigma: {train_sigma_mean:.4f} | '
                      f'LR: {current_lr:.2e} | '
                      f'Valid: {valid_batches}/{len(train_loader)}')
            
            # 早停和最佳模型保存
            if train_avg_loss < minimum_train_loss:
                minimum_train_loss = train_avg_loss
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1
                patience_counter = 0
                print(f"✓ 新的最佳模型 - 训练损失: {train_avg_loss:.6f}")
                # 可视化训练过程
                # if not skip_validation:
                    # plot_training_curves(train_losses, val_losses, best_epoch if 'best_epoch' in locals() else len(train_losses))
                # else:
                    # 只绘制训练损失曲线
                    # plot_training_curves_single(train_losses, best_epoch if 'best_epoch' in locals() else len(train_losses))
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"早停于epoch {epoch+1}, 最佳epoch: {best_epoch}")
                break
        else:
            # print(f'Epoch {epoch+1:3d} - 没有有效的批次')
            continue
    
    # 保存最佳模型
    if best_model_state is not None:
        # torch.save(best_model_state, 'best_model_fbtcn_stable.pt')
        torch.save(best_model_state, best_pt_model_name)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"最佳模型已保存，训练损失: {minimum_train_loss:.6f}, 保存时间: {current_time}")
    
    print(f'\n训练耗时: {time.time() - start_time:.0f} 秒')
    
    
    return train_losses, val_losses, best_epoch if 'best_epoch' in locals() else len(train_losses)

def evaluate_model_stable(model, data_loader, device):
    """稳定的模型评估"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            try:
                # 多次前向传播以获得更稳定的预测
                predictions = []
                for _ in range(3):
                    mu, _, _ = model(data)
                    predictions.append(mu)
                
                mean_pred = torch.stack(predictions).mean(dim=0)
                loss = F.mse_loss(mean_pred, targets)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1
                    
            except Exception as e:
                print(f"验证出错: {str(e)}")
                continue
    
    return total_loss / num_batches if num_batches > 0 else float('inf')

def plot_training_curves(train_losses, val_losses, best_epoch):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(15, 5))
    
    # 训练损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(val_losses, 'r-', label='Validation Loss')
    plt.axvline(x=best_epoch-1, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 训练稳定性分析
    plt.subplot(1, 3, 2)
    if len(train_losses) > 1:
        loss_changes = [abs(train_losses[i] - train_losses[i-1]) / train_losses[i-1] 
                       for i in range(1, len(train_losses))]
        plt.plot(loss_changes, 'g-', label='Loss Change Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Change Rate')
        plt.title('Training Stability')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stable_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_curves_single(train_losses, best_epoch):
    """绘制仅训练损失曲线（跳过验证集时使用）"""
    plt.figure(figsize=(12, 5))
    
    # 训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.axvline(x=best_epoch-1, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 训练稳定性分析
    plt.subplot(1, 2, 2)
    if len(train_losses) > 1:
        loss_changes = [abs(train_losses[i] - train_losses[i-1]) / train_losses[i-1] 
                       for i in range(1, len(train_losses))]
        plt.plot(loss_changes, 'g-', label='Loss Change Rate', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Change Rate')
        plt.title('Training Stability')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_only_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_stable_optimizer(model, config):
    """获取稳定的优化器配置"""
    # 使用更保守的学习率
    lr = config['learn_rate'] * 0.5
    
    if config['opt'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5, eps=1e-8)
    elif config['opt'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)
    elif config['opt'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr*0.1, momentum=0.9, weight_decay=1e-5)
    else:
        raise ValueError(f"不支持的优化器: {config['opt']}")
    
    return optimizer

def get_kl_weight_with_warmup(epoch, total_epochs, base_weight=0.01):
    """
    KL权重warm-up策略
    前20%的epoch逐渐增加，然后保持稳定或缓慢衰减
    """
    warmup_epochs = total_epochs * 0.2
    
    if epoch < warmup_epochs:
        # Warm-up阶段：从0逐渐增加到base_weight
        warmup_factor = epoch / warmup_epochs
        return base_weight * warmup_factor
    else:
        # 稳定阶段：缓慢衰减
        decay_factor = np.exp(-2 * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
        return base_weight * max(decay_factor, 0.1)  # 最小保持10%

def get_training_phase_kl_weight(epoch, total_epochs):
    """
    分阶段KL权重策略
    """
    phase1_end = total_epochs * 0.3  # 前30%: 纯数据拟合
    phase2_end = total_epochs * 0.7  # 中30%: 逐渐加入KL
    # 后40%: 稳定KL权重
    
    if epoch < phase1_end:
        return 0.0  # 纯数据拟合，无KL约束
    elif epoch < phase2_end:
        # 线性增加KL权重
        progress = (epoch - phase1_end) / (phase2_end - phase1_end)
        return 0.01 * progress
    else:
        # 稳定期，较小的KL权重
        return 0.005

def adaptive_kl_weight(kl_values_history, base_weight=0.01):
    """
    基于KL值历史的自适应权重调整
    """
    if len(kl_values_history) < 5:
        return base_weight * 0.1  # 初期使用很小的权重
    
    # 计算最近5个epoch的KL值变化
    recent_kl = np.array(kl_values_history[-5:])
    kl_std = np.std(recent_kl)
    kl_mean = np.mean(recent_kl)
    
    # 如果KL值变化太大，减小权重
    if kl_std > kl_mean * 0.5:  # 变化超过均值的50%
        return base_weight * 0.5
    elif kl_std > kl_mean * 0.2:  # 变化超过均值的20%
        return base_weight * 0.8
    else:
        return base_weight

def smoothed_kl_loss(current_kl, kl_history, smooth_factor=0.9):
    """
    对KL损失进行指数移动平均平滑
    """
    if len(kl_history) == 0:
        return current_kl
    
    # 指数移动平均
    smoothed_kl = smooth_factor * kl_history[-1] + (1 - smooth_factor) * current_kl
    kl_history.append(smoothed_kl)
    
    # 保持历史长度不超过10
    if len(kl_history) > 10:
        kl_history.pop(0)
    
    return smoothed_kl
