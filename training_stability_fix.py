#!/usr/bin/env python3
"""
修复Bayesian TCN训练不稳定问题的解决方案
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

def stabilized_training_loop(model, train_loader, context_loader, validation_loader, config, device):
    """
    稳定化的训练循环
    解决梯度爆炸、loss震荡等问题
    """
    
    # 优化器设置 - 使用更保守的学习率
    if config['opt'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learn_rate'] * 0.5, weight_decay=1e-5)
    elif config['opt'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learn_rate'] * 0.5, weight_decay=1e-5)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learn_rate'] * 0.1, momentum=0.9, weight_decay=1e-5)
    
    # 学习率调度器 - 使用ReduceLROnPlateau，更平滑
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15, verbose=True, min_lr=1e-7)
    
    # 损失函数改进
    loss_function = StabilizedAUNLL()
    
    # 训练记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('patience', 50)
    
    # 梯度裁剪参数
    max_grad_norm = 0.5  # 更严格的梯度裁剪
    
    print("开始稳定化训练...")
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_losses = []
        epoch_kl_losses = []
        epoch_nll_losses = []
        
        # 动态调整KL权重 - 更平滑的衰减
        base_kl_weight = config.get('kl_weight', 0.1)
        kl_weight = base_kl_weight * np.exp(-epoch / (config['epochs'] * 0.3))
        kl_weight = max(kl_weight, base_kl_weight * 0.01)  # 最小权重
        
        import itertools
        context_iter = itertools.cycle(context_loader)
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            context_batch = next(context_iter)
            context_data, _ = context_batch
            
            data, targets = data.to(device), targets.to(device)
            context_data = context_data.to(device)
            
            # 前向传播
            try:
                optimizer.zero_grad()
                
                # 模型预测
                mu, sigma, model_kl = model(data)
                
                # Function KL计算（如果有的话）
                if hasattr(model, 'function_kl_loss'):
                    try:
                        from function_kl import get_bayesian_model_mu_rho, calculate_function_kl
                        params_mean, params_logvar = get_bayesian_model_mu_rho(model)
                        function_kl = calculate_function_kl(params_mean, params_logvar, context_data, model=model)
                    except:
                        function_kl = torch.tensor(0.0, device=device)
                else:
                    function_kl = torch.tensor(0.0, device=device)
                
                # 稳定化的损失计算
                nll_loss = loss_function(targets, mu, sigma)
                total_kl = model_kl + function_kl
                total_loss = nll_loss + kl_weight * total_kl
                
                # 检查损失值
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"跳过无效损失 - Epoch {epoch+1}, Batch {batch_idx+1}")
                    continue
                
                # 如果损失过大，使用损失裁剪
                if total_loss.item() > 1e6:
                    print(f"损失过大，使用裁剪 - Epoch {epoch+1}, Loss: {total_loss.item():.2e}")
                    total_loss = torch.clamp(total_loss, max=1e6)
                
                # 反向传播
                total_loss.backward()
                
                # 梯度检查和裁剪
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # 如果梯度范数过大，跳过这次更新
                if total_grad_norm > 10.0:
                    print(f"跳过梯度过大的更新 - Grad norm: {total_grad_norm:.2f}")
                    continue
                
                # 参数更新
                optimizer.step()
                
                # 记录损失
                epoch_losses.append(total_loss.item())
                epoch_kl_losses.append(total_kl.item())
                epoch_nll_losses.append(nll_loss.item())
                
            except Exception as e:
                print(f"训练出错 - Epoch {epoch+1}, Batch {batch_idx+1}: {str(e)}")
                continue
        
        # 计算平均损失
        if len(epoch_losses) > 0:
            avg_train_loss = np.mean(epoch_losses)
            avg_kl_loss = np.mean(epoch_kl_losses)
            avg_nll_loss = np.mean(epoch_nll_losses)
            train_losses.append(avg_train_loss)
        else:
            print(f"Epoch {epoch+1}: 没有有效的batch")
            continue
        
        # 验证
        model.eval()
        val_loss = evaluate_model_stable(model, validation_loader, device)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印信息
        if 'sigma' in locals():
            sigma_mean = sigma.mean().item()
        else:
            sigma_mean = 0.0
            
        print(f'Epoch {epoch+1:3d}/{config["epochs"]} | '
              f'Train: {avg_train_loss:.6f} | '
              f'Val: {val_loss:.6f} | '
              f'KL: {avg_kl_loss:.4f} | '
              f'NLL: {avg_nll_loss:.6f} | '
              f'Sigma: {sigma_mean:.4f} | '
              f'LR: {current_lr:.2e} | '
              f'KL_w: {kl_weight:.4f}')
        
        # 早停和模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'config': config
            }, 'best_model_stable.pt')
            print(f"✓ 保存最佳模型 (验证损失: {val_loss:.6f})")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"早停于epoch {epoch+1}")
            break
    
    return train_losses, val_losses


class StabilizedAUNLL(nn.Module):
    """稳定化的Aleatoric Uncertainty NLL损失"""
    
    def __init__(self, lambda_pos=0.1, sigma_min=-10.0, sigma_max=3.0):
        super().__init__()
        self.lambda_pos = lambda_pos
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def forward(self, targets, mean, log_var):
        """
        Args:
            targets: 真实值
            mean: 预测均值
            log_var: 预测的对数方差
        """
        # 裁剪log_var以避免数值不稳定
        log_var = torch.clamp(log_var, min=self.sigma_min, max=self.sigma_max)
        
        # 计算NLL损失
        precision = torch.exp(-log_var)  # 1/σ²
        squared_error = (targets - mean) ** 2
        
        # NLL = 0.5 * (log(2π) + log_var + (y-μ)²/σ²)
        # 忽略常数项 0.5 * log(2π)
        nll = 0.5 * (log_var + precision * squared_error)
        
        # 恒正约束
        positive_penalty = self.lambda_pos * torch.mean(torch.relu(-mean))
        
        # 方差正则化 - 防止方差过小或过大
        var_reg = 0.01 * torch.mean((log_var + 5.0) ** 2)  # 鼓励log_var接近-5
        
        return nll.mean() + positive_penalty + var_reg


def evaluate_model_stable(model, data_loader, device):
    """稳定的模型评估"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            
            try:
                # 多次前向传播取平均
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


def analyze_training_issues(train_losses):
    """分析训练问题"""
    print("\n=== 训练问题分析 ===")
    
    if len(train_losses) < 2:
        print("训练轮数不足以分析")
        return
    
    # 计算损失变化率
    loss_changes = np.diff(train_losses)
    loss_ratios = np.abs(loss_changes) / np.array(train_losses[:-1])
    
    # 检测突然的损失增长
    large_increases = np.where(loss_ratios > 1.0)[0]
    if len(large_increases) > 0:
        print(f"发现 {len(large_increases)} 次损失突增:")
        for idx in large_increases[:5]:  # 只显示前5次
            print(f"  Epoch {idx+1}->{idx+2}: {train_losses[idx]:.2e} -> {train_losses[idx+1]:.2e}")
    
    # 检测损失爆炸
    max_loss = max(train_losses)
    min_loss = min(train_losses)
    if max_loss / min_loss > 1000:
        print(f"检测到损失爆炸: 最大/最小 = {max_loss/min_loss:.2e}")
    
    # 稳定性评估
    if len(train_losses) > 10:
        recent_losses = train_losses[-10:]
        stability = np.std(recent_losses) / np.mean(recent_losses)
        print(f"最近10轮稳定性 (CV): {stability:.4f}")
        
        if stability > 0.5:
            print("⚠️  训练不稳定 - 建议:")
            print("   1. 降低学习率")
            print("   2. 增强梯度裁剪")
            print("   3. 调整KL权重衰减")
            print("   4. 检查数据预处理")


# 修复配置建议
def get_stable_config_suggestions():
    """获取稳定训练的配置建议"""
    return {
        "learn_rate": 5e-5,  # 降低学习率
        "kl_weight": 0.05,   # 降低KL权重
        "batch_size": 64,    # 适中的batch size
        "opt": "AdamW",      # 使用AdamW
        "epochs": 300,       # 适当的训练轮数
        "patience": 30,      # 早停耐心
        "gradient_clip": 0.5, # 梯度裁剪
        "weight_decay": 1e-5, # L2正则化
    }


if __name__ == "__main__":
    print("训练稳定性修复工具")
    print("主要改进:")
    print("1. 梯度裁剪和范数检查")
    print("2. 稳定化的损失函数")
    print("3. 自适应学习率调度")
    print("4. 损失值监控和裁剪")
    print("5. 改进的KL权重衰减")
    
    # 显示建议配置
    suggestions = get_stable_config_suggestions()
    print("\n推荐配置:")
    for key, value in suggestions.items():
        print(f"  {key}: {value}") 