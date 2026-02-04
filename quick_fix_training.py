#!/usr/bin/env python3
"""
快速修复FBTCN训练不稳定的简单版本
可以直接在Jupyter notebook中使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import itertools

def stabilized_loss_function(targets, mean, log_var, lambda_pos=0.05, sigma_min=-8.0, sigma_max=2.0):
    """稳定化的损失函数"""
    # 裁剪log_var以避免数值不稳定
    log_var = torch.clamp(log_var, min=sigma_min, max=sigma_max)
    
    # 计算NLL损失
    precision = torch.exp(-log_var)
    squared_error = (targets - mean) ** 2
    nll = 0.5 * (log_var + precision * squared_error)
    
    # 恒正约束（更温和）
    positive_penalty = lambda_pos * torch.mean(torch.relu(-mean))
    
    # 方差正则化
    var_reg = 0.001 * torch.mean((log_var + 3.0) ** 2)
    
    return nll.mean() + positive_penalty + var_reg

def stable_training_loop(epochs, model, optimizer, train_loader, context_loader, validation_loader, device, config):
    """稳定的训练循环 - 简化版"""
    model = model.to(device)
    
    # 训练记录
    train_losses = []
    val_losses = []
    best_train_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 30  # 固定耐心值
    
    # 梯度裁剪参数
    max_grad_norm = 1.0
    
    print("开始稳定化训练...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        epoch_kl_losses = []
        epoch_nll_losses = []
        valid_batches = 0
        
        # 动态KL权重 - 更平滑的衰减
        init_kl_weight = config.get('kl_weight', 0.1)
        kl_weight = init_kl_weight * np.exp(-2 * epoch / epochs)
        kl_weight = max(kl_weight, init_kl_weight * 0.01)
        
        # 降低学习率
        if epoch == 20:  # 第20轮后降低学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        elif epoch == 50:  # 第50轮后再次降低
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        
        context_iter = itertools.cycle(context_loader)
        
        for i, train_batch in enumerate(train_loader):
            seq, labels = train_batch
            context_batch = next(context_iter)
            context_seq, _ = context_batch
            seq, labels = seq.to(device), labels.to(device)
            context_seq = context_seq.to(device)
            
            optimizer.zero_grad()
            
            try:
                # 前向传播
                mu, sigma, model_kl = model(seq)
                
                # Function KL计算（简化处理异常）
                try:
                    from function_kl import get_bayesian_model_mu_rho, calculate_function_kl
                    params_variational_mean, params_variational_logvar = get_bayesian_model_mu_rho(model)
                    function_kl = calculate_function_kl(params_variational_mean, params_variational_logvar, context_seq, model=model)
                except:
                    function_kl = torch.tensor(0.0, device=device)
                
                # 使用稳定化损失函数
                nll_loss = stabilized_loss_function(labels, mu, sigma)
                total_loss = nll_loss + kl_weight * (model_kl + function_kl)
                
                # 检查损失合理性
                if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 1e6:
                    print(f"跳过异常损失 - Epoch {epoch+1}, Batch {i+1}, Loss: {total_loss.item():.2e}")
                    continue
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # 如果梯度过大，跳过更新
                if grad_norm > 5.0:
                    print(f"跳过大梯度更新 - Grad norm: {grad_norm:.2f}")
                    continue
                
                # 参数更新
                optimizer.step()
                
                # 记录损失
                epoch_losses.append(total_loss.item())
                epoch_kl_losses.append((model_kl + function_kl).item())
                epoch_nll_losses.append(nll_loss.item())
                valid_batches += 1
                
            except Exception as e:
                print(f"批次出错 - Epoch {epoch+1}, Batch {i+1}: {str(e)}")
                continue
        
        # 计算平均损失
        if len(epoch_losses) > 0:
            avg_train_loss = np.mean(epoch_losses)
            avg_kl_loss = np.mean(epoch_kl_losses)
            avg_nll_loss = np.mean(epoch_nll_losses)
            train_losses.append(avg_train_loss)
            
            # 验证损失计算
            val_loss = evaluate_model_simple(model, validation_loader, device)
            val_losses.append(val_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # 训练集sigma
            if 'sigma' in locals():
                train_sigma_mean = sigma.mean().item()
            else:
                train_sigma_mean = 0.0
                
            print(f'Epoch {epoch+1:3d}/{epochs} | '
                  f'Train: {avg_train_loss:.6f} | '
                  f'Val: {val_loss:.6f} | '
                  f'KL: {avg_kl_loss:.4f} | '
                  f'NLL: {avg_nll_loss:.6f} | '
                  f'Sigma: {train_sigma_mean:.4f} | '
                  f'LR: {current_lr:.2e} | '
                  f'Valid: {valid_batches}/{len(train_loader)}')
            
            # 早停检查
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1
                patience_counter = 0
                print(f"✓ 新的最佳模型 - 损失: {avg_train_loss:.6f}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"早停于epoch {epoch+1}, 最佳epoch: {best_epoch}")
                break
        else:
            print(f'Epoch {epoch+1:3d} - 没有有效批次')
    
    # 保存最佳模型
    if best_model_state is not None:
        torch.save(best_model_state, 'best_model_fbtcn_stable.pt')
        print(f"最佳模型已保存，损失: {best_train_loss:.6f}")
    
    print(f'\n训练耗时: {time.time() - start_time:.0f} 秒')
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    if len(train_losses) > 5:
        # 损失稳定性
        changes = np.abs(np.diff(train_losses))
        plt.plot(changes, 'g-', label='Loss Change')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Change')
        plt.title('Training Stability')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return train_losses, val_losses, best_epoch if 'best_epoch' in locals() else len(train_losses)

def evaluate_model_simple(model, data_loader, device):
    """简单的模型评估"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            
            try:
                mu, _, _ = model(data)
                loss = F.mse_loss(mu, targets)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1
            except:
                continue
    
    return total_loss / num_batches if num_batches > 0 else float('inf')

# 使用示例
def fix_training_config(config):
    """修复配置以提高稳定性"""
    stable_config = config.copy()
    stable_config['learn_rate'] = min(config['learn_rate'] * 0.5, 5e-5)  # 降低学习率
    stable_config['kl_weight'] = min(config['kl_weight'] * 0.5, 0.05)    # 降低KL权重
    stable_config['batch_size'] = min(config['batch_size'], 64)          # 适中的batch size
    return stable_config

print("快速修复工具已加载！")
print("使用方法：")
print("1. stable_config = fix_training_config(config)")
print("2. train_losses, val_losses, best_epoch = stable_training_loop(epochs, model, optimizer, train_loader, context_loader, validation_loader, device, stable_config)") 