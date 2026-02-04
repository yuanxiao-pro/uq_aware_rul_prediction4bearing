#!/usr/bin/env python3
"""
贝叶斯神经网络分布可视化工具
用于可视化先验分布和变分分布，帮助诊断KL散度问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def visualize_parameter_distributions(model, init_model, save_path=None):
    """
    可视化贝叶斯神经网络的参数分布
    
    Args:
        model: 当前训练的贝叶斯模型
        init_model: 初始化的贝叶斯模型（用作先验）
        save_path: 保存图片的路径（可选）
    """
    # 获取参数
    from function_kl import get_bayesian_model_mu_rho
    
    # 变分分布参数
    var_mu, var_rho = get_bayesian_model_mu_rho(model)
    # 先验分布参数  
    prior_mu, prior_rho = get_bayesian_model_mu_rho(init_model)
    
    # 设置图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Bayesian Neural Network Parameter Distribution Comparison', fontsize=16, fontweight='bold')
    
    # 分析参数统计
    print("=== Parameter Distribution Statistical Analysis ===")
    
    # 1. 参数均值分布对比
    ax = axes[0, 0]
    var_mu_values = torch.cat([v.flatten() for v in var_mu.values()]).cpu().numpy()
    prior_mu_values = torch.cat([v.flatten() for v in prior_mu.values()]).cpu().numpy()
    
    ax.hist(prior_mu_values, bins=50, alpha=0.7, label='Prior Mean', color='blue', density=True)
    ax.hist(var_mu_values, bins=50, alpha=0.7, label='Variational Mean', color='red', density=True)
    ax.set_xlabel('Parameter Mean')
    ax.set_ylabel('Density')
    ax.set_title('Parameter Mean Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print(f"Prior mean statistics: mean={np.mean(prior_mu_values):.6f}, std={np.std(prior_mu_values):.6f}")
    print(f"Variational mean statistics: mean={np.mean(var_mu_values):.6f}, std={np.std(var_mu_values):.6f}")
    
    # 2. 参数方差分布对比
    ax = axes[0, 1]
    var_variance = torch.cat([torch.exp(v).flatten() for v in var_rho.values()]).cpu().numpy()
    prior_variance = torch.cat([torch.exp(v).flatten() for v in prior_rho.values()]).cpu().numpy()
    
    # 对数尺度显示方差
    ax.hist(np.log10(prior_variance + 1e-8), bins=50, alpha=0.7, label='Prior Variance(log10)', color='blue', density=True)
    ax.hist(np.log10(var_variance + 1e-8), bins=50, alpha=0.7, label='Variational Variance(log10)', color='red', density=True)
    ax.set_xlabel('log10(Variance)')
    ax.set_ylabel('Density')
    ax.set_title('Parameter Variance Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print(f"Prior variance statistics: mean={np.mean(prior_variance):.6f}, std={np.std(prior_variance):.6f}")
    print(f"Variational variance statistics: mean={np.mean(var_variance):.6f}, std={np.std(var_variance):.6f}")
    
    # 3. rho参数分布对比
    ax = axes[0, 2]
    var_rho_values = torch.cat([v.flatten() for v in var_rho.values()]).cpu().numpy()
    prior_rho_values = torch.cat([v.flatten() for v in prior_rho.values()]).cpu().numpy()
    
    ax.hist(prior_rho_values, bins=50, alpha=0.7, label='Prior Rho', color='blue', density=True)
    ax.hist(var_rho_values, bins=50, alpha=0.7, label='Variational Rho', color='red', density=True)
    ax.set_xlabel('Rho Value')
    ax.set_ylabel('Density')
    ax.set_title('Rho Parameter Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print(f"Prior rho statistics: mean={np.mean(prior_rho_values):.6f}, std={np.std(prior_rho_values):.6f}")
    print(f"Variational rho statistics: mean={np.mean(var_rho_values):.6f}, std={np.std(var_rho_values):.6f}")
    
    # 4. 层级分析
    ax = axes[1, 0]
    layer_names = []
    prior_means = []
    var_means = []
    
    for name in var_mu.keys():
        if 'weight' in name:  # 只分析权重层
            layer_names.append(name.replace('mu_', '').replace('.weight', ''))
            prior_means.append(prior_mu[name].mean().item())
            var_means.append(var_mu[name].mean().item())
    
    x_pos = np.arange(len(layer_names))
    width = 0.35
    
    ax.bar(x_pos - width/2, prior_means, width, label='Prior Mean', alpha=0.8, color='blue')
    ax.bar(x_pos + width/2, var_means, width, label='Variational Mean', alpha=0.8, color='red')
    ax.set_xlabel('Network Layer')
    ax.set_ylabel('Average Parameter Value')
    ax.set_title('Layer-wise Parameter Mean Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 参数差异分析
    ax = axes[1, 1]
    param_diff = var_mu_values - prior_mu_values[:len(var_mu_values)]
    
    ax.hist(param_diff, bins=50, alpha=0.7, color='green', density=True)
    ax.axvline(np.mean(param_diff), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(param_diff):.6f}')
    ax.set_xlabel('Parameter Difference (Variational - Prior)')
    ax.set_ylabel('Density')
    ax.set_title('Parameter Mean Shift Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print(f"Parameter shift statistics: mean={np.mean(param_diff):.6f}, std={np.std(param_diff):.6f}")
    
    # 6. KL散度贡献分析
    ax = axes[1, 2]
    layer_kl_contributions = []
    layer_names_kl = []
    
    for name in var_mu.keys():
        if name in prior_mu:
            # 计算每层的近似KL贡献
            mu_var = var_mu[name].flatten()
            mu_prior = prior_mu[name].flatten()
            
            rho_name = name.replace('mu_', 'rho_')
            if rho_name in var_rho and rho_name in prior_rho:
                var_var = torch.exp(var_rho[rho_name]).flatten()
                var_prior = torch.exp(prior_rho[rho_name]).flatten()
                
                # 近似KL散度计算 (单变量高斯)
                kl_approx = 0.5 * (
                    torch.log(var_prior / var_var) + 
                    var_var / var_prior + 
                    (mu_var - mu_prior).pow(2) / var_prior - 1
                ).sum().item()
                
                layer_kl_contributions.append(kl_approx)
                layer_names_kl.append(name.replace('mu_', ''))
    
    if layer_kl_contributions:
        ax.bar(range(len(layer_kl_contributions)), layer_kl_contributions, alpha=0.8, color='orange')
        ax.set_xlabel('Network Layer')
        ax.set_ylabel('Approximate KL Contribution')
        ax.set_title('Layer-wise KL Divergence Contribution Analysis')
        ax.set_xticks(range(len(layer_names_kl)))
        ax.set_xticklabels(layer_names_kl, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        print(f"Total approximate KL divergence: {sum(layer_kl_contributions):.2f}")
        print(f"Maximum KL contribution layer: {layer_names_kl[np.argmax(layer_kl_contributions)]}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return {
        'prior_mu_stats': {'mean': np.mean(prior_mu_values), 'std': np.std(prior_mu_values)},
        'var_mu_stats': {'mean': np.mean(var_mu_values), 'std': np.std(var_mu_values)},
        'prior_var_stats': {'mean': np.mean(prior_variance), 'std': np.std(prior_variance)},
        'var_var_stats': {'mean': np.mean(var_variance), 'std': np.std(var_variance)},
        'total_kl_approx': sum(layer_kl_contributions) if layer_kl_contributions else 0
    }

def visualize_function_space_distributions(model, init_model, context_inputs, save_path=None):
    """
    可视化函数空间的分布
    
    Args:
        model: 当前训练的贝叶斯模型
        init_model: 初始化的贝叶斯模型
        context_inputs: 上下文输入数据
        save_path: 保存路径
    """
    from function_kl import calculate_moments, get_bayesian_model_mu_rho
    
    # 获取参数
    prior_mu, prior_rho = get_bayesian_model_mu_rho(init_model)
    var_mu, var_rho = get_bayesian_model_mu_rho(model)
    
    # 计算函数空间的分布
    model_copy = torch.nn.utils.deepcopy.deepcopy(model)
    
    # 先验分布在函数空间的投影
    prior_mean, prior_cov = calculate_moments(model_copy, prior_mu, prior_rho, context_inputs)
    # 变分分布在函数空间的投影  
    var_mean, var_cov = calculate_moments(model_copy, var_mu, var_rho, context_inputs)
    
    # 转换为numpy
    prior_mean_np = prior_mean[:, 0].detach().cpu().numpy()
    var_mean_np = var_mean[:, 0].detach().cpu().numpy()
    prior_std_np = torch.sqrt(torch.diagonal(prior_cov[:, :, 0])).detach().cpu().numpy()
    var_std_np = torch.sqrt(torch.diagonal(var_cov[:, :, 0])).detach().cpu().numpy()
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Function Space Distribution Visualization', fontsize=16, fontweight='bold')
    
    # 1. 函数均值对比
    ax = axes[0, 0]
    x_axis = np.arange(len(prior_mean_np))
    ax.plot(x_axis, prior_mean_np, 'b-', label='Prior Mean', alpha=0.8, linewidth=2)
    ax.plot(x_axis, var_mean_np, 'r-', label='Variational Mean', alpha=0.8, linewidth=2)
    ax.fill_between(x_axis, prior_mean_np - prior_std_np, prior_mean_np + prior_std_np, 
                    alpha=0.3, color='blue', label='Prior ±1σ')
    ax.fill_between(x_axis, var_mean_np - var_std_np, var_mean_np + var_std_np, 
                    alpha=0.3, color='red', label='Variational ±1σ')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Function Output')
    ax.set_title('Function Space Mean and Standard Deviation Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 不确定性对比
    ax = axes[0, 1]
    ax.plot(x_axis, prior_std_np, 'b-', label='Prior Uncertainty', alpha=0.8, linewidth=2)
    ax.plot(x_axis, var_std_np, 'r-', label='Variational Uncertainty', alpha=0.8, linewidth=2)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Uncertainty Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 函数输出分布
    ax = axes[1, 0]
    ax.hist(prior_mean_np, bins=30, alpha=0.7, label='Prior Output Distribution', color='blue', density=True)
    ax.hist(var_mean_np, bins=30, alpha=0.7, label='Variational Output Distribution', color='red', density=True)
    ax.set_xlabel('Function Output Value')
    ax.set_ylabel('Density')
    ax.set_title('Function Output Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 协方差矩阵对比
    ax = axes[1, 1]
    prior_cov_np = prior_cov[:, :, 0].detach().cpu().numpy()
    var_cov_np = var_cov[:, :, 0].detach().cpu().numpy()
    
    # 显示协方差矩阵的对角线和部分非对角线元素
    n_samples = min(20, len(prior_cov_np))  # 限制显示大小
    im = ax.imshow(var_cov_np[:n_samples, :n_samples] - prior_cov_np[:n_samples, :n_samples], 
                   cmap='RdBu_r', aspect='auto')
    ax.set_title('Covariance Difference Matrix (Variational - Prior)')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Function space visualization saved to: {save_path}")
    
    plt.show()
    
    # 计算实际的Function KL
    try:
        from torch.distributions import MultivariateNormal, kl_divergence
        
        cov_jitter = 1e-6
        device = prior_cov.device
        
        _prior_mean = prior_mean[:, 0].reshape(-1)
        _prior_cov = prior_cov[:, :, 0] + torch.eye(len(_prior_mean), device=device) * cov_jitter
        
        _var_mean = var_mean[:, 0].reshape(-1)
        _var_cov = var_cov[:, :, 0] + torch.eye(len(_var_mean), device=device) * cov_jitter
        
        q = MultivariateNormal(loc=_var_mean, covariance_matrix=_var_cov)
        p = MultivariateNormal(loc=_prior_mean, covariance_matrix=_prior_cov)
        function_kl = kl_divergence(q, p).item()
        
        print(f"\n=== Function Space KL Divergence ===")
        print(f"Function KL divergence: {function_kl:.2f}")
        print(f"Prior distribution statistics: mean={prior_mean_np.mean():.4f}, std={prior_std_np.mean():.4f}")
        print(f"Variational distribution statistics: mean={var_mean_np.mean():.4f}, std={var_std_np.mean():.4f}")
        
        return function_kl
        
    except Exception as e:
        print(f"Error calculating Function KL: {e}")
        return None

def quick_diagnosis(model, init_model, context_inputs):
    """
    快速诊断贝叶斯模型的主要问题
    """
    print("=== Bayesian Model Quick Diagnosis ===")
    
    # 可视化参数分布
    param_stats = visualize_parameter_distributions(model, init_model)
    
    # 可视化函数空间
    function_kl = visualize_function_space_distributions(model, init_model, context_inputs)
    
    # 诊断结论
    print("\n=== Diagnosis Results ===")
    
    if param_stats['total_kl_approx'] > 1000:
        print("⚠️  Parameter space KL divergence too large, suggestions:")
        print("   - Reduce learning rate")
        print("   - Add KL weight warm-up")
        print("   - Check parameter initialization")
    
    if function_kl and function_kl > 100:
        print("⚠️  Function space KL divergence too large, suggestions:")
        print("   - Add KL divergence clipping")
        print("   - Use parameter space KL instead of function space KL")
        print("   - Increase covariance matrix numerical stability")
    
    var_ratio = param_stats['var_var_stats']['mean'] / param_stats['prior_var_stats']['mean']
    if var_ratio > 10 or var_ratio < 0.1:
        print(f"⚠️  Abnormal variance ratio ({var_ratio:.2f}), suggest adjusting rho initialization")
    
    return param_stats, function_kl 