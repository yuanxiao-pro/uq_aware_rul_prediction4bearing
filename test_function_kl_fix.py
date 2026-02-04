#!/usr/bin/env python3
"""
测试修复后的Function KL计算稳定性
"""

import sys
sys.path.append('剩余寿命预测模型')

import torch
import torch.nn as nn
from bayesian_torch.layers import Conv1dReparameterization, LinearReparameterization
from function_kl import get_bayesian_model_mu_rho, calculate_function_kl, ensure_positive_definite
import numpy as np

# 简化的BayesianTCN模型用于测试
class SimpleBayesianModel(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=64, output_dim=1):
        super().__init__(self)
        self.feature = LinearReparameterization(
            input_dim, hidden_dim,
            prior_mean=0, prior_variance=1, 
            posterior_mu_init=0, posterior_rho_init=-3
        )
        self.mu = LinearReparameterization(
            hidden_dim, output_dim,
            prior_mean=0, prior_variance=1,
            posterior_mu_init=0, posterior_rho_init=-3
        )
    
    def forward(self, x, feature=False):
        x = x.view(x.size(0), -1)
        feat, _ = self.feature(x)
        mu, _ = self.mu(feat)
        if feature:
            return mu, None, 0.0, feat
        return mu, None, 0.0
    
    def generate_init_params(self, sample_input):
        with torch.no_grad():
            _ = self.forward(sample_input)
            return {k: v.clone() for k, v in self.state_dict().items()}

def test_positive_definite_function():
    """测试正定性检查函数"""
    print("=== 测试正定性检查函数 ===")
    
    # 测试1: 已经正定的矩阵
    pos_def_matrix = torch.eye(3) * 2.0
    result1 = ensure_positive_definite(pos_def_matrix)
    print(f"测试1 - 正定矩阵: 输入特征值 {torch.linalg.eigvals(pos_def_matrix)}")
    print(f"         输出特征值 {torch.linalg.eigvals(result1)}")
    
    # 测试2: 半正定矩阵（有零特征值）
    semi_pos_def = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    result2 = ensure_positive_definite(semi_pos_def)
    print(f"测试2 - 半正定矩阵: 输入特征值 {torch.linalg.eigvals(semi_pos_def)}")
    print(f"           输出特征值 {torch.linalg.eigvals(result2)}")
    
    # 测试3: 负定矩阵
    neg_def = torch.tensor([[-2.0, 0.5], [0.5, -1.0]])
    result3 = ensure_positive_definite(neg_def)
    print(f"测试3 - 负定矩阵: 输入特征值 {torch.linalg.eigvals(neg_def)}")
    print(f"         输出特征值 {torch.linalg.eigvals(result3)}")
    
    # 验证所有结果都是正定的
    for i, result in enumerate([result1, result2, result3], 1):
        try:
            torch.linalg.cholesky(result)
            print(f"✓ 测试{i}: 修复后的矩阵是正定的")
        except:
            print(f"✗ 测试{i}: 修复后的矩阵仍然不是正定的")
    print()

def test_function_kl_stability():
    """测试Function KL计算稳定性"""
    print("=== 测试Function KL计算稳定性 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试模型
    model = SimpleBayesianModel().to(device)
    
    # 创建测试数据
    batch_sizes = [16, 32, 64]
    success_count = 0
    total_tests = 0
    
    for batch_size in batch_sizes:
        print(f"\n测试批次大小: {batch_size}")
        
        for test_i in range(10):  # 每个批次大小测试10次
            total_tests += 1
            
            # 生成随机输入
            test_input = torch.randn(batch_size, 11).to(device)
            
            try:
                # 获取模型参数
                params_mean, params_logvar = get_bayesian_model_mu_rho(model)
                
                # 计算Function KL
                function_kl = calculate_function_kl(
                    params_mean, params_logvar, test_input, model=model
                )
                
                # 检查结果
                if torch.isnan(function_kl) or torch.isinf(function_kl):
                    print(f"  测试 {test_i+1}: ✗ 结果无效 ({function_kl})")
                elif function_kl < 0:
                    print(f"  测试 {test_i+1}: ⚠ 负值 ({function_kl:.6f})")
                    success_count += 0.5  # 部分成功
                else:
                    print(f"  测试 {test_i+1}: ✓ 成功 ({function_kl:.6f})")
                    success_count += 1
                    
            except Exception as e:
                print(f"  测试 {test_i+1}: ✗ 异常 - {str(e)[:50]}...")
    
    success_rate = success_count / total_tests * 100
    print(f"\n总体成功率: {success_rate:.1f}% ({success_count}/{total_tests})")
    
    if success_rate > 80:
        print("🎉 Function KL计算稳定性测试通过！")
    elif success_rate > 50:
        print("⚠️  Function KL计算部分稳定，但仍有改进空间")
    else:
        print("❌ Function KL计算仍然不稳定")

def test_covariance_properties():
    """测试协方差矩阵的数学性质"""
    print("\n=== 测试协方差矩阵数学性质 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleBayesianModel().to(device)
    
    # 小批次测试，便于检查
    test_input = torch.randn(8, 11).to(device)
    params_mean, params_logvar = get_bayesian_model_mu_rho(model)
    
    # 导入内部函数进行测试
    from function_kl import calculate_moments
    
    try:
        # 计算协方差矩阵
        _, cov_matrix = calculate_moments(model, params_mean, params_logvar, test_input)
        cov_2d = cov_matrix[:, :, 0]  # 提取2D矩阵
        
        print(f"协方差矩阵形状: {cov_2d.shape}")
        print(f"协方差矩阵对角线最小值: {torch.diag(cov_2d).min():.6f}")
        print(f"协方差矩阵对角线最大值: {torch.diag(cov_2d).max():.6f}")
        
        # 检查对称性
        is_symmetric = torch.allclose(cov_2d, cov_2d.t(), rtol=1e-5)
        print(f"矩阵对称性: {'✓' if is_symmetric else '✗'}")
        
        # 检查正定性
        eigenvals = torch.linalg.eigvals(cov_2d).real
        min_eigenval = eigenvals.min()
        print(f"最小特征值: {min_eigenval:.6f}")
        print(f"正定性: {'✓' if min_eigenval > 1e-8 else '✗'}")
        
        # 尝试Cholesky分解
        try:
            torch.linalg.cholesky(cov_2d)
            print("Cholesky分解: ✓")
        except:
            print("Cholesky分解: ✗")
            
    except Exception as e:
        print(f"协方差矩阵计算失败: {e}")

if __name__ == "__main__":
    print("开始测试修复后的Function KL计算...")
    print("=" * 50)
    
    # 运行所有测试
    test_positive_definite_function()
    test_function_kl_stability()
    test_covariance_properties()
    
    print("\n" + "=" * 50)
    print("测试完成！") 