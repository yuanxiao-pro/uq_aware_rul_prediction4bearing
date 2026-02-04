#!/usr/bin/env python3
"""
分布可视化工具测试脚本
"""

import torch
import numpy as np
import json
from distribution_visualizer import quick_diagnosis, visualize_parameter_distributions, visualize_function_space_distributions

def test_distribution_visualization():
    """
    测试分布可视化工具
    """
    print("=== 测试贝叶斯分布可视化工具 ===")
    
    # 1. 加载配置
    with open('config/fbtcn_config.json', 'r') as f:
        config = json.load(f)
    
    # 2. 创建模型（需要导入你的BayesianTCN）
    try:
        # 这里需要根据你的实际模型路径调整
        import sys
        sys.path.append('剩余寿命预测模型')
        from stable_function_kl import BayesianTCN  # 假设这是你的模型类
        
        # 创建模型
        model = BayesianTCN(
            input_size=config['input_dim'],
            output_size=config['output_dim'],
            num_channels=config['num_channels'],
            kernel_size=config['kernel_size'],
            dropout=config['dropout']
        )
        
        # 创建初始模型（用作先验）
        init_model = BayesianTCN(
            input_size=config['input_dim'],
            output_size=config['output_dim'],
            num_channels=config['num_channels'],
            kernel_size=config['kernel_size'],
            dropout=config['dropout']
        )
        
        # 3. 创建测试数据
        batch_size = 32
        window_size = config.get('window_size', 1)
        input_dim = config['input_dim']
        
        # 模拟输入数据
        context_inputs = torch.randn(batch_size, input_dim, window_size)
        
        # 4. 运行可视化
        print("开始可视化分析...")
        
        # 参数分布可视化
        print("\n1. 参数分布可视化:")
        param_stats = visualize_parameter_distributions(
            model, init_model, 
            save_path='parameter_distributions.png'
        )
        
        # 函数空间分布可视化
        print("\n2. 函数空间分布可视化:")
        function_kl = visualize_function_space_distributions(
            model, init_model, context_inputs,
            save_path='function_space_distributions.png'
        )
        
        # 综合诊断
        print("\n3. 综合诊断:")
        param_stats, function_kl = quick_diagnosis(model, init_model, context_inputs)
        
        return param_stats, function_kl
        
    except ImportError as e:
        print(f"模型导入失败: {e}")
        print("请确保BayesianTCN模型类可以正确导入")
        return None, None
    except Exception as e:
        print(f"测试过程中出错: {e}")
        return None, None

def test_with_existing_models(model_path=None, init_model_path=None):
    """
    使用已存在的模型进行测试
    
    Args:
        model_path: 训练后的模型路径
        init_model_path: 初始模型路径
    """
    print("=== 使用已存在模型测试 ===")
    
    if model_path and init_model_path:
        try:
            # 加载模型
            model = torch.load(model_path)
            init_model = torch.load(init_model_path)
            
            # 创建测试数据
            context_inputs = torch.randn(16, 11, 1)  # 根据你的实际输入维度调整
            
            # 运行诊断
            param_stats, function_kl = quick_diagnosis(model, init_model, context_inputs)
            
            return param_stats, function_kl
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None, None
    else:
        print("请提供模型路径")
        return None, None

if __name__ == "__main__":
    # 运行测试
    param_stats, function_kl = test_distribution_visualization()
    
    if param_stats:
        print("\n=== 测试完成 ===")
        print(f"参数空间KL近似: {param_stats['total_kl_approx']:.2f}")
        if function_kl:
            print(f"函数空间KL: {function_kl:.2f}")
    else:
        print("测试失败，请检查模型导入") 