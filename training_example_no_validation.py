#!/usr/bin/env python3
"""
FBTCN训练示例 - 跳过验证集计算
这个脚本展示了如何在训练过程中跳过验证集计算来加快训练速度
"""

import json
import torch
from joblib import load
import torch.utils.data as Data
import os
import sys

# 添加路径
sys.path.append('剩余寿命预测模型')
from stable_fbtcn_training import model_train_stable, StabilizedAUNLL, get_stable_optimizer

def main():
    # 加载配置
    with open('config/fbtcn_config.json', 'r') as f:
        config = json.load(f)
    
    print("=== FBTCN训练配置 ===")
    print(f"是否跳过验证集: {'是' if config.get('skip_validation', False) else '否'}")
    print(f"训练轮数: {config['epochs']}")
    print(f"学习率: {config['learn_rate']}")
    print(f"KL权重: {config['kl_weight']}")
    print(f"模型通道: {config['num_channels']}")
    print()
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 这里你需要加载你的数据和模型
    # train_loader, context_loader, validation_loader = load_your_data()
    # model = create_your_model()
    
    print("📋 训练模式对比:")
    print("1. 跳过验证集 (skip_validation=True):")
    print("   ✅ 训练速度更快")
    print("   ✅ 节省计算资源") 
    print("   ✅ 专注于训练损失优化")
    print("   ❌ 无法监控过拟合")
    print("   ❌ 无法使用验证集早停")
    print()
    
    print("2. 包含验证集 (skip_validation=False):")
    print("   ✅ 可以监控过拟合")
    print("   ✅ 支持验证集早停")
    print("   ✅ 更好的模型选择")
    print("   ❌ 训练速度较慢")
    print("   ❌ 需要更多计算资源")
    print()
    
    print("💡 使用建议:")
    print("- 初期调试和快速实验: 使用 skip_validation=True")
    print("- 正式训练和模型选择: 使用 skip_validation=False")
    print("- 大数据集训练: 可以考虑间隔性验证（如每10个epoch验证一次）")
    
    # 示例训练调用
    """
    # 快速训练模式（跳过验证）
    train_losses, val_losses, best_epoch = model_train_stable(
        epochs=100, 
        model=model, 
        optimizer=optimizer, 
        loss_function=StabilizedAUNLL(), 
        train_loader=train_loader, 
        context_loader=context_loader, 
        validation_loader=validation_loader,  # 即使不使用也需要传入
        device=device, 
        config=config,
        skip_validation=True  # 关键参数
    )
    
    # 完整训练模式（包含验证）
    train_losses, val_losses, best_epoch = model_train_stable(
        epochs=100, 
        model=model, 
        optimizer=optimizer, 
        loss_function=StabilizedAUNLL(), 
        train_loader=train_loader, 
        context_loader=context_loader, 
        validation_loader=validation_loader, 
        device=device, 
        config=config,
        skip_validation=False  # 默认值
    )
    """

if __name__ == "__main__":
    main() 