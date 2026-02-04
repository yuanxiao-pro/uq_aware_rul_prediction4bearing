#!/usr/bin/env python3
"""
TCN通道数分析和建议
针对RUL预测任务的最佳实践
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_rul_data_characteristics():
    """分析RUL数据特征，指导TCN设计"""
    print("=== RUL预测数据特征分析 ===")
    
    # RUL数据特点
    characteristics = {
        "时序长度": "通常较短 (50-500个时间点)",
        "特征维度": "中等 (10-50维)",
        "数据模式": "退化趋势 + 噪声",
        "关键信息": "长期趋势 > 短期波动",
        "预测目标": "单值回归 (剩余寿命)",
    }
    
    for key, value in characteristics.items():
        print(f"{key}: {value}")
    
    print("\n基于这些特征，TCN设计建议:")
    print("1. 感受野要足够大，捕获长期退化趋势")
    print("2. 通道数不宜过多，避免过拟合")
    print("3. 重点关注特征提取而非复杂映射")

def calculate_receptive_field(num_layers, kernel_size, dilation_base=2):
    """计算TCN的感受野"""
    receptive_field = 1
    for i in range(num_layers):
        dilation = dilation_base ** i
        receptive_field += (kernel_size - 1) * dilation
    return receptive_field

def analyze_channel_configurations():
    """分析不同通道配置的特点"""
    print("\n=== TCN通道配置分析 ===")
    
    configs = [
        ([8, 16, 8], "轻量级"),
        ([16, 32, 16], "平衡型"),
        ([32, 64, 32], "标准型"),
        ([32, 64, 64, 32], "复杂型"),
        ([64, 128, 64], "重型"),
    ]
    
    input_dim = 11
    kernel_size = 3
    
    print(f"{'配置':<20} {'类型':<10} {'参数量':<10} {'感受野':<10} {'适用场景'}")
    print("-" * 70)
    
    for channels, config_type in configs:
        # 估算参数量
        params = estimate_tcn_parameters(input_dim, channels, kernel_size)
        # 计算感受野
        rf = calculate_receptive_field(len(channels), kernel_size)
        
        # 适用场景
        if params < 20000:
            scenario = "小数据集"
        elif params < 50000:
            scenario = "中等数据集"
        elif params < 100000:
            scenario = "大数据集"
        else:
            scenario = "超大数据集"
            
        print(f"{str(channels):<20} {config_type:<10} {params:<10,} {rf:<10} {scenario}")

def estimate_tcn_parameters(input_dim, channels, kernel_size):
    """估算TCN参数量"""
    total_params = 0
    prev_channels = input_dim
    
    for curr_channels in channels:
        # 每个TCN块有两个卷积层
        conv1_params = prev_channels * curr_channels * kernel_size + curr_channels
        conv2_params = curr_channels * curr_channels * kernel_size + curr_channels
        
        # downsample层（如果通道数改变）
        if prev_channels != curr_channels:
            downsample_params = prev_channels * curr_channels + curr_channels
        else:
            downsample_params = 0
            
        block_params = conv1_params + conv2_params + downsample_params
        total_params += block_params
        prev_channels = curr_channels
    
    # 输出层 (mu + sigma)
    output_params = prev_channels * 2 + 2
    total_params += output_params
    
    return total_params

def recommend_channels_for_rul(train_samples, input_dim, target_ratio=5):
    """基于数据量推荐通道配置"""
    print(f"\n=== 基于数据量的通道推荐 ===")
    print(f"训练样本数: {train_samples:,}")
    print(f"输入维度: {input_dim}")
    print(f"目标参数/样本比例: {target_ratio}:1")
    
    target_params = train_samples * target_ratio
    print(f"推荐参数量上限: {target_params:,}")
    
    # 测试不同配置
    test_configs = [
        [8, 16, 8],
        [12, 24, 12], 
        [16, 32, 16],
        [20, 40, 20],
        [24, 48, 24],
        [32, 64, 32],
    ]
    
    print(f"\n{'通道配置':<15} {'参数量':<10} {'比例':<8} {'推荐'}")
    print("-" * 45)
    
    best_config = None
    best_ratio = float('inf')
    
    for channels in test_configs:
        params = estimate_tcn_parameters(input_dim, channels, 3)
        ratio = params / train_samples
        recommend = "✓" if ratio <= target_ratio else "✗"
        
        if ratio <= target_ratio and abs(ratio - target_ratio) < abs(best_ratio - target_ratio):
            best_config = channels
            best_ratio = ratio
            
        print(f"{str(channels):<15} {params:<10,} {ratio:<8.1f} {recommend}")
    
    if best_config:
        print(f"\n推荐配置: {best_config}")
        print(f"参数/样本比例: {best_ratio:.1f}:1")
    else:
        print(f"\n警告: 所有配置都可能导致过拟合！")
        print(f"建议使用最小配置: [8, 16, 8]")

def plot_channel_scaling_analysis():
    """绘制通道数与性能的关系分析"""
    # 模拟不同通道配置的性能
    configs = [[8,16,8], [16,32,16], [24,48,24], [32,64,32], [48,96,48], [64,128,64]]
    params = [estimate_tcn_parameters(11, c, 3) for c in configs]
    
    # 模拟性能曲线（基于经验）
    train_performance = [0.85, 0.90, 0.93, 0.95, 0.97, 0.98]  # 训练性能
    val_performance = [0.83, 0.88, 0.89, 0.87, 0.82, 0.78]    # 验证性能（过拟合）
    
    plt.figure(figsize=(12, 5))
    
    # 子图1: 参数量 vs 性能
    plt.subplot(1, 2, 1)
    plt.plot(params, train_performance, 'b-o', label='train performance', linewidth=2)
    plt.plot(params, val_performance, 'r-o', label='valid performance', linewidth=2)
    plt.xlabel('params')
    plt.ylabel('R²')
    plt.title('TCN channels vs performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 标注最佳点
    best_idx = np.argmax(val_performance)
    plt.annotate(f'best config\n{configs[best_idx]}', 
                xy=(params[best_idx], val_performance[best_idx]),
                xytext=(params[best_idx]*1.2, val_performance[best_idx]+0.05),
                arrowprops=dict(arrowstyle='->', color='green'),
                color='green', fontweight='bold')
    
    # 子图2: 过拟合程度
    plt.subplot(1, 2, 2)
    overfitting = np.array(train_performance) - np.array(val_performance)
    plt.bar(range(len(configs)), overfitting, color=['green' if x < 0.05 else 'orange' if x < 0.1 else 'red' for x in overfitting])
    plt.xlabel('config no.')
    plt.ylabel('over-fitting (train-valid)')
    plt.title('over-fitting')
    plt.xticks(range(len(configs)), [str(c) for c in configs], rotation=45)
    
    # 添加安全线
    plt.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='warning')
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('tcn_channel_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def generate_config_recommendations():
    """生成具体的配置建议"""
    print("\n=== RUL预测TCN配置建议 ===")
    
    recommendations = {
        "小数据集 (<3000样本)": {
            "channels": [8, 16, 8],
            "kernel_size": 3,
            "dropout": 0.3,
            "说明": "轻量级配置，防止过拟合"
        },
        "中等数据集 (3000-8000样本)": {
            "channels": [16, 32, 16],
            "kernel_size": 3,
            "dropout": 0.2,
            "说明": "平衡配置，适合大多数RUL任务"
        },
        "大数据集 (>8000样本)": {
            "channels": [24, 48, 24],
            "kernel_size": 5,
            "dropout": 0.1,
            "说明": "增强配置，充分利用数据"
        },
        "你的当前情况 (5892样本)": {
            "channels": [12, 24, 12],
            "kernel_size": 3,
            "dropout": 0.3,
            "说明": "针对你的数据量优化的配置"
        }
    }
    
    for scenario, config in recommendations.items():
        print(f"\n{scenario}:")
        for key, value in config.items():
            print(f"  {key}: {value}")

def main():
    """主函数"""
    print("TCN通道配置分析工具 - RUL预测专用")
    print("=" * 50)
    
    # 分析RUL数据特征
    analyze_rul_data_characteristics()
    
    # 分析不同通道配置
    analyze_channel_configurations()
    
    # 基于用户数据推荐
    recommend_channels_for_rul(train_samples=5892, input_dim=11)
    
    # 生成配置建议
    generate_config_recommendations()
    
    # 绘制分析图
    plot_channel_scaling_analysis()
    
    print("\n" + "=" * 50)
    print("关键建议:")
    print("1. 对于你的5892个样本，推荐使用 [12, 24, 12] 或 [16, 32, 16]")
    print("2. 避免使用 [32, 64, 64, 32] - 参数过多会导致过拟合")
    print("3. 配合高dropout(0.3-0.5)和早停机制")
    print("4. 监控训练/验证损失差距，及时调整")

if __name__ == "__main__":
    main() 