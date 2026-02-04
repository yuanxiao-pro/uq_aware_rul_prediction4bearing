import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from DE import DE

def load_bearing_data(bearing_name, data_dir='datasetresult/xjtu'):
    """
    加载轴承数据
    
    Args:
        bearing_name: 轴承名称，如 'c1_Bearing1_1'
        data_dir: 数据目录
    
    Returns:
        features_df: 特征数据DataFrame
        labeled_df: 带标签数据DataFrame
    """
    # 查找对应的文件
    all_files = os.listdir(data_dir)
    
    # 查找特征文件
    # features_files = [f for f in all_files if bearing_name in f and f.endswith('_features_df')]
    # labeled_files = [f for f in all_files if bearing_name in f and f.endswith('_labeled.csv')]
    
    features_files = [f for f in all_files]
    labeled_files = [f for f in all_files]
    

    if not features_files:
        print(f"Features file not found for {bearing_name}")
        return None, None
    
    if not labeled_files:
        print(f"Labels file not found for {bearing_name}")
        return None, None
    
    # 加载特征数据
    features_path = os.path.join(data_dir, features_files[0])
    features_df = load(features_path)
    
    # 加载标签数据
    labeled_path = os.path.join(data_dir, labeled_files[0])
    labeled_df = pd.read_csv(labeled_path)
    
    print(f"Successfully loaded {bearing_name} data:")
    print(f"  Features data shape: {features_df.shape}")
    print(f"  Labels data shape: {labeled_df.shape}")
    
    return features_df, labeled_df


def calculate_de_for_features(features_df, m=4, epsilon=30):
    """
    为每个特征计算DE值
    
    Args:
        features_df: 特征DataFrame
        m: DE参数m
        epsilon: DE参数epsilon
    
    Returns:
        de_values: 每个特征的DE值字典
    """
    de_values = {}
    
    print(f"\nCalculating DE values (m={m}, epsilon={epsilon}):")
    
    for column in features_df.columns:
        data = features_df[column].values
        de_value = DE(data, m=m, epsilon=epsilon)
        de_values[column] = de_value
        print(f"  {column}: {de_value:.6f}")
    
    return de_values


def visualize_bearing_data_and_de(features_df, labeled_df, de_values, bearing_name):
    """
    可视化轴承数据和DE值
    
    Args:
        features_df: 特征DataFrame
        labeled_df: 标签DataFrame
        de_values: DE值字典
        bearing_name: 轴承名称
    """
    # 创建图形
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(f'{bearing_name} Feature Data and DE Analysis', fontsize=16, fontweight='bold')
    
    # 获取特征名称
    feature_names = list(features_df.columns)
    
    # 绘制每个特征的时间序列
    for i, feature_name in enumerate(feature_names):
        row = i // 3
        col = i % 3
        
        ax = axes[row, col]
        
        # 绘制特征时间序列
        ax.plot(features_df[feature_name].values, 'b-', linewidth=1, alpha=0.7)
        
        # 设置标题，包含DE值
        de_value = de_values[feature_name]
        ax.set_title(f'{feature_name}\nDE = {de_value:.6f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Feature Value')
        ax.grid(True, alpha=0.3)
        
        # 设置颜色，DE值越高颜色越红
        max_de = max(de_values.values())
        min_de = min(de_values.values())
        if max_de > min_de:
            color_intensity = (de_value - min_de) / (max_de - min_de)
            ax.spines['top'].set_color((1, 1-color_intensity, 1-color_intensity))
            ax.spines['bottom'].set_color((1, 1-color_intensity, 1-color_intensity))
            ax.spines['left'].set_color((1, 1-color_intensity, 1-color_intensity))
            ax.spines['right'].set_color((1, 1-color_intensity, 1-color_intensity))
            ax.spines['top'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(f'{bearing_name}_features_de_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建DE值柱状图
    plt.figure(figsize=(12, 8))
    
    feature_names = list(de_values.keys())
    de_vals = list(de_values.values())
    
    # 创建颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
    
    bars = plt.bar(range(len(feature_names)), de_vals, color=colors)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, de_vals)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{val:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Feature Names', fontsize=12, fontweight='bold')
    plt.ylabel('Diversity Entropy (DE)', fontsize=12, fontweight='bold')
    plt.title(f'{bearing_name} Diversity Entropy (DE) Values for Each Feature', fontsize=14, fontweight='bold')
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    mean_de = np.mean(de_vals)
    std_de = np.std(de_vals)
    plt.axhline(y=mean_de, color='red', linestyle='--', alpha=0.7, 
                label=f'Mean: {mean_de:.6f}')
    plt.axhline(y=mean_de + std_de, color='orange', linestyle=':', alpha=0.7, 
                label=f'Mean+Std: {mean_de + std_de:.6f}')
    plt.axhline(y=mean_de - std_de, color='orange', linestyle=':', alpha=0.7, 
                label=f'Mean-Std: {mean_de - std_de:.6f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{bearing_name}_de_values_bar.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 如果有标签数据，绘制RUL曲线
    if 'RUL' in labeled_df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(labeled_df['RUL'].values, 'r-', linewidth=2, label='RUL')
        plt.xlabel('Time Steps', fontsize=12, fontweight='bold')
        plt.ylabel('Remaining Useful Life (RUL)', fontsize=12, fontweight='bold')
        plt.title(f'{bearing_name} Remaining Useful Life Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{bearing_name}_rul_curve.png', dpi=300, bbox_inches='tight')
        plt.show()


def analyze_de_correlation(features_df, labeled_df, de_values):
    """
    分析DE值与数据特性的相关性
    
    Args:
        features_df: 特征DataFrame
        labeled_df: 标签DataFrame  
        de_values: DE值字典
    """
    print("\n=== DE Analysis Report ===")
    
    # 基本统计
    de_vals = list(de_values.values())
    print(f"DE Statistics:")
    print(f"  Mean: {np.mean(de_vals):.6f}")
    print(f"  Standard Deviation: {np.std(de_vals):.6f}")
    print(f"  Maximum: {np.max(de_vals):.6f} ({list(de_values.keys())[np.argmax(de_vals)]})")
    print(f"  Minimum: {np.min(de_vals):.6f} ({list(de_values.keys())[np.argmin(de_vals)]})")
    
    # 特征变异性分析
    print(f"\nFeature Variability Analysis:")
    for feature_name, de_value in de_values.items():
        data = features_df[feature_name].values
        cv = np.std(data) / np.mean(data) if np.mean(data) != 0 else 0  # 变异系数
        print(f"  {feature_name}: DE={de_value:.6f}, CV={cv:.6f}")
    
    # DE值排序
    sorted_de = sorted(de_values.items(), key=lambda x: x[1], reverse=True)
    print(f"\nDE Values Ranking (High to Low):")
    for i, (feature, de_val) in enumerate(sorted_de, 1):
        print(f"  {i}. {feature}: {de_val:.6f}")


def main():
    """主函数"""
    print("=== Bearing1_1 DE Analysis ===")
    
    # 加载数据
    # bearing_name = 'c1_Bearing1_1'
    bearing_name = 'Bearing1_1'

    features_df, labeled_df = load_bearing_data(bearing_name, data_dir='datasetresult/xjtu/XJTU-SY_Bearing_Datasets/Bearing1_1')
    
    if features_df is None:
        print("Data loading failed, exiting program")
        return
    
    # 计算DE值
    de_values = calculate_de_for_features(features_df, m=4, epsilon=30)
    
    # 可视化
    visualize_bearing_data_and_de(features_df, labeled_df, de_values, bearing_name)
    
    # 分析报告
    analyze_de_correlation(features_df, labeled_df, de_values)
    
    print(f"\nAnalysis completed! Generated image files:")
    print(f"  - {bearing_name}_features_de_analysis.png")
    print(f"  - {bearing_name}_de_values_bar.png")
    print(f"  - {bearing_name}_rul_curve.png")


if __name__ == "__main__":
    main() 