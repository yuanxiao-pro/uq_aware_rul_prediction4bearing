"""
数据可视化模块
读取CSV文件并可视化每个特征的时间序列
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# 导入字体配置工具
try:
    from matplotlib_chinese_config import setup_chinese_font
    # 设置中文字体为宋体，西文字体为Times New Roman
    setup_chinese_font(chinese_font_name='SimSun', western_font_name='Times New Roman')
except ImportError:
    # 如果导入失败，使用本地配置
    def setup_chinese_font():
        """配置matplotlib以支持中文显示（宋体）和西文（Times New Roman）"""
        available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
        
        # 中文字体：宋体
        chinese_font_list = ['SimSun', 'NSimSun', 'STSong', 'Songti SC']
        chinese_font = None
        for font in chinese_font_list:
            if font in available_fonts:
                chinese_font = font
                break
        
        # 西文字体：Times New Roman
        western_font_list = ['Times New Roman', 'Times', 'DejaVu Serif']
        western_font = None
        for font in western_font_list:
            if font in available_fonts:
                western_font = font
                break
        
        if chinese_font:
            plt.rcParams['font.sans-serif'] = [chinese_font] + plt.rcParams['font.sans-serif']
            print(f"✓ 中文字体已设置: {chinese_font}")
        if western_font:
            plt.rcParams['font.serif'] = [western_font] + plt.rcParams['font.serif']
            plt.rcParams['mathtext.fontset'] = 'stix'
            print(f"✓ 西文字体已设置: {western_font}")
        
        plt.rcParams['axes.unicode_minus'] = False
        return chinese_font, western_font
    
    setup_chinese_font()

def visualize_features(csv_path='datasetresult/xjtu/Bearing1_1_labeled_fpt.csv'):
    """
    读取CSV文件并可视化每个特征
    
    Args:
        csv_path: CSV文件路径
    """
    # 检查文件路径是否存在
    if not os.path.exists(csv_path):
        # 尝试使用绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        abs_csv_path = os.path.join(script_dir, csv_path)
        if os.path.exists(abs_csv_path):
            csv_path = abs_csv_path
            print(f"Using absolute path: {csv_path}")
        else:
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}\n"
                f"Also tried: {abs_csv_path}\n"
                f"Current working directory: {os.getcwd()}"
            )
    
    # 读取CSV文件
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 排除rul列（如果存在）
    feature_columns = [col for col in df.columns if col != 'rul']
    
    if len(feature_columns) == 0:
        print("No feature columns found!")
        return
    
    print(f"Found {len(feature_columns)} features: {feature_columns}")
    print(f"Data shape: {df.shape}")
    
    # 计算子图布局（尽量接近正方形）
    n_features = len(feature_columns)
    n_cols = int(np.ceil(np.sqrt(n_features)))
    n_rows = int(np.ceil(n_features / n_cols))
    
    # 创建图形
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle('Feature Visualization: Bearing1_1', fontsize=16, fontweight='bold')
    
    # 如果只有一个子图，axes不是数组
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    # 为每个特征创建子图
    for idx, feature_name in enumerate(feature_columns):
        ax = axes[idx]
        
        # 获取特征数据
        feature_data = df[feature_name].values
        
        # 创建时间索引（样本索引）
        time_index = np.arange(len(feature_data))
        
        # 绘制时间序列
        ax.plot(time_index, feature_data, linewidth=6, alpha=0.7)
        ax.set_title(feature_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Sample Index', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_val = np.mean(feature_data)
        std_val = np.std(feature_data)
        ax.axhline(mean_val, color='r', linestyle='--', alpha=0.5, linewidth=1, label=f'Mean: {mean_val:.3f}')
        ax.legend(loc='best', fontsize=8)
    
    # 隐藏多余的子图
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = csv_path.replace('.csv', '_visualization.svg')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='svg')
    print(f"\nVisualization saved to: {output_path}")
    
    # 显示图形
    plt.show()
    
    return fig, axes


if __name__ == "__main__":
    # 默认文件路径
    csv_file = 'datasetresult/xjtu/Bearing1_1_labeled_fpt.csv'
    
    # 可视化特征
    visualize_features(csv_file)
