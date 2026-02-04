"""
绘制高斯分布（无坐标轴）
Plot Gaussian Distribution (No Axes)
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_gaussian(mean=0, std=1, x_range=(-10, 10), num_points=1000, figsize=(8, 6)):
    """
    绘制高斯分布曲线
    
    Args:
        mean: 均值，默认0
        std: 标准差，默认1
        x_range: x轴范围，默认(-4, 4)
        num_points: 采样点数，默认1000
        figsize: 图像大小，默认(8, 6)
    """
    # 生成x值
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    # 计算高斯分布的概率密度函数
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制曲线
    ax.plot(x, y, 'b-', linewidth=2.5, color='black')
    
    # 填充曲线下方区域（可选）
    ax.fill_between(x, y, alpha=0.3, color='white')
    
    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # 设置背景为白色
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 调整布局
    plt.tight_layout()
    
    return fig, ax

if __name__ == "__main__":
    # 绘制标准高斯分布（均值0，标准差1）
    fig, ax = plot_gaussian(mean=0, std=8)
    
    # 保存图片
    plt.savefig('gaussian_distribution.png', dpi=1000, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='png')
    print("图片已保存为: gaussian_distribution.png")
    
    # 显示图片
    plt.show()

