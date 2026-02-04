"""
绘制两个预测区间（一宽一窄）
Plot Two Prediction Intervals (One Wide, One Narrow)
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_prediction_intervals(
    x=None,
    y_mean=None,
    y_lower_wide=None,
    y_upper_wide=None,
    y_lower_narrow=None,
    y_upper_narrow=None,
    figsize=(10, 6),
    show_axes=False
):
    """
    绘制两个预测区间（一宽一窄）
    
    Args:
        x: x轴数据，如果为None则自动生成
        y_mean: 预测均值（中心线）
        y_lower_wide: 宽区间的下界
        y_upper_wide: 宽区间的上界
        y_lower_narrow: 窄区间的下界
        y_upper_narrow: 窄区间的上界
        figsize: 图像大小，默认(10, 6)
        show_axes: 是否显示坐标轴，默认False
    """
    # 如果没有提供x，自动生成
    if x is None:
        x = np.linspace(0, 1, 200)
    
    # 如果没有提供y_mean，生成示例数据
    if y_mean is None:
        # 生成一个示例的预测均值曲线
        y_mean = 50 + 20 * np.sin(x / 10) + np.random.normal(0, 2, len(x))
        y_mean = np.convolve(y_mean, np.ones(10)/10, mode='same')  # 平滑
    
    # 如果没有提供区间，生成示例数据
    if y_lower_wide is None or y_upper_wide is None:
        # 宽区间：较大的不确定性
        uncertainty_wide = 0.1 + 0.05 * np.sin(x / 15)
        y_lower_wide = y_mean - uncertainty_wide
        y_upper_wide = y_mean + uncertainty_wide
    
    if y_lower_narrow is None or y_upper_narrow is None:
        # 窄区间：较小的不确定性
        uncertainty_narrow = 0.05 + 0.025 * np.sin(x / 20)
        y_lower_narrow = y_mean - uncertainty_narrow
        y_upper_narrow = y_mean + uncertainty_narrow
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制宽区间（用灰色填充）
    ax.fill_between(x, y_lower_wide, y_upper_wide, 
                    alpha=0.4, color='lightgray', 
                    label='Wide Prediction Interval', 
                    edgecolor='darkgray', linewidth=1.5)
    
    # 绘制窄区间（用浅灰色填充）
    ax.fill_between(x, y_lower_narrow, y_upper_narrow, 
                    alpha=0.5, color='gray', 
                    label='Narrow Prediction Interval',
                    edgecolor='gray', linewidth=1.5)
    
    # 绘制中心线（预测均值）
    ax.plot(x, y_mean, 'k-', linewidth=2, label='Predicted Mean', color='black', zorder=3)
    
    # 绘制 y = -x + 100 直线
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = -x_line + 1
    ax.plot(x_line, y_line, 'b--', linewidth=2, color='black', zorder=4)
    
    # 设置坐标轴
    if not show_axes:
        # 隐藏坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    else:
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('RUL', fontsize=12)
        ax.set_ylim(0, 1)  # y轴从100到0（反向）
        # ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # 设置背景为白色
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 调整布局
    plt.tight_layout()
    
    return fig, ax

def plot_simple_intervals(figsize=(10, 6), show_axes=False):
    """
    绘制简单的两个预测区间示例（一宽一窄）
    预测曲线和区间拟合 y = -x + 100 直线（从100到0）
    
    Args:
        figsize: 图像大小，默认(10, 6)
        show_axes: 是否显示坐标轴，默认False
    """
    # 生成x轴数据（0到1）
    x = np.linspace(0, 1, 100)
    
    # 预测均值等于 y = -x + 100（从100到0的直线）
    y_mean = -x + 1
    
    # 宽区间：较大的不确定性（标准差较大）
    uncertainty_wide = 0.1 + 0.05 * np.sin(x / 10)
    y_lower_wide = y_mean - uncertainty_wide
    y_upper_wide = y_mean + uncertainty_wide
    
    # 窄区间：较小的不确定性（标准差较小）
    uncertainty_narrow = 0.05 + 0.025 * np.sin(x / 1.5)
    y_lower_narrow = y_mean - uncertainty_narrow
    y_upper_narrow = y_mean + uncertainty_narrow
    
    return plot_prediction_intervals(
        x=x,
        y_mean=y_mean,
        y_lower_wide=y_lower_wide,
        y_upper_wide=y_upper_wide,
        y_lower_narrow=y_lower_narrow,
        y_upper_narrow=y_upper_narrow,
        figsize=figsize,
        show_axes=show_axes
    )

if __name__ == "__main__":
    # 绘制简单的两个预测区间示例（显示坐标轴）
    fig, ax = plot_simple_intervals(show_axes=True)
    
    # 保存图片
    plt.savefig('prediction_intervals.svg', dpi=1000, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='svg')
    print("图片已保存为: prediction_intervals.svg")
    
    # 显示图片
    plt.show()

