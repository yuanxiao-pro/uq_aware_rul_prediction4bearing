#!/usr/bin/env python3
"""
绘制不确定性分析和RUL分布的可视化图表

包含4个子图（2x2布局）：
1. AU和EU随时间变化的折线图
2. AU vs EU的散点分布图（包含平均值线）
3. 误差（|y_true - y_pred_mean|）与EU的关系散点图
4. RUL分布的高斯概率密度图（基于200次前向传播的结果）
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Tuple, Optional
from scipy.stats import norm


def plot_time_series(ax, y_pred_alea: np.ndarray, y_pred_epi: np.ndarray, 
                     sample_indices: np.ndarray) -> None:
    """
    绘制AU和EU随时间变化的折线图
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        要绘制的坐标轴对象
    y_pred_alea : np.ndarray
        AU (Aleatoric Uncertainty) 数据
    y_pred_epi : np.ndarray
        EU (Epistemic Uncertainty) 数据
    sample_indices : np.ndarray
        样本索引（x轴数据）
    """
    # 创建混合字体属性（中文用宋体，西文用新罗马）
    font_chinese = FontProperties(family='SimSun', size=42, weight='bold')
    font_english = FontProperties(family='Times New Roman', size=34)
    
    # 将sample_indices从1开始（而不是0）
    sample_indices_1based = sample_indices + 1
    
    # 绘制折线图（去掉端点，直接使用AU，不取对数）
    ax.plot(sample_indices_1based, y_pred_alea, label='Aleatoric(log-variance)', 
            linewidth=3, color='#4A90A4')
    ax.plot(sample_indices_1based, y_pred_epi, label='Epistemic', 
            linewidth=3, color='#8B4C8C')
    
    # 设置x轴刻度：从1开始，标记最后一个样本索引，并添加多个刻度
    n_samples = len(sample_indices)
    last_index = n_samples  # 因为从1开始，所以最后一个索引是n_samples
    # 设置x轴刻度，包括1、中间值和最后一个索引
    # 在起止节点左右留白：左边留出更多空间，右边也留出更多空间
    ax.set_xlim(left=-0.5, right=last_index + 1.5)  # 增加左右留白
    # 计算多个刻度位置（分成5个等间距的刻度）
    num_ticks = 5
    tick_positions = np.linspace(1, last_index, num_ticks)
    tick_positions = np.round(tick_positions).astype(int)
    # 确保包含1和最后一个索引
    tick_positions = np.unique(np.concatenate([[1], tick_positions, [last_index]]))
    # 设置主要刻度
    ax.set_xticks(tick_positions)
    
    # 设置标签和标题（坐标轴字号42，图例字号22，加粗）
    # 使用FontProperties设置更粗的字体
    label_font = FontProperties(family='Times New Roman', size=42, weight='black')
    ax.set_xlabel('Time(10 s)', fontproperties=label_font)
    ax.set_ylabel('Uncertainty Component', fontproperties=label_font)
    ax.tick_params(axis='both', which='major', labelsize=42)
    # 确保刻度标签不加粗
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_weight('normal')
    # 标题：中文用宋体，字母用新罗马
    ax.set_title('(a) AU/EU协同演进', fontproperties=font_chinese)
    
    # 添加图例（字号22）
    ax.legend(fontsize=34, loc='best', prop=font_english)
    
    # 设置框线宽度为2
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')


def plot_scatter_distribution(ax, y_pred_alea: np.ndarray, y_pred_epi: np.ndarray,
                              sample_indices: np.ndarray) -> None:
    """
    绘制AU vs EU的散点分布图，包含平均值线
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        要绘制的坐标轴对象
    y_pred_alea : np.ndarray
        AU (Aleatoric Uncertainty) 数据
    y_pred_epi : np.ndarray
        EU (Epistemic Uncertainty) 数据
    sample_indices : np.ndarray
        样本索引（用于颜色映射）
    """
    # 创建混合字体属性
    font_chinese = FontProperties(family='SimSun', size=42, weight='bold')
    font_english = FontProperties(family='Times New Roman', size=34)
    
    # 计算平均值
    au_mean = np.mean(y_pred_alea)
    eu_mean = np.mean(y_pred_epi)
    
    # 使用时间作为颜色映射，展示随时间的变化（参考提供的代码）
    scatter = ax.scatter(y_pred_epi, y_pred_alea, c=sample_indices, cmap='viridis', 
                         alpha=0.8, s=60, edgecolors='face', linewidth=3)
    
    # 添加颜色条（表示时间，参考提供的代码参数）
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.ax.tick_params(labelsize=42)
    # 设置颜色条刻度：从1开始，标记起止索引，并添加多个刻度
    n_samples = len(sample_indices)
    # sample_indices是从0开始的，但颜色条应该显示从1开始的索引
    # 颜色条的取值范围是[0, n_samples-1]，需要映射到[1, n_samples]
    cbar_min = 0  # sample_indices的最小值
    cbar_max = n_samples - 1  # sample_indices的最大值
    # 计算多个刻度位置（分成5个等间距的刻度）
    num_ticks = 5
    cbar_ticks = np.linspace(cbar_min, cbar_max, num_ticks)
    # 计算对应的索引值（从1开始）
    tick_indices = np.round(cbar_ticks + 1).astype(int)
    # 确保包含1和最后一个索引
    cbar_ticks_final = np.unique(np.concatenate([[cbar_min], cbar_ticks, [cbar_max]]))
    tick_indices_final = np.unique(np.concatenate([[1], tick_indices, [n_samples]]))
    # 设置颜色条的刻度
    cbar.set_ticks(cbar_ticks_final)
    # 设置刻度标签：从1开始
    cbar.set_ticklabels([str(idx) for idx in tick_indices_final])
    
    # 添加EU平均值垂直虚线（紫色，参考提供的代码）
    ax.axvline(x=eu_mean, color='purple', linestyle='--', linewidth=3, 
               label=f'Mean Epi: {eu_mean:.4f}', alpha=0.8)
    
    # 添加AU平均值水平虚线（橙色，参考提供的代码）
    ax.axhline(y=au_mean, color='orange', linestyle='--', linewidth=3, 
               label=f'Mean Alea: {au_mean:.3f}', alpha=0.8)
    
    # 设置标签和标题（坐标轴字号42，图例字号22，加粗）
    # 使用FontProperties设置更粗的字体
    label_font = FontProperties(family='Times New Roman', size=42, weight='black')
    ax.set_xlabel('EU', fontproperties=label_font)
    ax.set_ylabel('AU', fontproperties=label_font)
    ax.tick_params(axis='both', which='major', labelsize=42)
    # 确保刻度标签不加粗
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_weight('normal')
    # 标题：中文用宋体，字母用新罗马
    ax.set_title('(b) AU/EU聚类', fontproperties=font_chinese)
    
    # 添加图例（参考提供的代码样式，字号22）
    ax.legend(fontsize=34, frameon=True, fancybox=True, shadow=False, loc='upper right', prop=font_english)
    
    # 设置框线宽度为2
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')


def plot_error_vs_epi(ax, y_true: np.ndarray, y_pred_mean: np.ndarray, 
                      y_pred_epi: np.ndarray, sample_indices: np.ndarray) -> None:
    """
    绘制误差（|y_true - y_pred_mean|）与认知不确定性（y_pred_epi）的关系散点图
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        要绘制的坐标轴对象
    y_true : np.ndarray
        真实值
    y_pred_mean : np.ndarray
        预测均值
    y_pred_epi : np.ndarray
        EU (Epistemic Uncertainty) 数据
    sample_indices : np.ndarray
        样本索引（用于颜色映射）
    """
    # 创建混合字体属性
    font_chinese = FontProperties(family='SimSun', size=42, weight='bold')
    font_english = FontProperties(family='Times New Roman', size=30)
    
    # 计算绝对误差
    absolute_error = np.abs(y_true - y_pred_mean)
    
    # 使用时间作为颜色映射，展示随时间的变化（改为viridis，与子图2一致，不显示Data points图例）
    # 横纵轴对调：EU在横轴，Error在纵轴
    scatter = ax.scatter(y_pred_epi, absolute_error, c=sample_indices, cmap='viridis', 
                         s=50, alpha=0.6, edgecolors='black', linewidth=3)
    
    # 计算平均值
    error_mean = np.mean(absolute_error)
    epi_mean = np.mean(y_pred_epi)
    
    # 添加EU平均值垂直虚线（横轴）
    ax.axvline(x=epi_mean, color='#800080', linestyle='--', linewidth=3, 
               label=f'EU mean = {epi_mean:.6f}', alpha=0.8)
    
    # 添加误差平均值水平虚线（纵轴）
    ax.axhline(y=error_mean, color='#d62728', linestyle='--', linewidth=3, 
               label=f'Error mean = {error_mean:.6f}', alpha=0.8)
    
    # 设置标签和标题（坐标轴字号42，图例字号22，加粗）
    # 使用FontProperties设置更粗的字体
    label_font = FontProperties(family='Times New Roman', size=42, weight='black')
    ax.set_xlabel('EU', fontproperties=label_font)
    ax.set_ylabel('Error', fontproperties=label_font)
    ax.tick_params(axis='both', which='major', labelsize=42)
    # 确保刻度标签不加粗
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_weight('normal')
    # 标题：中文用宋体，字母用新罗马
    ax.set_title('(c) Error/EU聚类', fontproperties=font_chinese)
    
    # 添加颜色条（表示时间，不显示名称）
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.tick_params(labelsize=42)
    
    # 添加图例（字号22）
    ax.legend(fontsize=30, loc='upper left', framealpha=0.9, prop=font_english)
    
    # 设置框线宽度为2
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')


def load_rul_distribution_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载RUL分布CSV数据，提取y_true, y_pred_samples, 和 y_var
    
    Parameters:
    -----------
    csv_path : str
        CSV文件路径，格式：第一列是y_true，第2-201列是y_pred_sample_0到y_pred_sample_199，最后一列是y_var
    
    Returns:
    --------
    y_true : np.ndarray
        真实RUL值 [n_timepoints]
    y_pred_samples : np.ndarray
        预测RUL样本 [n_timepoints, n_samples=200]
    y_var : np.ndarray
        方差值 [n_timepoints]
    """
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    
    # First column is y_true
    y_true = data[:, 0]
    
    # Last column is y_var
    y_var = data[:, -1]
    
    # Columns 1 to 200 are y_pred_sample_0 to y_pred_sample_199
    y_pred_samples = data[:, 1:201]
    
    return y_true, y_pred_samples, y_var


def plot_rul_distribution(ax, y_true: float, y_pred_mean: float, y_pred_std: float,
                         time_idx: int) -> None:
    """
    在给定的坐标轴上绘制RUL的高斯分布图
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        要绘制的坐标轴对象
    y_true : float
        真实RUL值
    y_pred_mean : float
        预测RUL的均值
    y_pred_std : float
        预测RUL的标准差
    time_idx : int
        时间点索引（用于标题）
    """
    # Create x-axis range for PDF (0 to 1 for RUL)
    x_min = 0
    x_max = 1
    x = np.linspace(x_min, x_max, 1000)
    
    # Compute Gaussian PDF
    pdf = norm.pdf(x, loc=y_pred_mean, scale=y_pred_std)
    
    # Plot PDF as filled area (shaded，不显示图例)
    ax.fill_between(x, pdf, alpha=0.3, color='blue')
    
    # Plot PDF line
    ax.plot(x, pdf, 'b-', linewidth=3, alpha=0.7)
    
    # Draw vertical line for true RUL（改为Target）
    ax.axvline(y_true, color='red', linestyle='--', linewidth=3, 
               label=f'Target:{y_true:.3f}')
    
    # Draw vertical line for mean RUL（改为Prediction）
    ax.axvline(y_pred_mean, color='green', linestyle='--', linewidth=3, 
               label=f'Prediction:{y_pred_mean:.3f}')
    
    # 创建混合字体属性
    font_chinese = FontProperties(family='SimSun', size=42, weight='bold')
    font_english = FontProperties(family='Times New Roman', size=28)
    
    # Labels and title（坐标轴字号42，图例字号22，加粗）
    # 使用FontProperties设置更粗的字体
    label_font = FontProperties(family='Times New Roman', size=42, weight='black')
    ax.set_xlabel('RUL', fontproperties=label_font)
    ax.set_ylabel('PDF', fontproperties=label_font)
    ax.tick_params(axis='both', which='major', labelsize=42)
    # 确保刻度标签不加粗
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_weight('normal')
    # 标题：中文用宋体，字母用新罗马
    ax.set_title(f'(c) 样本索引{time_idx}预测分布', fontproperties=font_chinese)
    # 将图例位置固定在左上角，确保在子图内部（字号28）
    ax.legend(loc='upper left', fontsize=28, prop=font_english, framealpha=0.9)
    
    # Set x-axis ticks with interval of 0.2
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    
    # 设置框线宽度为2
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    ax.grid(True, alpha=0.3, linestyle='--')


def print_statistics(y_pred_alea: np.ndarray, y_pred_epi: np.ndarray, 
                     y_true: np.ndarray = None, y_pred_mean: np.ndarray = None) -> None:
    """
    打印数据统计信息
    
    Parameters:
    -----------
    y_pred_alea : np.ndarray
        AU (Aleatoric Uncertainty) 数据
    y_pred_epi : np.ndarray
        EU (Epistemic Uncertainty) 数据
    y_true : np.ndarray, optional
        真实值（用于计算误差统计）
    y_pred_mean : np.ndarray, optional
        预测均值（用于计算误差统计）
    """
    print(f"\n数据统计信息:")
    print(f"AU (y_pred_alea) - 最小值: {y_pred_alea.min():.6f}, 最大值: {y_pred_alea.max():.6f}, 平均值: {y_pred_alea.mean():.6f}")
    print(f"EU (y_pred_epi) - 最小值: {y_pred_epi.min():.6f}, 最大值: {y_pred_epi.max():.6f}, 平均值: {y_pred_epi.mean():.6f}")
    print(f"\n相关性分析:")
    correlation = np.corrcoef(y_pred_epi, y_pred_alea)[0, 1]
    print(f"AU 与 EU 的相关系数: {correlation:.4f}")
    
    # 如果提供了真实值和预测值，计算误差统计
    if y_true is not None and y_pred_mean is not None:
        absolute_error = np.abs(y_true - y_pred_mean)
        print(f"\n误差统计信息:")
        print(f"绝对误差 - 最小值: {absolute_error.min():.6f}, 最大值: {absolute_error.max():.6f}, 平均值: {absolute_error.mean():.6f}")
        error_epi_correlation = np.corrcoef(absolute_error, y_pred_epi)[0, 1]
        print(f"绝对误差 与 EU 的相关系数: {error_epi_correlation:.4f}")


def main():
    """主函数"""
    # 设置中文字体（如果需要显示中文）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 读取校准后的CSV文件（用于前3个子图）
    # calibrated_csv_path = '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_xjtu/c1_Bearing1_2_calibrated.csv'
    # # 读取RUL分布CSV文件（用于第4个子图）
    # rul_dist_csv_path = '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_xjtu/c1_Bearing1_2_labeled_fpt_scaler_fbtcn_result.csv'
    
    # 读取校准后的CSV文件（用于前3个子图）
    calibrated_csv_path = '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir/xjtu_to_femto/Bearing1_7_calibrated.csv'
    # 读取RUL分布CSV文件（用于第4个子图）
    rul_dist_csv_path = '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir/xjtu_to_femto/Bearing1_7_labeled_fpt_scaler_fbtcn_result.csv'
    
    # 读取校准后的CSV文件（用于前3个子图）
    # calibrated_csv_path = '/mnt/uq_aware_rul_prediction4bearing-main/auto_myexp_result/xjtu_to_femto/Bearing1_7_calibrated.csv'
    # 读取RUL分布CSV文件（用于第4个子图）
    # rul_dist_csv_path = '/mnt/uq_aware_rul_prediction4bearing-main/auto_myexp_result/xjtu_to_femto/Bearing1_7_labeled_fpt_scaler_fbtcn_result.csv'
    

    df = pd.read_csv(calibrated_csv_path)
    # 提取数据列
    y_true = df['y_true'].values
    y_pred_mean = df['y_pred_mean'].values
    y_pred_alea = df['y_pred_alea'].values  # AU (Aleatoric Uncertainty)
    y_pred_epi = df['y_pred_epi'].values    # EU (Epistemic Uncertainty)
    # 创建x轴（样本索引）
    sample_indices = np.arange(len(y_pred_alea))
    try:
        rul_y_true, rul_y_pred_samples, rul_y_var = load_rul_distribution_data(rul_dist_csv_path)
        n_timepoints = len(rul_y_true)
        # 选择中间时间点作为默认值
        # time_idx = n_timepoints // 2
        # time_idx = 58
        time_idx = 24


        print(f"[INFO] Loaded RUL distribution data: {n_timepoints} time points. Using time index {time_idx}.")
        
        # 提取指定时间点的数据
        true_rul = rul_y_true[time_idx]
        pred_samples = rul_y_pred_samples[time_idx, :]
        mean_rul = np.mean(pred_samples)
        variance_from_csv = rul_y_var[time_idx]
        std_rul = np.sqrt(variance_from_csv)
        
        rul_data_available = True
    except FileNotFoundError:
        print(f"[WARN] RUL distribution CSV file not found: {rul_dist_csv_path}")
        print(f"[WARN] Skipping RUL distribution plot (4th subplot).")
        rul_data_available = False
        time_idx = None
        true_rul = None
        mean_rul = None
        std_rul = None
    
    # 创建包含三个子图的图形（1x3布局），使用GridSpec控制宽度比例
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    fig = plt.figure(figsize=(36, 10))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1], wspace=0.4)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax4 = fig.add_subplot(gs[2])
    
    # 绘制第一个子图：折线图
    plot_time_series(ax1, y_pred_alea, y_pred_epi, sample_indices)
    
    # 绘制第二个子图：散点分布图
    plot_scatter_distribution(ax2, np.exp(y_pred_alea), y_pred_epi, sample_indices)
    
    # 绘制第三个子图：误差与EU的关系（暂时注释掉）
    # plot_error_vs_epi(ax3, y_true, y_pred_mean, y_pred_epi, sample_indices)
    
    # 绘制第三个子图：RUL分布
    if rul_data_available:
        plot_rul_distribution(ax4, true_rul, mean_rul, std_rul, time_idx)
    else:
        ax4.text(0.5, 0.5, 'RUL Distribution Data\nNot Available', 
                ha='center', va='center', fontsize=42, fontfamily='Times New Roman',
                transform=ax4.transAxes)
        ax4.set_title('RUL Distribution', fontsize=42, fontfamily='Times New Roman', fontweight='bold')
    
    # 调整布局，确保ax4完全可见（给图例留出足够空间）
    plt.subplots_adjust(left=0.04, right=0.99, top=0.94, bottom=0.12)
    
    # 保存图片（SVG格式）
    output_path = '/mnt/uq_aware_rul_prediction4bearing-main/alea_epi_plot_F.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    print(f"图片已保存至: {output_path}")
    
    # 显示图片
    plt.show()
    
    # 打印统计信息
    print_statistics(y_pred_alea, y_pred_epi, y_true, y_pred_mean)


if __name__ == '__main__':
    main()
