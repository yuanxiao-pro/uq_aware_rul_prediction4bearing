#!/usr/bin/env python3
"""
计算CSV文件中的PICP、NMPIW、CWC指标，并支持绘制预测区间/预测曲线/真实值。
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加metrics模块路径
sys.path.append('/mnt/uq_aware_rul_prediction4bearing-main/剩余寿命预测模型')
from metrics import picp, nmpiw, cwc, mae, rmse

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def calculate_metrics_from_csv(csv_path, alpha=0.05):
    """
    从CSV文件计算PICP、NMPIW、CWC指标
    
    Parameters:
    -----------
    csv_path : str
        CSV文件路径
    alpha : float
        显著性水平，默认0.05（对应95%置信区间）
    
    Returns:
    --------
    dict : 包含PICP、NMPIW、CWC的字典
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 提取数据
    y_true = df['y_true'].values
    y_pred_mean = df['y_pred_mean'].values
    y_lower = df['y_lower_calibrated'].values
    y_upper = df['y_upper_calibrated'].values
    
    # 计算R（真实值的范围），用于NMPIW归一化
    R = float(y_true.max() - y_true.min()) if y_true.size > 0 and y_true.max() != y_true.min() else 1.0
    
    # 计算指标
    mae_val = mae(y_true, y_pred_mean)
    rmse_val = rmse(y_true, y_pred_mean)
    picp_val = picp(y_true, y_lower, y_upper)
    nmpiw_val = nmpiw(y_lower, y_upper, R)
    cwc_val = cwc(picp_val, nmpiw_val, alpha=alpha)
    
    return {
        'MAE': mae_val,
        'RMSE': rmse_val,
        'PICP': picp_val,
        'NMPIW': nmpiw_val,
        'CWC': cwc_val,
        'R': R,
        'alpha': alpha,
        'target_coverage': 1.0 - alpha
    }


def _predictions_path_from_metrics_path(metrics_csv_path):
    """从 metrics CSV 路径得到同名的 predictions CSV 路径。"""
    if not metrics_csv_path.endswith('_metrics.csv'):
        return metrics_csv_path
    return metrics_csv_path.replace('_metrics.csv', '_predictions.csv')


def plot_predictions_from_csv(csv_path, save_path=None, figsize=(10, 6)):
    """
    从 predictions CSV 读取数据，绘制预测区间、预测曲线、真实值。
    若传入的是 _metrics.csv 路径，会自动改为同名的 _predictions.csv 读取。

    Parameters
    ----------
    csv_path : str
        CSV 路径（可为 _metrics.csv 或 _predictions.csv）
    save_path : str, optional
        图片保存路径，默认在 CSV 同目录下生成同名 .png
    figsize : tuple
        图像大小
    """
    if not _HAS_MATPLOTLIB:
        print("未安装 matplotlib，跳过画图。")
        return

    pred_path = _predictions_path_from_metrics_path(csv_path)
    if not os.path.exists(pred_path):
        print(f"画图失败：未找到预测数据文件 {pred_path}")
        return

    df = pd.read_csv(pred_path)

    # 兼容两种列名：ensemble 输出为 true_rul, pred_rul, y_lower, y_upper；校准脚本为 y_true, y_pred_mean, y_lower_calibrated, y_upper_calibrated
    if 'true_rul' in df.columns and 'pred_rul' in df.columns:
        y_true = df['true_rul'].values
        y_pred = df['pred_rul'].values
        y_lower = df['y_lower'].values
        y_upper = df['y_upper'].values
    elif 'y_true' in df.columns and 'y_pred_mean' in df.columns:
        y_true = df['y_true'].values
        y_pred = df['y_pred_mean'].values
        y_lower = df['y_lower_calibrated'].values if 'y_lower_calibrated' in df.columns else df['y_lower'].values
        y_upper = df['y_upper_calibrated'].values if 'y_upper_calibrated' in df.columns else df['y_upper'].values
    else:
        print("画图失败：CSV 中未找到 true_rul/pred_rul 或 y_true/y_pred_mean 等列。")
        return

    x = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(x, y_lower, y_upper, alpha=0.3, color='steelblue', label='Prediction Interval (PI)')
    ax.plot(x, y_pred, 'b-', linewidth=1.5, label='Predicted RUL')
    ax.plot(x, y_true, 'k-', linewidth=1.2, label='True RUL')

    ax.set_xlabel('Time step / Sample index', fontsize=11)
    ax.set_ylabel('RUL', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    if save_path is None:
        save_path = pred_path.replace('.csv', '.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"预测区间图已保存: {save_path}")


def main():
    # 默认使用 Bearing1_3 ensemble 结果：metrics 文件仅含汇总指标，画图会从同目录 _predictions.csv 读入
    csv_path = '/mnt/uq_aware_rul_prediction4bearing-main/auto_baselines_result/bagging_ens_mscrgat_seed0/xjtu_to_xjtu_ensemble_bagging/Bearing1_3____rga___a_e_e_mscrgat_ensemble_metrics.csv'
    
    if not os.path.exists(csv_path):
        print(f"错误：文件不存在: {csv_path}")
        return

    # 判断是否为“仅指标”的 CSV（单行 MAE/RMSE/PICP 等），这类文件没有逐点数据，只画图
    df_head = pd.read_csv(csv_path, nrows=1)
    has_per_point = 'y_true' in df_head.columns or 'true_rul' in df_head.columns

    if has_per_point:
        print(f"正在计算指标: {csv_path}")
        print("=" * 60)
        metrics = calculate_metrics_from_csv(csv_path, alpha=0.05)
        print(f"\n指标计算结果 (置信水平: {metrics['target_coverage']*100:.1f}%):")
        print("-" * 60)
        print(f"MAE   (Mean Absolute Error): {metrics['MAE']:.6f}")
        print(f"      (越小越好)")
        print(f"\nRMSE  (Root Mean Squared Error): {metrics['RMSE']:.6f}")
        print(f"      (越小越好)")
        print(f"\nPICP  (Prediction Interval Coverage Probability): {metrics['PICP']:.6f}")
        print(f"      目标覆盖率: {metrics['target_coverage']:.4f}")
        print(f"      覆盖率误差: {abs(metrics['PICP'] - metrics['target_coverage']):.6f}")
        print(f"\nNMPIW (Normalized Mean Prediction Interval Width): {metrics['NMPIW']:.6f}")
        print(f"      真实值范围 R: {metrics['R']:.6f}")
        print(f"\nCWC   (Coverage Width-based Criterion): {metrics['CWC']:.6f}")
        print(f"      (越小越好)")
        print("=" * 60)
        output_path = csv_path.replace('.csv', '_metrics.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"指标计算结果 (置信水平: {metrics['target_coverage']*100:.1f}%)\n")
            f.write("=" * 60 + "\n")
            f.write(f"MAE   (Mean Absolute Error): {metrics['MAE']:.6f}\n")
            f.write(f"      (越小越好)\n")
            f.write(f"\nRMSE  (Root Mean Squared Error): {metrics['RMSE']:.6f}\n")
            f.write(f"      (越小越好)\n")
            f.write(f"\nPICP  (Prediction Interval Coverage Probability): {metrics['PICP']:.6f}\n")
            f.write(f"      目标覆盖率: {metrics['target_coverage']:.4f}\n")
            f.write(f"      覆盖率误差: {abs(metrics['PICP'] - metrics['target_coverage']):.6f}\n")
            f.write(f"\nNMPIW (Normalized Mean Prediction Interval Width): {metrics['NMPIW']:.6f}\n")
            f.write(f"      真实值范围 R: {metrics['R']:.6f}\n")
            f.write(f"\nCWC   (Coverage Width-based Criterion): {metrics['CWC']:.6f}\n")
            f.write(f"      (越小越好)\n")
            f.write("=" * 60 + "\n")
        print(f"\n结果已保存到: {output_path}")
    else:
        print(f"当前 CSV 为汇总指标文件，仅进行画图（从同目录 _predictions.csv 读取逐点数据）。")

    # 绘制预测区间、预测曲线、真实值（若为 _metrics.csv 则自动使用同名的 _predictions.csv）
    plot_predictions_from_csv(csv_path)


if __name__ == "__main__":
    main()
