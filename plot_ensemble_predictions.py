#!/usr/bin/env python3
"""
根据 MSCRGAT ensemble 预测结果 CSV 绘制：真实值、预测值、预测区间。
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 尝试使用中文字体，失败则用英文
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


def plot_ensemble_results(csv_path: str, save_path: str = None, title: str = None, figsize=(12, 5)):
    """
    绘制真实值、预测值及预测区间。
    
    Parameters
    ----------
    csv_path : str
        预测结果 CSV 路径（含 true_rul, pred_rul, y_lower, y_upper 列）
    save_path : str, optional
        图片保存路径，默认在 CSV 同目录下生成同名 .png
    title : str, optional
        图标题
    figsize : tuple
        图片尺寸
    """
    df = pd.read_csv(csv_path)
    x = np.arange(len(df))

    true_rul = df['true_rul'].values
    pred_rul = df['pred_rul'].values
    y_lower = df['y_lower'].values
    y_upper = df['y_upper'].values

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(x, y_lower, y_upper, alpha=0.25, color='steelblue', label='95% Prediction Interval')
    ax.plot(x, true_rul, 'k-', linewidth=1.2, label='True RUL')
    ax.plot(x, pred_rul, 'b-', linewidth=1.0, label='Pred RUL')

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('RUL')
    ax.set_title(title or os.path.basename(csv_path).replace('.csv', ''))
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x.min(), x.max())

    plt.tight_layout()
    if save_path is None:
        save_path = csv_path.replace('.csv', '.png')
    print(f"Saving figure to: {save_path}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='绘制 ensemble 预测区间图')
    parser.add_argument(
        'csv_path',
        nargs='?',
        # default='/mnt/uq_aware_rul_prediction4bearing-main/auto_baselines_result/bagging_ens_mscrgat_seed0/xjtu_to_femto_ensemble_bagging/Bearing1_7____rga___a_e_e_mscrgat_ensemble_predictions.csv',
        default='/mnt/uq_aware_rul_prediction4bearing-main/auto_baselines_result/bagging_ens_mscrgat_seed0/xjtu_to_xjtu_ensemble_bagging/Bearing1_3____rga___a_e_e_mscrgat_ensemble_predictions.csv',
        help='预测结果 CSV 路径'
    )
    parser.add_argument('-o', '--output', type=str, default=None, help='输出图片路径')
    parser.add_argument('-t', '--title', type=str, default=None, help='图标题')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv_path if os.path.isabs(args.csv_path) else os.path.join(base_dir, args.csv_path)
    if not os.path.isfile(csv_path):
        print(f"文件不存在: {csv_path}")
        return

    save_path = args.output
    if save_path and not os.path.isabs(save_path):
        save_path = os.path.join(base_dir, save_path)

    plot_ensemble_results(csv_path, save_path=save_path, title=args.title)


if __name__ == '__main__':
    main()
