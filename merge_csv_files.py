"""
CSV文件合并工具

功能：读取多个CSV文件，合并后输出到控制台
修改：支持多个目录，计算指标并输出到CSV（纵坐标为指标，横坐标为轴承）
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
import re
from scipy.stats import norm

# 导入指标计算函数
try:
    from 剩余寿命预测模型.metrics import (
        mae, rmse, picp, nmpiw, cwc, ece, 
        sharpness, aleatoric_uncertainty
    )
except ImportError:
    # 如果导入失败，定义基本函数
    def mae(y_true, y_pred):
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
    
    def picp(y_true, y_lower, y_upper):
        y_true = np.array(y_true)
        y_lower = np.array(y_lower)
        y_upper = np.array(y_upper)
        within_interval = np.logical_and(y_true >= y_lower, y_true <= y_upper)
        return np.mean(within_interval)
    
    def nmpiw(lower_bound, upper_bound, R):
        return np.sum(upper_bound - lower_bound) / (R * lower_bound.shape[0])
    
    def cwc(picp_val, nmpiw_val, alpha=0.05, eta=50, beta=0.95):
        penalty = np.exp(-eta * (picp_val - beta)) if picp_val < beta else 1.0
        return nmpiw_val * penalty
    
    def ece(y_true, y_pred_mean, y_pred_std, n_bins=10, alpha=0.05):
        z = norm.ppf(1 - alpha / 2)
        abs_error = np.abs(y_pred_mean - y_true)
        conf = z * y_pred_std
        bins = np.linspace(0, np.max(conf), n_bins + 1)
        ece = 0.0
        total = len(y_true)
        for i in range(n_bins):
            idx = (conf >= bins[i]) & (conf < bins[i + 1])
            if np.sum(idx) == 0:
                continue
            acc = np.mean(abs_error[idx] <= conf[idx])
            conf_avg = np.mean(conf[idx]) / (np.max(abs_error) + 1e-8)
            ece += np.abs(acc - conf_avg) * np.sum(idx) / total
        return ece
    
    def sharpness(total_uncertainty, alpha=0.05):
        return np.mean(total_uncertainty)
    
    def aleatoric_uncertainty(y_pred_std):
        return np.mean(y_pred_std)


def merge_csv_files(csv_paths, output_format='console', save_path=None, 
                   ignore_index=True, sort=False, sort_by=None):
    """
    合并多个CSV文件
    
    Args:
        csv_paths: CSV文件路径列表
        output_format: 输出格式，'console'（控制台）或 'file'（文件）
        save_path: 如果output_format='file'，指定保存路径
        ignore_index: 是否忽略原索引，重新生成连续索引
        sort: 是否排序
        sort_by: 排序依据的列名（如果sort=True）
    
    Returns:
        merged_df: 合并后的DataFrame
    """
    if not csv_paths:
        print("错误：CSV文件路径列表为空！")
        return None
    
    dataframes = []
    valid_files = []
    invalid_files = []
    
    print(f"开始处理 {len(csv_paths)} 个CSV文件...")
    print("="*80)
    
    # 读取所有CSV文件
    for i, csv_path in enumerate(csv_paths, 1):
        try:
            if not os.path.exists(csv_path):
                print(f"[{i}/{len(csv_paths)}] 文件不存在: {csv_path}")
                invalid_files.append(csv_path)
                continue
            
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            dataframes.append(df)
            valid_files.append(csv_path)
            print("df", df.shape)
            print(f"[{i}/{len(csv_paths)}] 成功读取: {csv_path} (shape: {df.shape})")
            
        except Exception as e:
            print(f"[{i}/{len(csv_paths)}] 读取失败: {csv_path}")
            print(f"  错误信息: {str(e)}")
            invalid_files.append(csv_path)
            continue
    
    print("="*80)
    
    if not dataframes:
        print("错误：没有成功读取任何CSV文件！")
        return None
    
    # 合并DataFrame
    try:
        print(f"\n合并 {len(dataframes)} 个DataFrame...")
        merged_df = pd.concat(dataframes, ignore_index=ignore_index)
        
        print(f"合并成功！合并后的shape: {merged_df.shape}")
        print(f"  行数: {merged_df.shape[0]}")
        print(f"  列数: {merged_df.shape[1]}")
        
        # 排序（如果指定）
        if sort:
            if sort_by and sort_by in merged_df.columns:
                merged_df = merged_df.sort_values(by=sort_by, ignore_index=True)
                print(f"已按列 '{sort_by}' 排序")
            elif sort_by:
                print(f"警告：排序列 '{sort_by}' 不存在，跳过排序")
        
        # 输出到控制台
        if output_format == 'console':
            print("\n" + "="*80)
            print("合并后的数据（前100行）：")
            print("="*80)
            print(merged_df.head(100).to_string())
            
            if len(merged_df) > 100:
                print(f"\n... (仅显示前100行，共{len(merged_df)}行) ...")
                print("\n" + "="*80)
                print("合并后的数据（后100行）：")
                print("="*80)
                print(merged_df.tail(100).to_string())
            
            print("\n" + "="*80)
            print("数据摘要信息：")
            print("="*80)
            print(f"总行数: {len(merged_df)}")
            print(f"总列数: {len(merged_df.columns)}")
            print(f"\n列名: {list(merged_df.columns)}")
            print(f"\n数据类型:")
            print(merged_df.dtypes)
            print(f"\n基本统计信息:")
            print(merged_df.describe())
            print(f"\n缺失值统计:")
            print(merged_df.isnull().sum())
        
        # 保存到文件（如果指定）
        if output_format == 'file' and save_path:
            merged_df.to_csv(save_path, index=False)
            print(f"\n合并结果已保存到: {save_path}")
        
        # 输出统计信息
        if invalid_files:
            print(f"\n警告：{len(invalid_files)} 个文件处理失败:")
            for f in invalid_files:
                print(f"  - {f}")
        
        return merged_df
        
    except Exception as e:
        print(f"\n错误：合并DataFrame时发生错误: {str(e)}")
        return None


def merge_csv_files_with_info(csv_paths):
    """
    合并CSV文件并显示详细信息
    
    Args:
        csv_paths: CSV文件路径列表
    
    Returns:
        merged_df: 合并后的DataFrame
    """
    print("="*80)
    print("CSV文件合并工具")
    print("="*80)
    
    # 检查文件是否存在
    print("\n检查文件...")
    for i, path in enumerate(csv_paths, 1):
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"{status} [{i}/{len(csv_paths)}] {path}")
    
    # 合并文件
    merged_df = merge_csv_files(
        csv_paths,
        output_format='console',
        ignore_index=True,
        sort=False
    )
    
    return merged_df


def get_bearing_csv_files_from_dir(mode, directory):
    """
    从目录获取所有符合Bearing模式的指标CSV文件路径
    
    匹配模式：
    - Bearing*_*.csv (如: Bearing1_1.csv, Bearing2_3.csv)
    - c*_Bearing*_*.csv (如: c1_Bearing1_1.csv, c2_Bearing2_3.csv)
    - Bearing*_*_calibrated.csv (如: Bearing1_1_calibrated.csv, Bearing2_3_calibrated.csv)
    - c*_Bearing*_*_calibrated.csv (如: c1_Bearing1_1_calibrated.csv, c2_Bearing2_3_calibrated.csv)
    
    注意：排除包含 '_result' 的文件（这些是预测结果文件，不是指标文件）
    
    Args:
        directory: 目录路径
    
    Returns:
        csv_files: CSV文件路径列表（已排序）
    """
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return []
    
    # 获取目录下所有CSV文件
    all_csv_files = glob.glob(os.path.join(directory, '*.csv'))
    
    # 过滤出符合模式的文件，并排除结果文件
    bearing_csv_files = []
    if mode == "before":
        # patterns = [
        #     r'Bearing\d+_\d+\.csv$',  # Bearing*_*.csv 模式
        #     r'c\d+_Bearing\d+_\d+\.csv$',  # c*_Bearing*_*.csv 模式
        # ]
        patterns = [
            r'Bearing\d+_\d+_ensemble_metrics\.csv$',  # Bearing*_*_ensemble_metrics.csv
            r'Bearing\d+_\d+.*ensemble_metrics\.csv$',  # Bearing1_3____rga___a_e_e_ensemble_metrics.csv 等
        ]
    elif mode == "after":
        patterns = [
            r'Bearing\d+_\d+_calibrated_metrics\.csv$',  # Bearing*_*_calibrated.csv 模式
            r'c\d+_Bearing\d+_\d+_calibrated_metrics\.csv$',  # c*_Bearing*_*_calibrated.csv 模式
        ]
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    for csv_file in all_csv_files:
        filename = os.path.basename(csv_file)
        # 排除包含 '_result' 的文件（这些是预测结果文件，不是指标文件）
        if '_result' in filename:
            continue
        # 检查是否符合任一模式
        for pattern in patterns:
            if re.match(pattern, filename):
                print(f"匹配到文件: {csv_file}")
                bearing_csv_files.append(csv_file)
                break
    
    return sorted(bearing_csv_files)


def calculate_metrics_from_csv(csv_path, alpha=0.05):
    """
    从CSV文件计算所有指标
    
    Args:
        csv_path: CSV文件路径，包含 y_true, y_pred_sample_*, y_var 列
        alpha: 显著性水平，默认0.05
    
    Returns:
        metrics_dict: 包含所有指标的字典
    """
    try:
        df = pd.read_csv(csv_path)
        
        # 提取 y_true
        if 'y_true' not in df.columns:
            raise ValueError(f"CSV文件 {csv_path} 中缺少 'y_true' 列")
        y_true = np.array(df['y_true']).reshape(-1)
        
        # 提取所有 y_pred_sample_* 列
        pred_cols = [col for col in df.columns if col.startswith('y_pred_sample_')]
        if not pred_cols:
            raise ValueError(f"CSV文件 {csv_path} 中缺少 'y_pred_sample_*' 列")
        
        # 计算预测均值（所有样本的平均）
        y_pred_samples = df[pred_cols].values  # shape: (N, num_samples)
        y_pred_mean = np.mean(y_pred_samples, axis=1)
        
        # 提取 y_var (aleatoric uncertainty)
        if 'y_var' not in df.columns:
            print(f"警告: CSV文件 {csv_path} 中缺少 'y_var' 列，使用0代替")
            y_pred_alea = np.zeros_like(y_pred_mean)
        else:
            y_pred_alea = np.array(df['y_var']).reshape(-1)
            # 确保非负
            y_pred_alea = np.abs(y_pred_alea)
            # 判断 y_var 是方差还是标准差：如果值普遍较大（>1），可能是方差，需要开方
            # 如果值较小（<1），可能是标准差，直接使用
            if np.mean(y_pred_alea) > 1.0:
                # 可能是方差，开方得到标准差
                y_pred_alea = np.sqrt(y_pred_alea)
            # 否则直接使用（已经是标准差）
        
        # 计算 epistemic uncertainty (样本间的方差)
        y_pred_epi = np.var(y_pred_samples, axis=1)
        
        # 总不确定性
        y_pred_std_total = np.sqrt(y_pred_alea**2 + y_pred_epi)
        
        # 计算预测区间
        z = norm.ppf(1 - alpha / 2)
        y_lower = y_pred_mean - z * y_pred_std_total
        y_upper = y_pred_mean + z * y_pred_std_total
        
        # 真实 RUL 范围，用于 NMPIW
        R = float(y_true.max() - y_true.min()) if y_true.size > 0 and y_true.max() != y_true.min() else 1.0
        
        # 计算所有指标
        metrics_dict = {}
        metrics_dict['MAE'] = float(mae(y_true, y_pred_mean))
        metrics_dict['RMSE'] = float(rmse(y_true, y_pred_mean))
        
        picp_val = float(picp(y_true, y_lower, y_upper))
        nmpiw_val = float(nmpiw(y_lower, y_upper, R))
        metrics_dict['PICP'] = picp_val
        metrics_dict['NMPIW'] = nmpiw_val
        metrics_dict['CWC'] = float(cwc(picp_val, nmpiw_val, alpha=alpha))
        metrics_dict['ECE'] = float(ece(y_true, y_pred_mean, y_pred_std_total, n_bins=10, alpha=alpha))
        metrics_dict['Sharpness'] = float(sharpness(y_pred_std_total, alpha=alpha))
        metrics_dict['Mean AU'] = float(aleatoric_uncertainty(y_pred_alea))
        metrics_dict['Mean EU'] = float(np.mean(y_pred_epi))
        
        return metrics_dict
        
    except Exception as e:
        print(f"计算指标时出错 {csv_path}: {str(e)}")
        return None


def read_metrics_from_csv(csv_path):
    """
    从指标CSV文件读取指标
    
    Args:
        csv_path: 指标CSV文件路径，格式为：第一行是列名，第二行是指标值
    
    Returns:
        metrics_dict: 包含所有指标的字典，如果读取失败返回None
    """
    try:
        df = pd.read_csv(csv_path)
        # 指标CSV文件只有一行数据
        if len(df) == 0:
            return None
        
        # 将第一行转换为字典
        metrics_dict = df.iloc[0].to_dict()
        return metrics_dict
        
    except Exception as e:
        print(f"读取指标文件失败 {csv_path}: {str(e)}")
        return None


def main(mode, directories, output_csv_path=None):
    """
    主函数 - 从多个目录读取指标CSV文件，按纵坐标为轴承、横坐标为指标的方式组织并输出
    
    Args:
        directories: 目录路径列表（数组）
        output_csv_path: 输出CSV文件路径，如果为None则自动生成
    """
    if isinstance(directories, str):
        directories = [directories]
    
    print("="*80)
    print("从多个目录读取指标CSV文件并汇总")
    print("="*80)
    print(f"目录数量: {len(directories)}")
    for i, d in enumerate(directories, 1):
        print(f"  [{i}] {d}")
    print()
    
    # 按目录收集轴承文件及其指标
    # 结构：{directory_path: {bearing_name: metrics_dict}}
    # directory_path 是相对路径，格式如 "D_no_ir_batch64/xjtu_to_xjtu"
    directory_bearing_metrics = {}
    all_metrics_set = set()  # 收集所有出现的指标名称
    
    # 找到所有目录的共同父目录，用于提取相对路径
    if directories:
        # 找到所有目录路径的共同前缀
        common_prefix = os.path.commonpath(directories) if len(directories) > 1 else os.path.dirname(directories[0])
    else:
        common_prefix = '.'
    
    # 遍历所有目录
    for directory in directories:
        if not os.path.exists(directory):
            print(f"警告: 目录不存在，跳过: {directory}")
            continue
        print("mode", mode)
        csv_files = get_bearing_csv_files_from_dir(mode, directory)
        
        if not csv_files:
            print(f"在目录 {directory} 中未找到符合模式的CSV文件")
            continue
        
        print(f"\n处理目录: {directory}")
        print(f"找到 {len(csv_files)} 个符合模式的CSV文件")
        
        # 提取相对路径（相对于common_prefix）
        try:
            rel_path = os.path.relpath(directory, common_prefix)
            # 如果相对路径是 '.'，则使用目录名
            if rel_path == '.':
                rel_path = os.path.basename(directory.rstrip('/'))
        except ValueError:
            # 如果无法计算相对路径，使用目录的最后两级路径
            parts = directory.rstrip('/').split(os.sep)
            if len(parts) >= 2:
                rel_path = os.path.join(parts[-2], parts[-1])
            else:
                rel_path = os.path.basename(directory.rstrip('/'))
        
        # 初始化该目录的字典
        directory_bearing_metrics[rel_path] = {}
        
        # 处理每个轴承指标文件
        for csv_file in csv_files:
            bearing_name = os.path.basename(csv_file).replace('.csv', '')
            print(f"  读取: {bearing_name}...", end=' ')
            
            metrics = read_metrics_from_csv(csv_file)
            
            if metrics is not None:
                directory_bearing_metrics[rel_path][bearing_name] = metrics
                # 收集指标名称
                all_metrics_set.update(metrics.keys())
                print("✓")
            else:
                print("✗ (读取失败)")
    
    if not directory_bearing_metrics:
        print("\n错误: 没有成功读取任何轴承的指标！")
        return None
    
    # 组织数据：纵坐标为轴承（按目录分组），横坐标为指标
    # 获取所有指标名称（按固定顺序，如果存在的话）
    preferred_order = ['MAE', 'RMSE', 'PICP', 'NMPIW', 'CWC', 'ECE', 'Sharpness', 'Mean AU', 'Mean EU']
    # 先使用preferred_order中的指标，然后添加其他指标
    all_metrics = [m for m in preferred_order if m in all_metrics_set]
    all_metrics.extend([m for m in sorted(all_metrics_set) if m not in preferred_order])
    
    # 按目录顺序构建列名列表（每个目录的轴承放在一起）
    # 列名格式：目录路径/轴承名
    column_names = []
    directory_to_rel_path = {}  # 映射：原始目录 -> 相对路径
    
    # 先建立映射关系
    for directory in directories:
        if not os.path.exists(directory):
            continue
        try:
            rel_path = os.path.relpath(directory, common_prefix)
            if rel_path == '.':
                rel_path = os.path.basename(directory.rstrip('/'))
        except ValueError:
            parts = directory.rstrip('/').split(os.sep)
            if len(parts) >= 2:
                rel_path = os.path.join(parts[-2], parts[-1])
            else:
                rel_path = os.path.basename(directory.rstrip('/'))
        directory_to_rel_path[directory] = rel_path
    
    # 按目录顺序构建行名列表（轴承名称，作为纵坐标）
    # 行名格式：目录路径/轴承名
    row_names = []
    for directory in directories:
        rel_path = directory_to_rel_path.get(directory)
        if rel_path and rel_path in directory_bearing_metrics:
            # 对该目录下的轴承名称进行排序
            bearing_names = sorted(directory_bearing_metrics[rel_path].keys())
            # 行名格式：目录路径/轴承名
            for bearing_name in bearing_names:
                row_name = f"{rel_path}/{bearing_name}"
                row_names.append(row_name)
    
    # 创建DataFrame：行为轴承（按目录分组），列为指标
    metrics_df = pd.DataFrame(index=row_names, columns=all_metrics)
    
    # 填充数据
    for rel_path, bearing_metrics in directory_bearing_metrics.items():
        for bearing_name, metrics_dict in bearing_metrics.items():
            # 构建行名：目录路径/轴承名
            row_name = f"{rel_path}/{bearing_name}"
            for metric_name in all_metrics:
                if metric_name in metrics_dict:
                    metrics_df.loc[row_name, metric_name] = metrics_dict[metric_name]
    
    # 确定输出路径
    if output_csv_path is None:
        # 使用第一个目录的父目录作为输出目录
        base_dir = os.path.dirname(directories[0]) if directories else '.'
        output_csv_path = os.path.join(base_dir, 'metrics_summary_xjtu_to_xjtu.csv')
    
    # 保存到CSV
    metrics_df.to_csv(output_csv_path, encoding='utf-8-sig')
    print(f"\n指标汇总已保存到: {output_csv_path}")
    print(f"形状: {metrics_df.shape} (行=轴承, 列=指标)")
    
    # 显示摘要
    print("\n" + "="*80)
    print("指标摘要:")
    print("="*80)
    print(metrics_df.to_string())
    return metrics_df

if __name__ == "__main__":
    # 如果通过命令行传入目录路径
    if len(sys.argv) > 1:
        directories = sys.argv[1:]
        print(f"从命令行参数读取 {len(directories)} 个目录路径")
        metrics_df = main(directories)
    else:
        # 否则运行示例：使用多个目录
        # directories = [
        #     '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_xjtu',
        #     '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_femto',
        # ]
        # metrics_df = main("before", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/metrics_summary_A_no_rds.csv")
        # metrics_df = main("after", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/metrics_summary_A_no_rds_calibrated.csv")
        
        # directories = [
        #     '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds_seed0/xjtu_to_xjtu',
        #     '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds_seed0/xjtu_to_femto',
        # ]
        # metrics_df = main("before", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds_seed0/metrics_summary_A_no_rds_seed0.csv")
        # metrics_df = main("after", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds_seed0/metrics_summary_A_no_rds_seed0_calibrated.csv")
        
        # directories = [
        #     '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/C_compact/xjtu_to_xjtu',
        #     '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/C_compact/xjtu_to_femto',
        # ]
        # metrics_df = main("before", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/C_compact/metrics_summary_C_compact.csv")
        # metrics_df = main("after", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/C_compact/metrics_summary_C_compact_calibrated.csv")
        
        # directories = [
        #     '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir/xjtu_to_xjtu',
        #     '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir/xjtu_to_femto',
        # ]
        # metrics_df = main("before", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir/metrics_summary_E_no_crps_ir.csv")
        # metrics_df = main("after", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir/metrics_summary_E_no_crps_ir_calibrated.csv")
        

        # directories = [
        #     '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir_no_rds/xjtu_to_xjtu',
        #     '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir_no_rds/xjtu_to_femto',
        # ]
        # metrics_df = main("before", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir_no_rds/metrics_summary_E_no_crps_ir_no_rds.csv")
        # metrics_df = main("after", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir_no_rds/metrics_summary_E_no_crps_ir_no_rds_calibrated.csv")

        # directories = [
        #     '/mnt/uq_aware_rul_prediction4bearing-main/auto_myexp_result/xjtu_to_xjtu',
        #     '/mnt/uq_aware_rul_prediction4bearing-main/auto_myexp_result/xjtu_to_femto',
        # ]
        # metrics_df = main("before", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_myexp_result/metrics_summary_seed42.csv")
        # metrics_df = main("after", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_myexp_result/metrics_summary_seed42_calibrated.csv")

        # directories = [
        #     '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/F_no_sa/xjtu_to_xjtu',
        #     '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/F_no_sa/xjtu_to_femto',
        # ]
        # metrics_df = main("before", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/F_no_sa/metrics_summary_F_no_sa.csv")
        # metrics_df = main("after", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/F_no_sa/metrics_summary_F_no_sa_calibrated.csv")
        
        # directories = [
        #     '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/F_no_sa_no_rds/xjtu_to_xjtu',
        #     '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/F_no_sa_no_rds/xjtu_to_femto',
        # ]
        # metrics_df = main("before", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/F_no_sa_no_rds/metrics_summary_F_no_sa_no_rds.csv")
        # metrics_df = main("after", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/F_no_sa_no_rds/metrics_summary_F_no_sa_no_rds_calibrated.csv")
        
        directories = [
            '/mnt/uq_aware_rul_prediction4bearing-main/auto_baselines_result/bagging_ens_mscrgat_seed0/xjtu_to_xjtu_ensemble_bagging',
            '/mnt/uq_aware_rul_prediction4bearing-main/auto_baselines_result/bagging_ens_mscrgat_seed0/xjtu_to_femto_ensemble_bagging',
        ]
        metrics_df = main("before", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_baselines_result/bagging_ens_mscrgat_seed0/metrics_summary_xjtu_to_xjtu_ensemble_bagging_seed0.csv")
        metrics_df = main("after", directories, "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds_sa_revise/metrics_summary_A_no_rds_sa_revise_calibrated.csv")
        