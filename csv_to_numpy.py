"""
CSV文件转NumPy数组工具

功能：读取CSV文件，转换为NumPy数组
"""

import pandas as pd
import numpy as np
import os
import sys


def csv_to_numpy(csv_path, header=True, dtype=None, use_columns=None):
    """
    读取CSV文件并转换为NumPy数组
    
    Args:
        csv_path: CSV文件路径
        header: 是否包含表头（默认True），如果True则第一行作为列名，数据从第二行开始
        dtype: 数据类型（可选），如np.float32, np.float64等
        use_columns: 要使用的列（可选），可以是列名列表或列索引列表
    
    Returns:
        array: NumPy数组
        columns: 列名列表（如果header=True）
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")
    
    try:
        # 读取CSV文件
        if header:
            df = pd.read_csv(csv_path)
            columns = df.columns.tolist()
        else:
            df = pd.read_csv(csv_path, header=None)
            columns = None
        
        # 选择特定列（如果指定）
        if use_columns is not None:
            df = df[use_columns]
        
        # 转换为NumPy数组
        if dtype is not None:
            array = df.values.astype(dtype)
        else:
            array = df.values
        
        return array, columns
        
    except Exception as e:
        raise Exception(f"读取CSV文件时发生错误: {str(e)}")


def csv_to_numpy_simple(csv_path):
    """
    简化版：读取CSV文件并返回NumPy数组（包含表头时自动跳过）
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        array: NumPy数组
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df.values


def csv_to_numpy_with_info(csv_path, header=True, dtype=None, use_columns=None):
    """
    读取CSV文件转换为NumPy数组并显示信息
    
    Args:
        csv_path: CSV文件路径
        header: 是否包含表头
        dtype: 数据类型
        use_columns: 要使用的列
    
    Returns:
        array: NumPy数组
        columns: 列名列表
    """
    print(f"读取CSV文件: {csv_path}")
    print("="*80)
    
    array, columns = csv_to_numpy(csv_path, header=header, dtype=dtype, use_columns=use_columns)
    
    print(f"数组形状: {array.shape}")
    print(f"数据类型: {array.dtype}")
    if columns:
        print(f"列数: {len(columns)}")
        print(f"列名: {columns}")
    print(f"\n数组前5行:")
    print(array[:5])
    print(f"\n数组统计信息:")
    if array.dtype in [np.float32, np.float64, np.int32, np.int64]:
        print(f"  最小值: {np.min(array)}")
        print(f"  最大值: {np.max(array)}")
        print(f"  均值: {np.mean(array)}")
        print(f"  标准差: {np.std(array)}")
    
    return array, columns


def main():
    """主函数 - 示例用法"""
    
    # 示例1: 简单读取
    print("示例1: 简单读取CSV文件")
    print("="*80)
    
    csv_path = '/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/rds/xjtu_to_xjtu/c1_Bearing1_1.csv'
    
    if os.path.exists(csv_path):
        array = csv_to_numpy_simple(csv_path)
        print(f"数组形状: {array.shape}")
        print(f"数组:\n{array}")
    else:
        print(f"文件不存在: {csv_path}")
    
    # 示例2: 带信息读取
    print("\n\n示例2: 读取CSV文件并显示详细信息")
    print("="*80)
    
    if os.path.exists(csv_path):
        array, columns = csv_to_numpy_with_info(csv_path, header=True)
        print(f"\n成功读取，数组形状: {array.shape}")
    else:
        print(f"文件不存在: {csv_path}")
    
    # 示例3: 指定数据类型
    print("\n\n示例3: 指定数据类型读取")
    print("="*80)
    
    if os.path.exists(csv_path):
        array, columns = csv_to_numpy(csv_path, dtype=np.float32)
        print(f"数组形状: {array.shape}, 数据类型: {array.dtype}")
    
    # 示例4: 只读取特定列
    print("\n\n示例4: 只读取特定列")
    print("="*80)
    
    if os.path.exists(csv_path):
        # 先读取一次获取列名
        df = pd.read_csv(csv_path)
        print(f"可用列: {df.columns.tolist()}")
        
        # 只读取前几列（例如前3列）
        if len(df.columns) >= 3:
            array, columns = csv_to_numpy(csv_path, use_columns=df.columns[:3])
            print(f"只读取前3列，数组形状: {array.shape}")


if __name__ == "__main__":
    # 如果通过命令行传入文件路径
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        print(f"从命令行参数读取CSV文件: {csv_path}")
        array, columns = csv_to_numpy_with_info(csv_path)
    else:
        # 否则运行示例
        main()
    
    # 或者直接在代码中指定文件路径
    # csv_path = 'path/to/your/file.csv'
    # array, columns = csv_to_numpy_with_info(csv_path)
    # print(f"数组形状: {array.shape}")

