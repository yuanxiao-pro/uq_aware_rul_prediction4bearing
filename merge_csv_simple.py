"""
简洁版CSV文件合并工具

输入：CSV文件路径数组
输出：合并后的数据（控制台）
"""

import pandas as pd
import os


def merge_csv_to_console(csv_paths):
    """
    合并多个CSV文件并输出到控制台
    
    Args:
        csv_paths: CSV文件路径列表
    
    Returns:
        merged_df: 合并后的DataFrame（如果没有错误）
    """
    if not csv_paths:
        print("错误：CSV文件路径列表为空！")
        return None
    
    dataframes = []
    
    # 读取所有CSV文件
    print(f"正在读取 {len(csv_paths)} 个CSV文件...")
    for i, csv_path in enumerate(csv_paths, 1):
        try:
            if not os.path.exists(csv_path):
                print(f"警告：文件不存在，跳过: {csv_path}")
                continue
            
            df = pd.read_csv(csv_path)
            dataframes.append(df)
            print(f"[{i}/{len(csv_paths)}] ✓ {csv_path} ({df.shape[0]}行 x {df.shape[1]}列)")
            
        except Exception as e:
            print(f"[{i}/{len(csv_paths)}] ✗ 读取失败: {csv_path}")
            print(f"   错误: {str(e)}")
            continue
    
    if not dataframes:
        print("错误：没有成功读取任何CSV文件！")
        return None
    
    # 合并DataFrame
    try:
        merged_df = pd.concat(dataframes, ignore_index=True)
        print(f"\n合并成功！")
        print(f"合并后数据形状: {merged_df.shape[0]}行 x {merged_df.shape[1]}列\n")
        
        # 输出到控制台
        print("="*80)
        print("合并后的数据：")
        print("="*80)
        print(merged_df.to_string())
        
        print("\n" + "="*80)
        print("数据摘要：")
        print("="*80)
        print(f"行数: {len(merged_df)}, 列数: {len(merged_df.columns)}")
        print(f"列名: {list(merged_df.columns)}")
        print(f"\n缺失值:")
        print(merged_df.isnull().sum())
        
        return merged_df
        
    except Exception as e:
        print(f"错误：合并失败 - {str(e)}")
        return None


# 使用示例
if __name__ == "__main__":
    # 示例：指定要合并的CSV文件路径
    csv_files = [
        # 'datasetresult/xjtu/c1_Bearing1_1_features_df.csv',
        # 'datasetresult/xjtu/c1_Bearing1_2_features_df.csv',
        # 'datasetresult/xjtu/c1_Bearing1_3_features_df.csv',
    ]
    
    # 调用函数合并并输出
    merged_df = merge_csv_to_console(csv_files)
    
    # 如果需要保存到文件，可以使用：
    # if merged_df is not None:
    #     merged_df.to_csv('merged_output.csv', index=False)
    #     print("\n已保存到 merged_output.csv")

