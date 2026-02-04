import os
import glob
import pandas as pd
import numpy as np
import torch
from joblib import dump
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from joblib import load

FPT_dict_xj = {'Bearing1_1': 76, 'Bearing1_2': 44, 'Bearing1_3': 60, 'Bearing1_4': 0, 'Bearing1_5': 39,
            'Bearing2_1': 455, 'Bearing2_2': 48, 'Bearing2_3': 327, 'Bearing2_4': 32, 'Bearing2_5': 141,
            'Bearing3_1': 2344, 'Bearing3_2': 0, 'Bearing3_3': 340, 'Bearing3_4': 1418, 'Bearing3_5': 9}
FPT_dict_femto = {'Bearing1_1': 1314, 'Bearing1_2': 826,
                'Bearing2_1': 836, 'Bearing2_2': 790,
                'Bearing3_1': 490, 'Bearing3_2': 1446,
                'Bearing1_3': 1726, 'Bearing1_4': 1082,
                'Bearing1_5': 2412, 'Bearing1_6': 1631,
                'Bearing1_7': 2210, 'Bearing2_3': 779,
                'Bearing2_4': 373, 'Bearing2_5': 406,
                'Bearing2_6': 442, 'Bearing2_7': 162,
                'Bearing3_3': 322}

# 1. 定义make_data_labels和data_window_maker
def make_data_labels(x_data, y_label):
    '''
        返回 x_data: 数据集     torch.tensor
            y_label: 对应标签值  torch.tensor
    '''
    x_data = torch.tensor(x_data).float()
    y_label = torch.tensor(y_label).float()
    return x_data, y_label

def data_window_maker(x_var, ylable_data, window_size):
    '''
        参数:
        x_var      : 输入 变量数据
        ylable_data: 对应y数据
        window_size: 滑动窗口大小

        返回:
        data_x: 特征数据
        data_y: 标签数据
    '''
    data_x = []
    data_y = []
    data_len = x_var.shape[0]
    for i in range(data_len - window_size):
        data_x.append(x_var[i:i+window_size, :])
        data_y.append(ylable_data[i+window_size])
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_x, data_y = make_data_labels(data_x, data_y)
    return data_x, data_y

# 2. 处理所有labeled.csv
def process_all_labeled_csv(input_dir='datasetresult/xjtu', output_dir='datasetresult/xjtu_made', window_size=7):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_dir, '*_labeled_fpt.csv'))
    for csv_file in csv_files:
        print(f'Processing {csv_file}...')
        df = pd.read_csv(csv_file)
        # 只选数值型特征（去除非数值列）
        feature_cols = [col for col in df.columns if col != 'rul']
        X = df[feature_cols].values
        y = df[['rul']].values
        # 归一化
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        y_norm = scaler.fit_transform(y)
        # 保存归一化模型
        base = os.path.splitext(os.path.basename(csv_file))[0]
        dump(scaler, os.path.join(output_dir, f'{base}_scaler'))
        # 滑动窗口处理
        data_x, data_y = data_window_maker(X_norm, y_norm, window_size)
        print(data_x.shape, data_y.shape)
        # 保存数据
        dump(data_x, os.path.join(output_dir, f'{base}_data'))
        dump(data_y, os.path.join(output_dir, f'{base}_label'))
        print(f'Saved: {base}_data, {base}_label, {base}_scaler')

def load_and_test_data(input_dir='datasetresult/xjtu_made', window_size=7):
    """
    加载指定的数据集，绘制其rul图像
    """

    data_files = glob.glob(os.path.join(input_dir, '*_data'))
    label_files = glob.glob(os.path.join(input_dir, '*_label'))

    if not data_files or not label_files:
        print("No data or label files found in the specified directory.")
        return

    for data_file, label_file in zip(sorted(data_files), sorted(label_files)):
        base_name = os.path.basename(data_file).replace('_data', '')
        print(f"Loading {base_name}...")

        data_x = load(data_file)
        data_y = load(label_file)
        print(data_x.shape, data_y.shape)
        # Flatten if needed
        if data_y.ndim > 1 and data_y.shape[1] == 1:
            rul = data_y.flatten()
        else:
            rul = data_y

        plt.figure(figsize=(10, 4))
        plt.plot(rul, label='RUL')
        plt.title(f'RUL Curve for {base_name}')
        plt.xlabel('Sample Index')
        plt.ylabel('RUL (normalized)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{base_name}.png')
        plt.close()

def visualize_all_features(input_dir='datasetresult/xjtu_made', window_size=1):
    """
    可视化每个数据集的全部13个特征（每个特征随样本索引的变化曲线）
    """
    import matplotlib.pyplot as plt
    import os
    import glob

    data_files = glob.glob(os.path.join(input_dir, '*_data'))

    if not data_files:
        print("No data files found in the specified directory.")
        return

    for data_file in sorted(data_files):
        base_name = os.path.basename(data_file).replace('_data', '')
        print(f"Visualizing features for {base_name}...")

        data_x = load(data_file)  # shape: (num_samples, window_size, num_features)
        # 取每个样本窗口的最后一个时刻的特征（假设预测目标与窗口最后时刻对齐）
        if data_x.ndim == 3:
            features = data_x[:, -1, :]  # shape: (num_samples, num_features)
        elif data_x.ndim == 2:
            features = data_x  # shape: (num_samples, num_features)
        else:
            print(f"Unexpected data shape: {data_x.shape}")
            continue

        num_features = features.shape[1]
        print(num_features)
        plt.figure(figsize=(18, 12))
        for i in range(num_features):
            plt.subplot(4, 4, i+1)
            plt.plot(features[:, i])
            plt.title(f'Feature {i+1}')
            plt.xlabel('Sample Index')
            plt.ylabel(f'Feature {i+1}')
            plt.tight_layout()
        plt.suptitle(f'All Features for {base_name}', fontsize=16, y=1.02)
        plt.subplots_adjust(top=0.92, hspace=0.5)
        plt.savefig(f'{base_name}_all_features.png')
        plt.close()

if __name__ == '__main__':
    process_all_labeled_csv(input_dir='datasetresult/femto_mscrgat', output_dir='datasetresult/femto_made_mscrgat', window_size=1) 
    # load_and_test_data()
    # visualize_all_features(window_size=1)

    # process_all_labeled_csv(input_dir='datasetresult/femto', output_dir='datasetresult/femto_made', window_size=1) 
    # visualize_all_features(window_size=1)


