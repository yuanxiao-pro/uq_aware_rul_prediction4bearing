import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import kurtosis, entropy, skew
import nolds
from scipy.fft import fft, fftfreq
from scipy.signal import welch, find_peaks
from joblib import dump, load
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import glob
from DE import DE
try:
    import pywt
except ImportError:
    pywt = None

# 定义处理函数
def extract_features_from_file(args):
    f, feature_names = args
    print(f)
    try:
        data = pd.read_csv(f)
        h_data = data.iloc[:, -2].values
        v_data = data.iloc[:, -1].values
        signal = h_data.tolist()
        # 检查信号有效性
        if np.all(np.isnan(signal)) or np.all(np.array(signal) == signal[0]):
            print(f"Invalid signal in {f}, skipping.")
            return h_data, v_data, {name: np.nan for name in feature_names}
        kurt = kurtosis(signal)
        # ent = entropy(signal)
        # if ent < -0.000001:
        #     ent = 0
        fd = nolds.dfa(signal)
        print(fd)
        peak_factor = np.max(np.abs(signal)) / np.sqrt(np.mean(np.square(signal)))
        print(peak_factor)
        # pulse_factor = np.max(np.abs(signal)) / np.mean(np.abs(signal))
        # crest_factor = np.max(np.abs(signal)) / np.mean(np.sqrt(np.mean(np.square(signal))))
        sampling_rate = 1024
        freq, power_spectrum = welch(signal, fs=sampling_rate)
        peak_freqs, _ = find_peaks(power_spectrum, height=np.mean(power_spectrum))
        total_energy = np.sum(power_spectrum)
        peak_energy = np.sum(power_spectrum[peak_freqs])
        energy_ratio = peak_energy / total_energy
        spectral_flatness = np.exp(np.mean(np.log(power_spectrum))) / (np.mean(power_spectrum))
        mean = np.mean(signal)
        variance = np.var(signal)
        skewness = skew(signal)
        peak_vibration = np.max(np.abs(signal))
        de = DE(signal)
        # 快速傅里叶变换
        fft_result = fft(signal)
        fft_magnitude = np.abs(fft_result)[:len(signal)//2]  # 只取一半,因为是对称的
        fft_mean = np.mean(fft_magnitude)  # FFT幅值平均值
        feature_names.extend(['FFT_mean'])  # 扩展特征名列表
        feature_dict = dict(zip(feature_names, 
                [kurt, fd, peak_factor, energy_ratio, spectral_flatness, mean, variance, skewness, peak_vibration, de, fft_mean]))
        return h_data, v_data, feature_dict
    except Exception as e:
        print(f'Error processing file {f}: {e}')
        return None, None, None


# MSCRGAT 使用的 8 个非 WPD 特征（Table 4）：F1, F2, F4, F7, F9, F17, F20, F21
MSCRGAT_FEATURE_NAMES = [
    'F1_Maximum', 'F2_Minimum', 'F4_RMSE', 'F7_Kurtosis', 'F9_Impulse_Factor',
    'F17_Spectral_Kurtosis', 'F20_Dominant_Frequency', 'F21_Dominant_Amplitude'
]

# light_bar_features 对应的 13 个特征名（含 WPD F24,F25,F27,F30,F31，推测 F24–F31 为 8 个子带能量）
LIGHT_BAR_FEATURE_NAMES = [
    'F24_WPD_Subband0_Energy', 'F4_RMSE', 'F31_WPD_Subband7_Energy', 'F9_Impulse_Factor',
    'F20_Dominant_Frequency', 'F7_Kurtosis', 'F21_Dominant_Amplitude', 'F25_WPD_Subband1_Energy',
    'F2_Minimum', 'F17_Spectral_Kurtosis', 'F1_Maximum', 'F30_WPD_Subband6_Energy', 'F27_WPD_Subband3_Energy'
]


def _wp_subband_energies(signal, wavelet='db4', level=3):
    """
    3 层小波包分解，返回 8 个子带能量 [E0,...,E7]，对应 F24–F31。
    子带顺序：aaa,aad,ada,add,daa,dad,dda,ddd（由低频到高频）。
    """
    if pywt is None:
        return [np.nan] * 8
    wp = pywt.WaveletPacket(signal, wavelet=wavelet, maxlevel=level)
    nodes = [node.path for node in wp.get_level(level)]
    nodes.sort()
    energies = []
    for path in nodes:
        node = wp[path]
        coeffs = node.data
        energies.append(float(np.sum(coeffs ** 2)))
    return energies


def mscrgat_extract_features_from_file(args):
    """
    从单文件提取 light_bar_features 对应的 13 个特征（8 个非 WPD + 5 个 WPD 子带能量）。
    顺序：F24, F4, F31, F9, F20, F7, F21, F25, F2, F17, F1, F30, F27。
    接口与 extract_features_from_file 一致：(f, feature_names) -> (h_data, v_data, feature_dict)。
    """
    f, feature_names = args
    try:
        data = pd.read_csv(f)
        h_data = data.iloc[:, -2].values
        v_data = data.iloc[:, -1].values
        signal = np.asarray(h_data, dtype=np.float64)
        # 检查信号有效性
        if np.all(np.isnan(signal)) or (len(signal) > 0 and np.all(signal == signal[0])):
            return h_data, v_data, {name: np.nan for name in feature_names}
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------- 非 WPD ----------
        # F1: Maximum
        f1_maximum = np.max(signal)
        # F2: Minimum
        f2_minimum = np.min(signal)
        # F4: RMSE = sqrt(mean(x^2))
        f4_rmse = np.sqrt(np.mean(signal ** 2))
        # F7: Kurtosis
        f7_kurtosis = kurtosis(signal)
        # F9: Impulse Factor = max(|x|) / mean(|x|)
        abs_signal = np.abs(signal)
        mav = np.mean(abs_signal)
        f9_impulse_factor = np.max(abs_signal) / (mav + 1e-12)

        sampling_rate = 1024
        freq, power_spectrum = welch(signal, fs=sampling_rate)
        # F17: Spectral Kurtosis
        f17_spectral_kurtosis = kurtosis(power_spectrum)
        idx_dom = np.argmax(power_spectrum)
        # F20: Dominant Frequency
        f20_dominant_frequency = freq[idx_dom]
        # F21: Dominant Amplitude
        f21_dominant_amplitude = power_spectrum[idx_dom]

        # ---------- WPD F24–F31：8 个子带能量（推测） ----------
        wp_energies = _wp_subband_energies(signal)
        f24 = wp_energies[0]
        f25 = wp_energies[1]
        f27 = wp_energies[3]
        f30 = wp_energies[6]
        f31 = wp_energies[7]

        # 按 light_bar_features 顺序：F24, F4, F31, F9, F20, F7, F21, F25, F2, F17, F1, F30, F27
        values = [
            f24, f4_rmse, f31, f9_impulse_factor, f20_dominant_frequency, f7_kurtosis, f21_dominant_amplitude,
            f25, f2_minimum, f17_spectral_kurtosis, f1_maximum, f30, f27
        ]
        feature_dict = dict(zip(feature_names, values))
        return h_data, v_data, feature_dict
    except Exception as e:
        print(f'Error in mscrgat_extract_features_from_file {f}: {e}')
        return None, None, None


def get_a_bearings_data(folder):
    feature_names = [
        'Kurtosis', 'Fractal Dimension', 'Peak factor',
        'Energy ratio', 'Spectral flatness', 'Mean', 'Variance', 'Skewness', 'Peak vibration', 'DE', 'FFT_mean'
    ]
    csv_file_names = [f for f in os.listdir(folder) if f.endswith('.csv')]
    def extract_file_number(filename):
        if filename.startswith('acc_') and filename.endswith('.csv'):
            return int(filename[4:-4].lstrip('0') or '0')
        elif filename.endswith('.csv'):
            return int(filename[:-4])
        else:
            return float('inf')
    sorted_file_names = sorted(csv_file_names, key=extract_file_number)
    files = [os.path.join(folder, f) for f in sorted_file_names]
    h, v, features = [], [], []
    args_list = [(f, feature_names) for f in files]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(extract_features_from_file, args_list))
    for result in results:
        if result[0] is None:
            continue  # 跳过出错的文件
        h_data, v_data, feature_dict = result
        h.append(h_data)
        v.append(v_data)
        features.append(feature_dict)
    H = np.concatenate(h)
    V = np.concatenate(v)
    print(H.shape, V.shape)
    features_df = pd.DataFrame(features)
    return np.stack([H, V], axis=-1), features_df, feature_names

# 优化：批量处理所有轴承子目录
def process_all_bearings(root_dir='./XJTU-SY_Bearing_Datasets'):
    # 判断保存目录
    root_dir_lower = root_dir.lower()
    if 'xjtu' in root_dir_lower:
        save_dir = './datasetresult/xjtu'
    elif 'ottawa' in root_dir_lower:
        save_dir = './datasetresult/Ottawa'
    elif 'femto' in root_dir_lower:
        save_dir = './datasetresult/femto'
    else:
        save_dir = './datasetresult/other'
    os.makedirs(save_dir, exist_ok=True)

    bearing_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for subdir in dirnames:
            if subdir.startswith('Bearing'):
                bearing_dirs.append(os.path.join(dirpath, subdir))
    for bearing_path in tqdm(bearing_dirs, desc='Processing Bearings'):
        print(f'Processing {bearing_path}...')
        all_data, features_df, feature_names = get_a_bearings_data(bearing_path)
        out_prefix = os.path.relpath(bearing_path, root_dir).replace(os.sep, '_')
        dump(features_df, os.path.join(save_dir, f'{out_prefix}_features_df'))
        dump(all_data, os.path.join(save_dir, f'{out_prefix}_all_data'))
        features_df.to_csv(os.path.join(save_dir, f'{out_prefix}_features_df.csv'), index=False)
        print(f'{bearing_path} done!')

# ---------- MSCRGAT 特征流程（用 mscrgat_extract_features_from_file，结果存到独立目录，不覆盖原有数据） ----------
def get_a_bearings_data_mscrgat(folder):
    """与 get_a_bearings_data 结构相同，但使用 mscrgat_extract_features_from_file 提取 13 个 light_bar 特征。"""
    feature_names = list(LIGHT_BAR_FEATURE_NAMES)
    csv_file_names = [f for f in os.listdir(folder) if f.endswith('.csv')]

    def extract_file_number(filename):
        if filename.startswith('acc_') and filename.endswith('.csv'):
            return int(filename[4:-4].lstrip('0') or '0')
        elif filename.endswith('.csv'):
            return int(filename[:-4])
        else:
            return float('inf')

    sorted_file_names = sorted(csv_file_names, key=extract_file_number)
    files = [os.path.join(folder, f) for f in sorted_file_names]
    h, v, features = [], [], []
    args_list = [(f, feature_names) for f in files]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(mscrgat_extract_features_from_file, args_list))
    for result in results:
        if result[0] is None:
            continue
        h_data, v_data, feature_dict = result
        h.append(h_data)
        v.append(v_data)
        features.append(feature_dict)
    H = np.concatenate(h)
    V = np.concatenate(v)
    features_df = pd.DataFrame(features)
    return np.stack([H, V], axis=-1), features_df, feature_names


def process_all_bearings_mscrgat(root_dir='./XJTU-SY_Bearing_Datasets'):
    """
    使用 mscrgat_extract_features_from_file 提取 13 个特征，保存到独立子目录（*_mscrgat），
    不覆盖 process_all_bearings 生成的数据。
    """
    root_dir_lower = root_dir.lower()
    if 'xjtu' in root_dir_lower:
        save_dir = './datasetresult/xjtu_mscrgat'
    elif 'ottawa' in root_dir_lower:
        save_dir = './datasetresult/ottawa_mscrgat'
    elif 'femto' in root_dir_lower:
        save_dir = './datasetresult/femto_mscrgat'
    else:
        save_dir = './datasetresult/other_mscrgat'
    os.makedirs(save_dir, exist_ok=True)

    bearing_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for subdir in dirnames:
            if subdir.startswith('Bearing'):
                bearing_dirs.append(os.path.join(dirpath, subdir))
    for bearing_path in tqdm(bearing_dirs, desc='MSCRGAT Bearings'):
        print(f'MSCRGAT Processing {bearing_path}...')
        all_data, features_df, feature_names = get_a_bearings_data_mscrgat(bearing_path)
        out_prefix = os.path.relpath(bearing_path, root_dir).replace(os.sep, '_')
        dump(features_df, os.path.join(save_dir, f'{out_prefix}_mscrgat_features_df'))
        dump(all_data, os.path.join(save_dir, f'{out_prefix}_mscrgat_all_data'))
        features_df.to_csv(os.path.join(save_dir, f'{out_prefix}_mscrgat_features_df.csv'), index=False)
        print(f'{bearing_path} MSCRGAT done!')

def label_all_bearings(dataset='xjtu'):
    features_df_files = []
    if dataset == 'xjtu' or dataset == 'xjtu_mscrgat':
        # 批量处理datasetresult/xjtu中的全部c*_Bearing*_*_features_df文件
        features_df_files = glob.glob(f'datasetresult/{dataset}/Bearing*_*_features_df')
    elif dataset == 'femto' or dataset == 'femto_mscrgat':
        # 批量处理datasetresult/femto中的全部c*_Bearing*_*_features_df文件
        features_df_files = glob.glob(f'datasetresult/{dataset}/*_Bearing*_*_features_df')
    elif dataset == 'ottawa':
        # 批量处理datasetresult/ottawa中的全部c*_Bearing*_*_features_df文件
        features_df_files = glob.glob(f'datasetresult/{dataset}/c*_Bearing*_*_features_df')
    elif dataset == 'other':
        # 批量处理datasetresult/other中的全部c*_Bearing*_*_features_df文件
        features_df_files = glob.glob(f'datasetresult/{dataset}/c*_Bearing*_*_features_df')
    else:
        raise ValueError(f'Invalid dataset: {dataset}')
        # return

    for features_df_file in features_df_files:
        features_df = load(features_df_file)
        total_rul = features_df.shape[0]
        # 生成RUL标签
        data_rul = [(total_rul - i) / total_rul for i in range(1, total_rul + 1)]
        features_df['rul'] = data_rul
        # 生成输出csv文件名
        out_csv = features_df_file.replace('_features_df', '_labeled.csv')
        features_df.to_csv(out_csv, index=False)
        print(f"{features_df_file} -> {out_csv} 完成, shape: {features_df.shape}")

def label_bearings_after_fpt(dataset='xjtu'):
    """
    只截取FPT后的数据，FPT对应的数据标签为1，最后一个数据的标签为0，线性下降
    输出csv文件，文件名要带fpt标记作为区分
    
    Args:
        dataset: 数据集名称，'xjtu', 'femto', 'ottawa', 'other'
    """
    # FPT字典定义
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
    
    # 根据数据集选择对应的FPT字典
    if dataset == 'xjtu' or dataset == 'xjtu_mscrgat':
        fpt_dict = FPT_dict_xj
        features_df_files = glob.glob(f'datasetresult/{dataset}/Bearing*_*_features_df')
    elif dataset == 'femto' or dataset == 'femto_mscrgat':
        fpt_dict = FPT_dict_femto
        features_df_files = glob.glob(f'datasetresult/{dataset}/*_Bearing*_*_features_df')
    elif dataset == 'ottawa':
        fpt_dict = {}  # ottawa数据集没有FPT字典，可以后续添加
        features_df_files = glob.glob(f'datasetresult/{dataset}/c*_Bearing*_*_features_df')
    elif dataset == 'other':
        fpt_dict = {}  # other数据集没有FPT字典，可以后续添加
        features_df_files = glob.glob(f'datasetresult/{dataset}/c*_Bearing*_*_features_df')
    else:
        raise ValueError(f'Invalid dataset: {dataset}')
        # return
    
    import re
    
    for features_df_file in features_df_files:
        # 从文件名中提取轴承名称
        # 例如: c1_Bearing1_1_features_df -> Bearing1_1
        filename = os.path.basename(features_df_file)
        # 匹配 Bearing数字_数字 的模式
        match = re.search(r'(Bearing\d+_\d+)', filename)
        if not match:
            print(f"无法从文件名 {filename} 中提取轴承名称，跳过")
            continue
        
        bearing_name = match.group(1)
        
        # 检查该轴承是否在FPT字典中
        if bearing_name not in fpt_dict:
            print(f"轴承 {bearing_name} 不在FPT字典中，跳过")
            continue
        
        fpt = fpt_dict[bearing_name]
        
        # 如果FPT为0，跳过（没有FPT点）
        # if fpt == 0:
        #     print(f"轴承 {bearing_name} 的FPT为0，跳过")
        #     continue
        
        # 加载数据
        features_df = load(features_df_file)
        total_rows = features_df.shape[0]
        
        # 检查FPT是否超出数据范围
        if fpt >= total_rows:
            print(f"轴承 {bearing_name} 的FPT ({fpt}) 超出数据范围 ({total_rows})，跳过")
            continue
        
        # 截取FPT之后的数据（包括FPT点）
        features_df_fpt = features_df.iloc[fpt:].copy()
        total_rul = features_df_fpt.shape[0]
        
        # 按照label_all_bearings的标注方法生成RUL标签
        # label_all_bearings使用: [(total_rul - i) / total_rul for i in range(1, total_rul + 1)]
        # 这里调整为: [(total_rul - i) / (total_rul - 1) for i in range(1, total_rul + 1)]
        # 确保FPT点（第一个数据）标签为1，最后一个数据标签为0，线性下降
        # 当i=1时: (total_rul - 1) / (total_rul - 1) = 1
        # 当i=total_rul时: (total_rul - total_rul) / (total_rul - 1) = 0
        if total_rul > 1:
            data_rul = [(total_rul - i) / (total_rul - 1) for i in range(1, total_rul + 1)]
        else:
            # 如果只有一个数据点，标签为1
            data_rul = [1.0]
        
        features_df_fpt['rul'] = data_rul
        
        # 生成输出csv文件名，添加fpt标记
        # 例如: c1_Bearing1_1_features_df -> c1_Bearing1_1_labeled_fpt.csv
        # 处理文件名：移除可能的.csv扩展名，替换_features_df为_labeled_fpt，然后添加.csv
        base_name = features_df_file
        if base_name.endswith('.csv'):
            base_name = base_name[:-4]  # 移除.csv扩展名
        # 替换_features_df为_labeled_fpt
        out_csv = base_name.replace('_features_df', '_labeled_fpt') + '.csv'
        
        features_df_fpt.to_csv(out_csv, index=False)
        print(f"{features_df_file} -> {out_csv} 完成, FPT={fpt}, 原始shape={features_df.shape}, FPT后shape={features_df_fpt.shape}")

if __name__ == '__main__':
    # 处理XJTU-SY_Bearing_Datasets数据集（原有流程，结果在 datasetresult/xjtu 等）
    # process_all_bearings('./datasetresult/femto_origin/')
    # label_bearings_after_fpt('femto')
    process_all_bearings('./datasetresult/xjtu_origin/')
    label_bearings_after_fpt('xjtu')

    # MSCRGAT 特征流程：用 mscrgat_extract_features_from_file 提取 13 个 light_bar 特征，结果存到 datasetresult/xjtu_mscrgat，不覆盖上面数据
    # process_all_bearings_mscrgat('./datasetresult/femto_origin/')
    # label_bearings_after_fpt('femto_mscrgat')

