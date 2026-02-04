import os
import json
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
import torch.utils.data as Data
from joblib import load, dump

from metrics import (
    mae,
    rmse,
    picp,
    nmpiw,
    ece,
    aleatoric_uncertainty,
    epistemic_uncertainty,
    sharpness,
    cwc,
)

def run_test_and_save(
    model: torch.nn.Module,
    test_loader: Data.DataLoader,
    forward_pass: int,
    test_bearings: str,
    results_dir: str,
    scaler_dir: str,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    封装 notebook 中的测试逻辑（Cell 9 + 10 的前半部分），并返回:
    - target: 反归一化后的真实 RUL，一维数组
    - prediction: 反归一化后的预测均值，一维数组
    - origin_prediction: 反归一化后的多次采样结果，形状 (forward_pass, N)
    - var_list: 预测 aleatoric log-variance（经 softplus 后 sigma），一维数组

    同时会按照 notebook 风格保存部分结果到 results_dir。

    参数说明：
    - model: 已加载好权重并与 notebook 一致的 BayesianTCN 模型
    - test_loader: notebook 中构造好的测试集 DataLoader
    - forward_pass: 每个样本的前向采样次数（对应 notebook 中 forward_pass）
    - test_bearings: 当前测试所用的轴承列表（对应 notebook 中 TEST_xj）
    - device/results_dir/scaler_dir: 与 notebook 中的设备与路径保持兼容
    """
    # 设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model.load_state_dict(torch.load(model_path))

    model = model.to(device)
    model.eval()

    # 4. 测试循环（对应 Cell 9）
    target: List[float] = []
    prediction: List[float] = []
    origin_prediction_batches: List[np.ndarray] = []
    var_list: List[float] = []

    with torch.no_grad():
        for data, label in test_loader:
            origin_label = label.tolist()
            target += origin_label

            data = data.to(device)
            label = label.to(device)

            mu_list = []
            sigma_list = []
            for _ in range(forward_pass):
                mu, sigma, _ = model(data)
                mu_list.append(mu.cpu().numpy())
                sigma_list.append(sigma.cpu().numpy())

            mu_samples = np.stack(mu_list, axis=0)  # (N, batch, 1)
            sigma_samples = np.stack(sigma_list, axis=0)  # (N, batch, 1)

            mu_mean = np.mean(mu_samples, axis=0)  # (batch, 1)
            sigma_mean = np.mean(sigma_samples, axis=0)  # (batch, 1)

            prediction += mu_mean.squeeze(-1).tolist()
            var_list += sigma_mean.squeeze(-1).tolist()

            mu_samples_squeezed = mu_samples.squeeze(-1)  # (N, batch)
            origin_prediction_batches.append(mu_samples_squeezed)

    if len(origin_prediction_batches) > 0:
        origin_prediction = np.concatenate(origin_prediction_batches, axis=1)  # (N, total_samples)
    else:
        origin_prediction = np.array([])

    target_arr = np.array(target).reshape(-1)
    prediction_arr = np.array(prediction).reshape(-1)
    var_arr = np.array(var_list).reshape(-1)

    # 5. 反归一化（对应 Cell 10 的前半部分）
    if not test_bearings:
        raise ValueError("test_bearings 为空，无法确定 scaler 文件名")
    
    # 查找 scaler 文件：确保只匹配以 _scaler 结尾的文件
    all_files = os.listdir(scaler_dir)
    # print(f"Scaler目录中的所有文件: {all_files}")
    
    # 优先匹配精确的文件名（如果 test_bearings 已经包含 _labeled_fpt_scaler）
    if test_bearings.endswith('_scaler'):
        matched_files = [f for f in all_files if f == test_bearings or f.endswith(test_bearings)]
    else:
        # 否则匹配包含 test_bearings 且以 _scaler 结尾的文件
        matched_files = [f for f in all_files if test_bearings in f and f.endswith('_scaler')]
    
    # print(f"匹配到的 scaler 文件: {matched_files}")
    
    if not matched_files:
        raise FileNotFoundError(
            f"未找到匹配 {test_bearings} 的 scaler 文件, 目录: {scaler_dir}\n"
            f"请确保文件名包含 '{test_bearings}' 且以 '_scaler' 结尾"
        )
    
    scaler_path = os.path.join(scaler_dir, matched_files[0])
    scaler = load(scaler_path)
    
    # 类型检查：确保加载的是 scaler 对象
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    if not isinstance(scaler, (StandardScaler, MinMaxScaler)):
        raise TypeError(
            f"加载的文件不是 scaler 对象，而是 {type(scaler)}。\n"
            f"文件路径: {scaler_path}\n"
            f"请检查文件是否正确，或是否匹配到了错误的文件。"
        )

    target_arr = scaler.inverse_transform(target_arr.reshape(-1, 1)).reshape(-1)
    prediction_arr = scaler.inverse_transform(prediction_arr.reshape(-1, 1)).reshape(-1)
    if origin_prediction.size > 0:
        origin_prediction = scaler.inverse_transform(origin_prediction.T).T  # 每列对应一个样本

    # 6. 保存基本结果（可选，但与 notebook 行为对齐）
    os.makedirs(results_dir, exist_ok=True)
    base_name = test_bearings

    # 保存原始 target / prediction
    dump(target_arr, os.path.join(results_dir, f"{base_name}_fbtcn_origin"))
    dump(prediction_arr, os.path.join(results_dir, f"{base_name}_fbtcn_pre"))
    dump(var_arr, os.path.join(results_dir, f"{base_name}_fbtcn_var"))
    # 将真实值和全部预测样本保存为 csv：
    # 第一列为 y_true，后面每一列为一次 forward_pass 采样得到的预测值
    import pandas as pd

    if origin_prediction.size > 0:
        # origin_prediction: (forward_pass, N) -> (N, forward_pass)
        preds_matrix = origin_prediction.T
        data_mat = np.concatenate([target_arr.reshape(-1, 1), preds_matrix, var_arr.reshape(-1, 1)], axis=1)
        columns = ["y_true"] + [f"y_pred_sample_{i}" for i in range(preds_matrix.shape[1])] + ["y_var"]
    else:
        # 退化情况：没有采样，只保存均值
        data_mat = np.column_stack([target_arr, prediction_arr, var_arr])
        columns = ["y_true", "y_pred", "y_var"]

    result_df = pd.DataFrame(data_mat, columns=columns)
    result_df.to_csv(
        os.path.join(results_dir, f"{base_name}_fbtcn_result.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    return target_arr, prediction_arr, origin_prediction, var_arr, mu_samples


def evaluate_and_save_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    origin_prediction: np.ndarray,
    var_list: np.ndarray,
    mu_samples: np.ndarray,
    metrics_csv_path: str,
    figure_png_path: str,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    基于 run_test_and_save 的输出，计算并保存全部指标，并绘图。

    注意：本函数不创建任何目录，调用方必须确保传入路径所在目录已存在。
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    y_true = np.asarray(target).reshape(-1)
    y_pred_mean = np.asarray(prediction).reshape(-1)

    y_pred_alea = np.asarray(var_list).reshape(-1)

    # 样本维度：origin_prediction 形状为 (forward_pass, N)
    if origin_prediction is not None and origin_prediction.size > 0:
        y_pred_samples = np.asarray(origin_prediction)
        # print("origin_prediction.shape", origin_prediction.shape)
        # print("mu_samples.shape", mu_samples.shape)
        print("y_pred_samples", y_pred_samples.shape)
        y_pred_epi = y_pred_samples.var(axis=0)
        # y_pred_epi = np.var(mu_samples.squeeze(-1), axis=0)
    else:
        y_pred_samples = None
        y_pred_epi = np.zeros_like(y_pred_alea)
    print("y_pred_epi", y_pred_epi.shape)
    print("y_pred_alea", y_pred_alea.shape)
    # 总不确定性：简单地按方差加和
    y_pred_std_total = np.sqrt(y_pred_alea + y_pred_epi)

    # 构造预测区间（正态近似）
    # z = 1.96 if np.isclose(alpha, 0.05) else torch.distributions.Normal(0, 1).icdf(
    #     torch.tensor(1 - alpha / 2.0)
    # ).item()
    y_lower = y_pred_mean - 1.96 * y_pred_std_total
    y_upper = y_pred_mean + 1.96 * y_pred_std_total

    # 真实 RUL 范围，用于 NMPIW
    R = float(y_true.max() - y_true.min()) if y_true.size > 0 else 1.0

    # 标量指标
    metric_values: Dict[str, float] = {}
    metric_values["MAE"] = float(mae(y_true, y_pred_mean))
    metric_values["RMSE"] = float(rmse(y_true, y_pred_mean))
    picp_val = float(picp(y_true, y_lower, y_upper))
    nmpiw_val = float(nmpiw(y_lower, y_upper, R))
    metric_values["PICP"] = picp_val
    metric_values["NMPIW"] = nmpiw_val
    metric_values["CWC"] = float(cwc(picp_val, nmpiw_val, alpha=alpha))
    metric_values["ECE"] = float(ece(y_true, y_pred_mean, y_pred_alea + y_pred_epi))
    metric_values["Sharpness"] = float(sharpness(y_pred_alea + y_pred_epi, alpha=alpha))
    metric_values["Mean AU"] = float(aleatoric_uncertainty(y_pred_alea))
    metric_values["Mean EU"] = float(y_pred_epi.mean())
    
    # if y_pred_samples is not None:
    # else:
    #     metric_values["Epistemic_Uncertainty"] = float("nan")

    # 打印所有指标
    print("\n===== 测试指标 =====")
    for k, v in metric_values.items():
        print(f"{k}: {v:.6f}" if np.isfinite(v) else f"{k}: NaN")

    # 保存指标到 CSV（单行）
    metrics_df = pd.DataFrame([metric_values])
    metrics_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")

    # 绘制曲线：预测、区间、真实值，以及 AU / EU
    idx = np.arange(len(y_true))
    au_curve = y_pred_alea
    eu_curve = y_pred_epi

    plt.figure(figsize=(12, 6))

    # 子图1：预测与区间
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(idx, y_true, label="True RUL", color="black", linewidth=1.0)
    ax1.plot(idx, y_pred_mean, label="Predicted RUL", color="tab:blue", linewidth=1.0)
    ax1.fill_between(
        idx,
        y_lower,
        y_upper,
        color="tab:blue",
        alpha=0.2,
        label=f"{int((1-alpha)*100)}% PI",
    )
    ax1.set_ylabel("RUL")
    ax1.legend(loc="best")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # 子图2：AU & EU
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(idx, au_curve, label="Aleatoric Uncertainty", color="tab:orange", linewidth=1.0)
    ax2.plot(idx, eu_curve, label="Epistemic Uncertainty", color="tab:green", linewidth=1.0)
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Uncertainty")
    ax2.legend(loc="best")
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(figure_png_path, dpi=200)
    plt.close()

    return metric_values


def save_model_and_config(
    model: torch.nn.Module,
    config: Dict,
    model_path: str,
    config_path: str,
) -> None:
    """
    将模型权重保存为 .pt 文件，并将当前 config 参数保存为 JSON。

    注意：本函数不创建任何目录，调用方必须确保传入路径所在目录已存在。
    """
    # 保存模型权重
    torch.save(model.state_dict(), model_path)

    # 保存配置
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def save_config(
    config: Dict,
    config_path: str,
) -> None:
    """
    将当前 config 参数保存为 JSON。

    注意：本函数不创建任何目录，调用方必须确保传入路径所在目录已存在。
    """
    # 保存配置
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

__all__ = ["run_test_and_save", "evaluate_and_save_metrics", "save_model_and_config"]
