import numpy as np
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt
def mae(y_true, y_pred):
    """
    Mean Absolute Error (MAE)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def picp(y_true, y_lower, y_upper):
    """
    Calculate the Prediction Interval Coverage Probability (PICP)
    越大越好
    Parameters:
    y_true (array-like): True values.
    y_lower (array-like): Lower bounds of the prediction intervals.
    y_upper (array-like): Upper bounds of the prediction intervals.

    Returns:
    float: PICP value
    """
    y_true = np.array(y_true)
    y_lower = np.array(y_lower)
    y_upper = np.array(y_upper)

    # Keep the decimal places consistent with the label values
    y_lower = np.round(y_lower, 2)
    y_upper = np.round(y_upper, 2)
    y_true = np.round(y_true, 2)

    within_interval = np.logical_and(y_true >= y_lower, y_true <= y_upper)
    picp_value = np.mean(within_interval)
    return picp_value

def nmpiw(lower_bound, upper_bound, R):
    """
    归一化平均预测区间宽度（Normalized Mean Prediction Interval Width, NMPIW）
    """
    return np.sum(upper_bound - lower_bound) / (R * lower_bound.shape[0])

# def ece(y_true, y_pred_mean, y_pred_std, n_bins=10, alpha=0.05):
#     """
#     期望校准误差（Expected Calibration Error, ECE）
#     """
#     z = norm.ppf(1 - alpha / 2)
#     abs_error = np.abs(y_pred_mean - y_true)
#     conf = z * y_pred_std
#     bins = np.linspace(0, np.max(conf), n_bins + 1)
#     ece = 0.0
#     total = len(y_true)
#     for i in range(n_bins):
#         idx = (conf >= bins[i]) & (conf < bins[i + 1])
#         if np.sum(idx) == 0:
#             continue
#         acc = np.mean(abs_error[idx] <= conf[idx])
#         conf_avg = np.mean(conf[idx]) / (np.max(abs_error) + 1e-8)
#         ece += np.abs(acc - conf_avg) * np.sum(idx) / total
#     return ece

def ece(y_true, mu, var, n_bins=20, eps=1e-12):
    """
    Regression ECE via quantile calibration for Gaussian predictive distribution.

    Args:
        y_true: (N,) array
        mu:     (N,) array
        var:    (N,) array, predictive variance
        n_bins: number of quantile levels (e.g., 20, 50)
    Returns:
        ece: float in [0,1]
        qs, empirical: arrays for plotting reliability curve
    """
    y_true = np.asarray(y_true).reshape(-1)
    mu = np.asarray(mu).reshape(-1)
    var = np.asarray(var).reshape(-1)

    sigma = np.sqrt(np.maximum(var, eps))

    qs = np.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins)  # avoid 0/1
    empirical = np.empty_like(qs)

    for j, q in enumerate(qs):
        z = norm.ppf(q)
        thresh = mu + sigma * z
        empirical[j] = np.mean(y_true <= thresh)
    ece = float(np.mean(np.abs(empirical - qs)))
    # return ece, qs, empirical
    return ece

def aleatoric_uncertainty(y_pred_std):
    """
    固有不确定性（Aleatoric Uncertainty, AU）
    输入为预测标准差
    """
    return np.mean(y_pred_std)

def epistemic_uncertainty(y_pred_samples):
    """
    认知不确定性（Epistemic Uncertainty, EU）
    输入为多次采样预测 (T, N)
    """
    return np.mean(np.var(y_pred_samples, axis=0))

def ood_detection(y_pred_samples, threshold=None):
    """
    OOD检测：基于认知不确定性（采样方差）
    threshold: 若为None，返回每个样本的EU，否则返回是否为OOD
    """
    eu = np.std(y_pred_samples, axis=0)
    if threshold is None:
        return eu
    else:
        return (eu > threshold).astype(np.int32)

def sharpness(total_uncertainty, alpha=0.05):
    """
    参考这个论文：A Bayesian Deep Learning Framework for RUL Prediction Incorporating Uncertainty
    Quantification and Calibration

    Sharpness指标：
    total_uncertainty: 总预测不确定性
    alpha: 显著性水平，默认0.05，对应置信度95%
    """
    return np.mean(total_uncertainty)

# def cwc(picp, nmpiw, alpha=0.05, eta=50, beta=0.95):
#     """
#     Coverage Width-based Criterion (CWC)
#     结合区间覆盖率（PICP）和区间宽度（MPIW/NMPIW）进行综合评估。
#     picp: 区间覆盖率
#     nmpiw: 归一化区间宽度
#     alpha: 显著性水平，默认为0.05（置信度95%）
#     eta: 惩罚项指数，默认50
#     beta: 覆盖率目标（通常设置为1-alpha=0.95）
#     返回: CWC值（越小越好）
#     """
#     # CWC惩罚项
#     penalty = np.exp(-eta * (picp - beta)) if picp < beta else 1.0
#     # CWC(标准化)
#     cwc_val = nmpiw * penalty
#     return cwc_val

def cwc(picp, nmpiw, alpha=0.05, eta=2):
    """
    Coverage Width-based Criterion (CWC), lower is better.
    picp: prediction interval coverage probability (PICP)
    nmpiw: normalized mean prediction interval width (NMPIW)
    """
    mu = 1.0 - alpha              # target coverage
    gamma = 1.0 if picp < mu else 0.0
    penalty = gamma * np.exp(-eta * (picp - mu))  # >0 only when under-covered
    return float(nmpiw * (1.0 + penalty))


def nmpiw_torch(y_true, y_pred_mean, y_pred_std, alpha=0.05):
    z = torch.tensor(norm.ppf(1 - alpha / 2), device=y_pred_mean.device, dtype=y_pred_mean.dtype)
    interval_width = 2 * z * y_pred_std
    nmpiw = interval_width.mean() / (y_true.max() - y_true.min() + 1e-8)
    return nmpiw.item()

def ece_torch(y_true, y_pred_mean, y_pred_std, n_bins=10, alpha=0.05):
    z = torch.tensor(norm.ppf(1 - alpha / 2), device=y_pred_mean.device, dtype=y_pred_mean.dtype)
    abs_error = torch.abs(y_pred_mean - y_true)
    conf = z * y_pred_std
    bins = torch.linspace(0, conf.max(), n_bins + 1, device=conf.device)
    ece = 0.0
    total = y_true.numel()
    for i in range(n_bins):
        idx = (conf >= bins[i]) & (conf < bins[i + 1])
        if idx.sum() == 0:
            continue
        acc = (abs_error[idx] <= conf[idx]).float().mean()
        conf_avg = conf[idx].mean() / (abs_error.max() + 1e-8)
        ece += torch.abs(acc - conf_avg) * idx.sum().float() / total
    return ece.item()

def aleatoric_uncertainty_torch(y_pred_std):
    return y_pred_std.mean().item()

def epistemic_uncertainty_torch(y_pred_samples):
    return y_pred_samples.std(dim=0).mean().item()

def ood_detection_torch(y_pred_samples, threshold=None):
    eu = y_pred_samples.std(dim=0)
    if threshold is None:
        return eu
    else:
        return (eu > threshold).int()

def sharpness_torch(y_pred_std, alpha=0.05):
    z = torch.tensor(norm.ppf(1 - alpha / 2), device=y_pred_std.device, dtype=y_pred_std.dtype)
    interval_width = 2 * z * y_pred_std
    return interval_width.mean().item()

def plot_calibration(y_true, y_pred_mean, y_pred_std, n_bins=10, alpha=0.05, 
                     figsize=(8, 6), save_path=None, show_plot=True):
    """
    绘制校准图（Calibration Plot）
    
    用于可视化模型的不确定性校准情况。理想情况下，实际覆盖率应该等于预测置信度，
    即图中的点应该落在对角线上。
    
    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred_mean : array-like
        预测均值
    y_pred_std : array-like
        预测标准差
    n_bins : int, default=10
        分箱数量
    alpha : float, default=0.05
        显著性水平，默认0.05对应95%置信度
    figsize : tuple, default=(8, 6)
        图像大小
    save_path : str, optional
        保存路径，如果为None则不保存
    show_plot : bool, default=True
        是否显示图像
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        图像对象
    ax : matplotlib.axes.Axes
        坐标轴对象
    """
    y_true = np.array(y_true)
    y_pred_mean = np.array(y_pred_mean)
    y_pred_std = np.array(y_pred_std)
    
    # 计算置信度区间
    z = norm.ppf(1 - alpha / 2)
    abs_error = np.abs(y_pred_mean - y_true)
    conf = z * y_pred_std
    
    # 将置信度分成多个区间
    bins = np.linspace(0, np.max(conf), n_bins + 1)
    
    # 计算每个区间的实际覆盖率和平均置信度
    actual_coverage = []
    expected_coverage = []
    bin_counts = []
    
    for i in range(n_bins):
        idx = (conf >= bins[i]) & (conf < bins[i + 1])
        if i == n_bins - 1:  # 最后一个区间包含最大值
            idx = (conf >= bins[i]) & (conf <= bins[i + 1])
        
        if np.sum(idx) == 0:
            continue
        
        # 实际覆盖率：误差在置信区间内的比例
        acc = np.mean(abs_error[idx] <= conf[idx])
        # 期望覆盖率：归一化的平均置信度
        conf_avg = np.mean(conf[idx]) / (np.max(abs_error) + 1e-8)
        
        actual_coverage.append(acc)
        expected_coverage.append(conf_avg)
        bin_counts.append(np.sum(idx))
    
    actual_coverage = np.array(actual_coverage)
    expected_coverage = np.array(expected_coverage)
    bin_counts = np.array(bin_counts)
    
    # 绘制校准图
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制对角线（理想校准线）
    min_val = min(np.min(actual_coverage), np.min(expected_coverage))
    max_val = max(np.max(actual_coverage), np.max(expected_coverage))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', 
            label='Perfect Calibration', linewidth=2, alpha=0.7)
    
    # 绘制实际校准曲线
    ax.plot(expected_coverage, actual_coverage, 'o-', 
            label='Model Calibration', linewidth=2, markersize=8)
    
    # 根据每个bin的样本数量调整点的大小
    if len(bin_counts) > 0:
        sizes = 100 + 500 * (bin_counts / np.max(bin_counts))
        for i, (x, y, size) in enumerate(zip(expected_coverage, actual_coverage, sizes)):
            ax.scatter(x, y, s=size, alpha=0.5, zorder=5)
    
    ax.set_xlabel('Expected Coverage (Normalized Confidence)', fontsize=12)
    ax.set_ylabel('Actual Coverage', fontsize=12)
    ax.set_title(f'Calibration Plot (α={alpha}, {int((1-alpha)*100)}% Confidence)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig, ax
