"""
预测区间校准方法 (Prediction Interval Calibration)

本文件实现了多种预测区间校准方法，用于改进不确定性量化模型的预测区间质量。

## 校准方法优缺点对比

### 1. 分位数校准 (Quantile Calibration)
**优点：**
- 简单直观，易于实现
- 计算效率高
- 适用于任何分位数预测模型
- 不需要额外的模型训练

**缺点：**
- 假设预测误差分布对称（对于中位数校准）
- 对于非对称分布效果较差
- 可能过度调整导致区间过宽或过窄

**适用场景：** 分位数回归模型，需要快速校准的场景

---

### 2. 等渗回归校准 (Isotonic Regression Calibration)
**优点：**
- 保持单调性，不会产生交叉的分位数
- 非参数方法，适应性强
- 对于小样本也能工作
- 理论上保证最优的单调映射

**缺点：**
- 计算复杂度O(n log n)，大数据集较慢
- 可能过拟合小样本
- 需要额外的验证集
- 对于极端值可能不稳定

**适用场景：** 需要保持分位数单调性的场景，中等规模数据集

---

### 3. Platt缩放 (Platt Scaling)
**优点：**
- 简单快速，只需拟合一个sigmoid函数
- 参数少，不易过拟合
- 适用于概率校准

**缺点：**
- 假设sigmoid映射，可能不适合所有分布
- 主要用于二分类，回归任务需要修改
- 对于多峰分布效果差

**适用场景：** 简单的概率校准，二分类任务

---

### 4. 温度缩放 (Temperature Scaling)
**优点：**
- 单参数校准，简单高效
- 不改变预测的排序，只调整置信度
- 适用于神经网络

**缺点：**
- 主要用于分类任务
- 对于回归任务需要修改
- 假设单一的温度参数适用于所有样本

**适用场景：** 神经网络分类任务的不确定性校准

---

### 5. 直方图分箱校准 (Histogram Binning)
**优点：**
- 非参数方法，适应性强
- 实现简单
- 不需要额外的模型训练

**缺点：**
- 分箱数量需要手动选择
- 可能产生不连续的校准函数
- 对于稀疏区域效果差
- 需要足够的样本填充每个箱子

**适用场景：** 快速原型验证，中等规模数据集

---

### 6. 保形预测 (Conformal Prediction)
**优点：**
- 理论保证：在有限样本下也能保证覆盖概率
- 不需要分布假设
- 适用于任何预测模型
- 提供有限样本的统计保证

**缺点：**
- 需要保留校准集，不能用于训练
- 计算复杂度较高（需要计算所有校准样本的残差）
- 对于非交换数据需要特殊处理
- 区间宽度可能不稳定

**适用场景：** 需要理论保证的场景，小样本情况

---

### 7. 分位数回归校准 (Quantile Regression Calibration)
**优点：**
- 直接建模分位数，不需要分布假设
- 可以同时校准多个分位数
- 适用于非对称分布
- 灵活性高

**缺点：**
- 需要训练额外的模型
- 计算成本较高
- 可能过拟合
- 需要选择合适的基函数

**适用场景：** 需要精确控制分位数的场景，有足够训练数据

---

### 8. 方差缩放 (Variance Scaling)
**优点：**
- 简单高效，只需缩放预测方差
- 保持预测均值不变
- 适用于高斯假设的场景

**缺点：**
- 假设误差分布为正态分布
- 对于非高斯分布效果差
- 可能过度或不足缩放

**适用场景：** 高斯过程模型，贝叶斯神经网络

---

## 推荐方法选择

1. **快速原型/简单场景**：分位数校准、方差缩放
2. **需要理论保证**：保形预测
3. **需要保持单调性**：等渗回归校准
4. **大规模数据**：分位数校准、方差缩放
5. **小样本**：保形预测、等渗回归校准
6. **非对称分布**：分位数回归校准、等渗回归校准
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, List
from sklearn.isotonic import IsotonicRegression
from scipy import stats
import warnings


class PredictionIntervalCalibrator:
    """
    预测区间校准器基类
    """
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: 显著性水平，预测区间覆盖概率为 1 - alpha
        """
        self.alpha = alpha
        self.confidence_level = 1 - alpha
        self.is_fitted = False
    
    def fit(self, y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray):
        """
        在校准集上拟合校准器
        
        Args:
            y_true: 真实值，shape (n_samples,)
            y_pred_mean: 预测均值，shape (n_samples,)
            y_pred_std: 预测标准差，shape (n_samples,)
        """
        raise NotImplementedError
    
    def calibrate(self, y_pred_mean: np.ndarray, y_pred_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        校准预测区间
        
        Args:
            y_pred_mean: 预测均值，shape (n_samples,)
            y_pred_std: 预测标准差，shape (n_samples,)
        
        Returns:
            y_lower: 校准后的下界，shape (n_samples,)
            y_upper: 校准后的上界，shape (n_samples,)
        """
        raise NotImplementedError


class QuantileCalibrator(PredictionIntervalCalibrator):
    """
    分位数校准器
    
    通过调整分位数来校准预测区间，使得实际覆盖概率接近目标覆盖概率。
    """
    
    def __init__(self, alpha: float = 0.05, method: str = 'empirical'):
        """
        Args:
            alpha: 显著性水平
            method: 校准方法
                - 'empirical': 经验分位数方法
                - 'normal': 假设正态分布的分位数
        """
        super().__init__(alpha)
        self.method = method
        self.calibration_factor = None
    
    def fit(self, y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray):
        """拟合校准器"""
        residuals = y_true - y_pred_mean
        normalized_residuals = residuals / (y_pred_std + 1e-8)
        
        if self.method == 'empirical':
            # 计算经验分位数
            lower_quantile = np.percentile(normalized_residuals, (self.alpha / 2) * 100)
            upper_quantile = np.percentile(normalized_residuals, (1 - self.alpha / 2) * 100)
            # 计算校准因子：使得覆盖概率为目标值
            target_lower = stats.norm.ppf(self.alpha / 2)
            target_upper = stats.norm.ppf(1 - self.alpha / 2)
            self.calibration_factor = max(
                abs(lower_quantile / target_lower) if target_lower != 0 else 1.0,
                abs(upper_quantile / target_upper) if target_upper != 0 else 1.0
            )
        elif self.method == 'normal':
            # 假设残差服从正态分布，估计缩放因子
            std_estimated = np.std(normalized_residuals)
            self.calibration_factor = std_estimated
        
        self.is_fitted = True
        return self
    
    def calibrate(self, y_pred_mean: np.ndarray, y_pred_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """校准预测区间"""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")
        
        z_score = stats.norm.ppf(1 - self.alpha / 2)
        adjusted_std = y_pred_std * self.calibration_factor
        
        y_lower = y_pred_mean - z_score * adjusted_std
        y_upper = y_pred_mean + z_score * adjusted_std
        
        return y_lower, y_upper


class VarianceScalingCalibrator(PredictionIntervalCalibrator):
    """
    方差缩放校准器
    
    通过缩放预测方差来校准预测区间，保持预测均值不变。
    """
    
    def __init__(self, alpha: float = 0.05):
        super().__init__(alpha)
        self.scaling_factor = None
    
    def fit(self, y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray):
        """拟合校准器"""
        residuals = y_true - y_pred_mean
        # 计算实际残差的方差
        empirical_variance = np.var(residuals)
        # 计算预测方差的平均值
        predicted_variance_mean = np.mean(y_pred_std ** 2)
        
        # 计算缩放因子
        if predicted_variance_mean > 0:
            self.scaling_factor = np.sqrt(empirical_variance / predicted_variance_mean)
        else:
            self.scaling_factor = 1.0
        
        self.is_fitted = True
        return self
    
    def calibrate(self, y_pred_mean: np.ndarray, y_pred_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """校准预测区间"""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")
        
        z_score = stats.norm.ppf(1 - self.alpha / 2)
        adjusted_std = y_pred_std * self.scaling_factor
        
        y_lower = y_pred_mean - z_score * adjusted_std
        y_upper = y_pred_mean + z_score * adjusted_std
        
        return y_lower, y_upper


class TemperatureScalingCalibrator(PredictionIntervalCalibrator):
    """
    温度缩放校准器（回归版本）
    
    通过单一温度参数 T 缩放预测标准差：σ_cal = T * σ。
    保持预测均值不变，在校准集上最小化高斯负对数似然得到 T。
    适用于回归任务的预测区间校准。
    """
    
    def __init__(self, alpha: float = 0.05):
        super().__init__(alpha)
        self.temperature = None
    
    def fit(self, y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray):
        """拟合校准器：最小化校准集上的高斯 NLL 得到温度 T。"""
        residuals_sq = (y_true - y_pred_mean) ** 2
        # 避免除零
        var_pred = np.maximum(y_pred_std ** 2, 1e-12)
        # 闭式解：T^2 = mean((y-mu)^2 / sigma^2) => 使标准化残差平方的均值为 1
        t_sq = np.mean(residuals_sq / var_pred)
        self.temperature = np.sqrt(max(t_sq, 1e-12))
        self.is_fitted = True
        return self
    
    def calibrate(self, y_pred_mean: np.ndarray, y_pred_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """校准预测区间"""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")
        
        z_score = stats.norm.ppf(1 - self.alpha / 2)
        adjusted_std = y_pred_std * self.temperature
        
        y_lower = y_pred_mean - z_score * adjusted_std
        y_upper = y_pred_mean + z_score * adjusted_std
        
        return y_lower, y_upper


class IsotonicRegressionCalibrator(PredictionIntervalCalibrator):
    """
    等渗回归校准器
    
    使用等渗回归来校准预测的不确定性，保持单调性。
    """
    
    def __init__(self, alpha: float = 0.05):
        super().__init__(alpha)
        self.isotonic_regressor = None
    
    def fit(self, y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray):
        """拟合校准器"""
        residuals = np.abs(y_true - y_pred_mean)
        # 使用预测的不确定性作为输入特征
        predicted_uncertainty = y_pred_std
        
        # 拟合等渗回归：实际残差 vs 预测不确定性
        self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_regressor.fit(predicted_uncertainty, residuals)
        
        self.is_fitted = True
        return self
    
    def calibrate(self, y_pred_mean: np.ndarray, y_pred_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """校准预测区间"""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")
        
        # 使用等渗回归预测校准后的不确定性
        calibrated_uncertainty = self.isotonic_regressor.predict(y_pred_std)
        
        z_score = stats.norm.ppf(1 - self.alpha / 2)
        y_lower = y_pred_mean - z_score * calibrated_uncertainty
        y_upper = y_pred_mean + z_score * calibrated_uncertainty
        
        return y_lower, y_upper


class HistogramBinningCalibrator(PredictionIntervalCalibrator):
    """
    直方图分箱校准器
    
    将预测不确定性分成多个箱子，在每个箱子内独立校准。
    """
    
    def __init__(self, alpha: float = 0.05, n_bins: int = 10):
        """
        Args:
            alpha: 显著性水平
            n_bins: 分箱数量
        """
        super().__init__(alpha)
        self.n_bins = n_bins
        self.bin_edges = None
        self.bin_calibration_factors = None
    
    def fit(self, y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray):
        """拟合校准器"""
        residuals = np.abs(y_true - y_pred_mean)
        normalized_residuals = residuals / (y_pred_std + 1e-8)
        
        # 根据预测不确定性分箱
        self.bin_edges = np.linspace(y_pred_std.min(), y_pred_std.max(), self.n_bins + 1)
        bin_indices = np.digitize(y_pred_std, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        # 在每个箱子内计算校准因子
        self.bin_calibration_factors = np.ones(self.n_bins)
        for i in range(self.n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                # 计算该箱子内的实际分位数
                bin_residuals = normalized_residuals[mask]
                if len(bin_residuals) > 0:
                    target_quantile = stats.norm.ppf(1 - self.alpha / 2)
                    empirical_quantile = np.percentile(bin_residuals, (1 - self.alpha / 2) * 100)
                    if target_quantile != 0:
                        self.bin_calibration_factors[i] = abs(empirical_quantile / target_quantile)
        
        self.is_fitted = True
        return self
    
    def calibrate(self, y_pred_mean: np.ndarray, y_pred_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """校准预测区间"""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")
        
        # 确定每个样本属于哪个箱子
        bin_indices = np.digitize(y_pred_std, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        # 获取对应的校准因子
        calibration_factors = self.bin_calibration_factors[bin_indices]
        
        z_score = stats.norm.ppf(1 - self.alpha / 2)
        adjusted_std = y_pred_std * calibration_factors
        
        y_lower = y_pred_mean - z_score * adjusted_std
        y_upper = y_pred_mean + z_score * adjusted_std
        
        return y_lower, y_upper


class ConformalPredictionCalibrator(PredictionIntervalCalibrator):
    """
    保形预测校准器
    
    使用保形预测方法，提供有限样本的统计保证。
    """
    
    def __init__(self, alpha: float = 0.05):
        super().__init__(alpha)
        self.calibration_residuals = None
    
    def fit(self, y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray):
        """拟合校准器（保存校准集的残差）"""
        residuals = np.abs(y_true - y_pred_mean)
        # 归一化残差（使用预测不确定性）
        normalized_residuals = residuals / (y_pred_std + 1e-8)
        self.calibration_residuals = normalized_residuals
        
        self.is_fitted = True
        return self
    
    def calibrate(self, y_pred_mean: np.ndarray, y_pred_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """校准预测区间"""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")
        
        # 计算分位数阈值
        quantile_level = np.ceil((1 - self.alpha) * (len(self.calibration_residuals) + 1)) / len(self.calibration_residuals)
        quantile_level = min(quantile_level, 1.0)
        threshold = np.percentile(self.calibration_residuals, quantile_level * 100)
        
        # 应用阈值到预测不确定性
        y_lower = y_pred_mean - threshold * y_pred_std
        y_upper = y_pred_mean + threshold * y_pred_std
        
        return y_lower, y_upper


class QuantileRegressionCalibrator(PredictionIntervalCalibrator):
    """
    分位数回归校准器
    
    使用分位数回归直接建模分位数，适用于非对称分布。
    """
    
    def __init__(self, alpha: float = 0.05, n_quantiles: int = 5):
        """
        Args:
            alpha: 显著性水平
            n_quantiles: 用于拟合的中间分位数数量
        """
        super().__init__(alpha)
        self.n_quantiles = n_quantiles
        self.lower_model = None
        self.upper_model = None
    
    def fit(self, y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray):
        """拟合校准器"""
        from sklearn.linear_model import QuantileRegressor
        
        residuals = y_true - y_pred_mean
        # 使用预测均值和标准差作为特征
        X = np.column_stack([y_pred_mean, y_pred_std])
        
        # 拟合下分位数回归
        lower_quantile = self.alpha / 2
        self.lower_model = QuantileRegressor(quantile=lower_quantile, alpha=0.0)
        self.lower_model.fit(X, residuals)
        
        # 拟合上分位数回归
        upper_quantile = 1 - self.alpha / 2
        self.upper_model = QuantileRegressor(quantile=upper_quantile, alpha=0.0)
        self.upper_model.fit(X, residuals)
        
        self.is_fitted = True
        return self
    
    def calibrate(self, y_pred_mean: np.ndarray, y_pred_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """校准预测区间"""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")
        
        X = np.column_stack([y_pred_mean, y_pred_std])
        
        # 预测下界和上界的残差
        lower_residual = self.lower_model.predict(X)
        upper_residual = self.upper_model.predict(X)
        
        y_lower = y_pred_mean + lower_residual
        y_upper = y_pred_mean + upper_residual
        
        return y_lower, y_upper


# ==================== 评估函数 ====================

def evaluate_calibration(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, 
                        confidence_level: float = 0.95) -> dict:
    """
    评估预测区间的校准质量
    
    Args:
        y_true: 真实值
        y_lower: 预测区间下界
        y_upper: 预测区间上界
        confidence_level: 目标置信水平
    
    Returns:
        包含各种评估指标的字典
    """
    # PICP (Prediction Interval Coverage Probability)
    within_interval = (y_true >= y_lower) & (y_true <= y_upper)
    picp = np.mean(within_interval)
    
    # 覆盖误差 (Coverage Error)
    coverage_error = abs(picp - confidence_level)
    
    # 平均区间宽度 (Mean Prediction Interval Width)
    mpiw = np.mean(y_upper - y_lower)
    
    # 归一化平均区间宽度 (Normalized Mean Prediction Interval Width)
    range_y = np.max(y_true) - np.min(y_true)
    nmpiw = mpiw / (range_y + 1e-8)
    
    # 区间宽度标准差
    interval_width_std = np.std(y_upper - y_lower)
    
    # 覆盖率统计
    n_samples = len(y_true)
    n_covered = np.sum(within_interval)
    
    return {
        'PICP': picp,
        'Coverage_Error': coverage_error,
        'Target_Coverage': confidence_level,
        'MPIW': mpiw,
        'NMPIW': nmpiw,
        'Interval_Width_Std': interval_width_std,
        'N_Covered': n_covered,
        'N_Samples': n_samples,
        'Coverage_Ratio': n_covered / n_samples
    }


def compare_calibrators(y_true_cal: np.ndarray, y_pred_mean_cal: np.ndarray, y_pred_std_cal: np.ndarray,
                       y_true_test: np.ndarray, y_pred_mean_test: np.ndarray, y_pred_std_test: np.ndarray,
                       alpha: float = 0.05) -> dict:
    """
    比较不同校准方法的效果
    
    Args:
        y_true_cal: 校准集真实值
        y_pred_mean_cal: 校准集预测均值
        y_pred_std_cal: 校准集预测标准差
        y_true_test: 测试集真实值
        y_pred_mean_test: 测试集预测均值
        y_pred_std_test: 测试集预测标准差
        alpha: 显著性水平
    
    Returns:
        包含各校准方法评估结果的字典
    """
    calibrators = {
        'Quantile': QuantileCalibrator(alpha=alpha),
        'VarianceScaling': VarianceScalingCalibrator(alpha=alpha),
        'IsotonicRegression': IsotonicRegressionCalibrator(alpha=alpha),
        'HistogramBinning': HistogramBinningCalibrator(alpha=alpha),
        'ConformalPrediction': ConformalPredictionCalibrator(alpha=alpha),
    }
    
    # 尝试添加分位数回归校准器（如果可用）
    try:
        calibrators['QuantileRegression'] = QuantileRegressionCalibrator(alpha=alpha)
    except ImportError:
        warnings.warn("QuantileRegressionCalibrator requires sklearn >= 1.0.0, skipping...")
    
    results = {}
    confidence_level = 1 - alpha
    
    # 评估未校准的预测区间
    z_score = stats.norm.ppf(1 - alpha / 2)
    y_lower_uncal = y_pred_mean_test - z_score * y_pred_std_test
    y_upper_uncal = y_pred_mean_test + z_score * y_pred_std_test
    results['Uncalibrated'] = evaluate_calibration(y_true_test, y_lower_uncal, y_upper_uncal, confidence_level)
    
    # 评估各校准方法
    for name, calibrator in calibrators.items():
        try:
            # 拟合校准器
            calibrator.fit(y_true_cal, y_pred_mean_cal, y_pred_std_cal)
            
            # 校准测试集
            y_lower_cal, y_upper_cal = calibrator.calibrate(y_pred_mean_test, y_pred_std_test)
            
            # 评估
            results[name] = evaluate_calibration(y_true_test, y_lower_cal, y_upper_cal, confidence_level)
        except Exception as e:
            warnings.warn(f"Error with {name} calibrator: {e}")
            results[name] = {'Error': str(e)}
    
    return results


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 示例：生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    
    # 模拟预测结果（未校准）
    y_true = np.random.randn(n_samples) * 2 + 10
    y_pred_mean = y_true + np.random.randn(n_samples) * 0.5  # 预测均值
    y_pred_std = np.abs(np.random.randn(n_samples) * 0.8 + 1.0)  # 预测标准差（可能不准确）
    
    # 分割校准集和测试集
    split_idx = n_samples // 2
    y_true_cal = y_true[:split_idx]
    y_pred_mean_cal = y_pred_mean[:split_idx]
    y_pred_std_cal = y_pred_std[:split_idx]
    
    y_true_test = y_true[split_idx:]
    y_pred_mean_test = y_pred_mean[split_idx:]
    y_pred_std_test = y_pred_std[split_idx:]
    
    # 比较不同校准方法
    print("=" * 60)
    print("预测区间校准方法比较")
    print("=" * 60)
    
    results = compare_calibrators(
        y_true_cal, y_pred_mean_cal, y_pred_std_cal,
        y_true_test, y_pred_mean_test, y_pred_std_test,
        alpha=0.05
    )
    
    # 打印结果
    print(f"\n目标覆盖概率: {1 - 0.05:.2%}")
    print("\n各方法评估结果:")
    print("-" * 60)
    for method, metrics in results.items():
        if 'Error' not in metrics:
            print(f"\n{method}:")
            print(f"  PICP: {metrics['PICP']:.4f} (目标: {metrics['Target_Coverage']:.4f})")
            print(f"  覆盖误差: {metrics['Coverage_Error']:.4f}")
            print(f"  NMPIW: {metrics['NMPIW']:.4f}")
            print(f"  MPIW: {metrics['MPIW']:.4f}")
    
    # 使用单个校准器示例
    print("\n" + "=" * 60)
    print("单个校准器使用示例")
    print("=" * 60)
    
    calibrator = VarianceScalingCalibrator(alpha=0.05)
    calibrator.fit(y_true_cal, y_pred_mean_cal, y_pred_std_cal)
    y_lower, y_upper = calibrator.calibrate(y_pred_mean_test, y_pred_std_test)
    
    metrics = evaluate_calibration(y_true_test, y_lower, y_upper)
    print(f"\n方差缩放校准结果:")
    print(f"  PICP: {metrics['PICP']:.4f}")
    print(f"  覆盖误差: {metrics['Coverage_Error']:.4f}")
    print(f"  NMPIW: {metrics['NMPIW']:.4f}")

