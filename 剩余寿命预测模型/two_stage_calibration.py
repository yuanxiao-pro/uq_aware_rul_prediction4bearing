"""
两步后处理校准 (Two-Stage Post-Processing Calibration)

本文件实现了两种两步后处理校准流程：

(1) 方差缩放 + 保序回归（TwoStageCalibrator）
    - 第一步：方差缩放（Variance Scaling）调整整体方差
    - 第二步：保序回归（Isotonic Regression）单调精细校准

(2) 温度缩放 + 保序回归（TemperatureThenIsotonicCalibrator）
    - 第一步：温度缩放（Temperature Scaling）单参数 T 缩放预测标准差
    - 第二步：保序回归在第一步输出的不确定性上再做单调校准
"""

import numpy as np
from typing import Tuple
from scipy import stats
from prediction_interval_calibration import (
    VarianceScalingCalibrator,
    TemperatureScalingCalibrator,
    IsotonicRegressionCalibrator,
    PredictionIntervalCalibrator,
    evaluate_calibration
)


class TwoStageCalibrator(PredictionIntervalCalibrator):
    """
    两步校准器
    先使用方差缩放校准，然后使用保序回归校准。
    """
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: 显著性水平，预测区间覆盖概率为 1 - alpha
        """
        super().__init__(alpha)
        self.variance_scaling_calibrator = VarianceScalingCalibrator(alpha=alpha)
        self.isotonic_calibrator = IsotonicRegressionCalibrator(alpha=alpha)
    
    def fit(self, y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray):
        """
        在校准集上拟合两步校准器
        
        Args:
            y_true: 真实值，shape (n_samples,)
            y_pred_mean: 预测均值，shape (n_samples,)
            y_pred_std: 预测标准差，shape (n_samples,)
        """
        # 第一步：拟合方差缩放校准器
        self.variance_scaling_calibrator.fit(y_true, y_pred_mean, y_pred_std)
        
        # 使用方差缩放校准器校准校准集，得到第一步校准后的区间
        y_lower_vs, y_upper_vs = self.variance_scaling_calibrator.calibrate(
            y_pred_mean, y_pred_std
        )
        
        # 从第一步校准后的区间中提取校准后的标准差
        z_score = stats.norm.ppf(1 - self.alpha / 2)
        # 计算校准后的标准差：(上界 - 下界) / (2 * z_score)
        calibrated_std_vs = (y_upper_vs - y_lower_vs) / (2 * z_score + 1e-8)
        
        # 第二步：使用第一步校准后的标准差拟合保序回归校准器
        self.isotonic_calibrator.fit(y_true, y_pred_mean, calibrated_std_vs)
        
        self.is_fitted = True
        return self
    
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
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")
        
        # 第一步：使用方差缩放校准
        y_lower_vs, y_upper_vs = self.variance_scaling_calibrator.calibrate(
            y_pred_mean, y_pred_std
        )
        
        # 从第一步校准后的区间中提取校准后的标准差
        z_score = stats.norm.ppf(1 - self.alpha / 2)
        calibrated_std_vs = (y_upper_vs - y_lower_vs) / (2 * z_score + 1e-8)
        
        # 第二步：使用保序回归进一步校准
        y_lower_final, y_upper_final = self.isotonic_calibrator.calibrate(
            y_pred_mean, calibrated_std_vs
        )
        
        return y_lower_final, y_upper_final


class TemperatureThenIsotonicCalibrator(PredictionIntervalCalibrator):
    """
    先温度缩放、再保序回归的两步校准器。
    
    第一步：用温度缩放校准预测不确定性（单参数 T，保持均值不变）；
    第二步：用保序回归在第一步输出的不确定性上再做单调校准。
    """
    
    def __init__(self, alpha: float = 0.05):
        super().__init__(alpha)
        self.temperature_calibrator = TemperatureScalingCalibrator(alpha=alpha)
        self.isotonic_calibrator = IsotonicRegressionCalibrator(alpha=alpha)
    
    def fit(self, y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray):
        # 第一步：拟合温度缩放
        self.temperature_calibrator.fit(y_true, y_pred_mean, y_pred_std)
        y_lower_ts, y_upper_ts = self.temperature_calibrator.calibrate(
            y_pred_mean, y_pred_std
        )
        z_score = stats.norm.ppf(1 - self.alpha / 2)
        calibrated_std_ts = (y_upper_ts - y_lower_ts) / (2 * z_score + 1e-8)
        # 第二步：用第一步校准后的标准差拟合保序回归
        self.isotonic_calibrator.fit(y_true, y_pred_mean, calibrated_std_ts)
        self.is_fitted = True
        return self
    
    def calibrate(self, y_pred_mean: np.ndarray, y_pred_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")
        y_lower_ts, y_upper_ts = self.temperature_calibrator.calibrate(
            y_pred_mean, y_pred_std
        )
        z_score = stats.norm.ppf(1 - self.alpha / 2)
        calibrated_std_ts = (y_upper_ts - y_lower_ts) / (2 * z_score + 1e-8)
        return self.isotonic_calibrator.calibrate(y_pred_mean, calibrated_std_ts)


def fit_two_stage_calibrator(
    y_true_cal: np.ndarray,
    y_pred_mean_cal: np.ndarray,
    y_pred_std_cal: np.ndarray,
    alpha: float = 0.05
) -> TwoStageCalibrator:
    """
    拟合两步校准器
    
    Args:
        y_true_cal: 校准集真实值
        y_pred_mean_cal: 校准集预测均值
        y_pred_std_cal: 校准集预测标准差
        alpha: 显著性水平
    
    Returns:
        calibrator: 拟合好的两步校准器
    """
    calibrator = TwoStageCalibrator(alpha=alpha)
    calibrator.fit(y_true_cal, y_pred_mean_cal, y_pred_std_cal)
    return calibrator


def calibrate_predictions(
    calibrator: TwoStageCalibrator,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用两步校准器校准预测结果
    
    Args:
        calibrator: 已拟合的两步校准器
        y_pred_mean: 预测均值
        y_pred_std: 预测标准差
    
    Returns:
        y_lower: 校准后的下界
        y_upper: 校准后的上界
    """
    return calibrator.calibrate(y_pred_mean, y_pred_std)


def compare_calibration_methods(
    y_true_cal: np.ndarray,
    y_pred_mean_cal: np.ndarray,
    y_pred_std_cal: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_mean_test: np.ndarray,
    y_pred_std_test: np.ndarray,
    alpha: float = 0.05
) -> dict:
    """
    比较不同校准方法的效果（包括两步校准）
    
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
    results = {}
    confidence_level = 1 - alpha
    
    # 评估未校准的预测区间
    z_score = stats.norm.ppf(1 - alpha / 2)
    y_lower_uncal = y_pred_mean_test - z_score * y_pred_std_test
    y_upper_uncal = y_pred_mean_test + z_score * y_pred_std_test
    results['Uncalibrated'] = evaluate_calibration(
        y_true_test, y_lower_uncal, y_upper_uncal, confidence_level
    )
    
    # 评估方差缩放校准
    vs_calibrator = VarianceScalingCalibrator(alpha=alpha)
    vs_calibrator.fit(y_true_cal, y_pred_mean_cal, y_pred_std_cal)
    y_lower_vs, y_upper_vs = vs_calibrator.calibrate(y_pred_mean_test, y_pred_std_test)
    results['VarianceScaling'] = evaluate_calibration(
        y_true_test, y_lower_vs, y_upper_vs, confidence_level
    )
    
    # 评估保序回归校准
    iso_calibrator = IsotonicRegressionCalibrator(alpha=alpha)
    iso_calibrator.fit(y_true_cal, y_pred_mean_cal, y_pred_std_cal)
    y_lower_iso, y_upper_iso = iso_calibrator.calibrate(y_pred_mean_test, y_pred_std_test)
    results['IsotonicRegression'] = evaluate_calibration(
        y_true_test, y_lower_iso, y_upper_iso, confidence_level
    )
    
    # 评估两步校准（方差缩放 + 保序回归）
    two_stage_calibrator = TwoStageCalibrator(alpha=alpha)
    two_stage_calibrator.fit(y_true_cal, y_pred_mean_cal, y_pred_std_cal)
    y_lower_two_stage, y_upper_two_stage = two_stage_calibrator.calibrate(
        y_pred_mean_test, y_pred_std_test
    )
    results['TwoStage'] = evaluate_calibration(
        y_true_test, y_lower_two_stage, y_upper_two_stage, confidence_level
    )
    
    # 评估温度缩放 + 保序回归
    temp_iso_calibrator = TemperatureThenIsotonicCalibrator(alpha=alpha)
    temp_iso_calibrator.fit(y_true_cal, y_pred_mean_cal, y_pred_std_cal)
    y_lower_ti, y_upper_ti = temp_iso_calibrator.calibrate(
        y_pred_mean_test, y_pred_std_test
    )
    results['TemperatureThenIsotonic'] = evaluate_calibration(
        y_true_test, y_lower_ti, y_upper_ti, confidence_level
    )
    
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
    print("=" * 70)
    print("两步后处理校准方法比较")
    print("=" * 70)
    
    results = compare_calibration_methods(
        y_true_cal, y_pred_mean_cal, y_pred_std_cal,
        y_true_test, y_pred_mean_test, y_pred_std_test,
        alpha=0.05
    )
    
    # 打印结果
    print(f"\n目标覆盖概率: {1 - 0.05:.2%}")
    print("\n各方法评估结果:")
    print("-" * 70)
    for method, metrics in results.items():
        print(f"\n{method}:")
        print(f"  PICP: {metrics['PICP']:.4f} (目标: {metrics['Target_Coverage']:.4f})")
        print(f"  覆盖误差: {metrics['Coverage_Error']:.4f}")
        print(f"  NMPIW: {metrics['NMPIW']:.4f}")
        print(f"  MPIW: {metrics['MPIW']:.4f}")
    
    # 使用两步校准器示例
    print("\n" + "=" * 70)
    print("两步校准器使用示例")
    print("=" * 70)
    
    # 拟合校准器
    calibrator = fit_two_stage_calibrator(
        y_true_cal, y_pred_mean_cal, y_pred_std_cal, alpha=0.05
    )
    
    # 校准测试集
    y_lower, y_upper = calibrate_predictions(
        calibrator, y_pred_mean_test, y_pred_std_test
    )
    
    # 评估结果
    metrics = evaluate_calibration(y_true_test, y_lower, y_upper)
    print(f"\n两步校准结果:")
    print(f"  PICP: {metrics['PICP']:.4f}")
    print(f"  覆盖误差: {metrics['Coverage_Error']:.4f}")
    print(f"  NMPIW: {metrics['NMPIW']:.4f}")
    print(f"  MPIW: {metrics['MPIW']:.4f}")

    # 温度缩放 + 保序回归示例
    print("\n" + "=" * 70)
    print("温度缩放 + 保序回归校准示例")
    print("=" * 70)
    cal_ti = TemperatureThenIsotonicCalibrator(alpha=0.05)
    cal_ti.fit(y_true_cal, y_pred_mean_cal, y_pred_std_cal)
    y_lower_ti, y_upper_ti = cal_ti.calibrate(y_pred_mean_test, y_pred_std_test)
    metrics_ti = evaluate_calibration(y_true_test, y_lower_ti, y_upper_ti)
    print(f"  PICP: {metrics_ti['PICP']:.4f}")
    print(f"  覆盖误差: {metrics_ti['Coverage_Error']:.4f}")
    print(f"  NMPIW: {metrics_ti['NMPIW']:.4f}")
    print(f"  MPIW: {metrics_ti['MPIW']:.4f}")
