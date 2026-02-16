"""
等渗回归校准模块
Isotonic Regression Calibration Module

提供等渗回归校准的完整功能，包括：
1. 在上下文数据集上进行预测
2. 拟合等渗回归校准器
3. 对测试集进行校准
4. 计算和保存校准后的结果
"""

import os
import sys
import numpy as np
import torch
import torch.utils.data as Data
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from joblib import load
from typing import Dict, List, Tuple, Optional, Any
import logging
import re

# 设置matplotlib支持中文显示（宋体）和西文（Times New Roman）
try:
    # 尝试从通用配置模块导入
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from matplotlib_chinese_config import setup_chinese_font
    # 设置中文字体为宋体，西文字体为Times New Roman
    setup_chinese_font(chinese_font_name='SimSun', western_font_name='Times New Roman')
except ImportError:
    # 如果导入失败，使用本地配置
    def setup_chinese_font():
        """配置matplotlib以支持中文显示（宋体）和西文（Times New Roman）"""
        available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
        
        # 中文字体：宋体
        chinese_font_list = ['SimSun', 'NSimSun', 'STSong', 'Songti SC']
        chinese_font = None
        for font in chinese_font_list:
            if font in available_fonts:
                chinese_font = font
                break
        
        # 西文字体：Times New Roman
        western_font_list = ['Times New Roman', 'Times', 'DejaVu Serif']
        western_font = None
        for font in western_font_list:
            if font in available_fonts:
                western_font = font
                break
        
        if chinese_font:
            plt.rcParams['font.sans-serif'] = [chinese_font] + plt.rcParams['font.sans-serif']
        if western_font:
            plt.rcParams['font.serif'] = [western_font] + plt.rcParams['font.serif']
            plt.rcParams['mathtext.fontset'] = 'stix'
        
        plt.rcParams['axes.unicode_minus'] = False
        return chinese_font, western_font
    
    setup_chinese_font()

# 添加路径以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from prediction_interval_calibration import IsotonicRegressionCalibrator, evaluate_calibration
from metrics import mae, rmse, picp, nmpiw, ece, cwc, sharpness, aleatoric_uncertainty, epistemic_uncertainty


def normalize_name(name: str) -> str:
    """去除数据集前缀，只保留从 'Bearing' 开始的部分，用于重叠检查"""
    if 'Bearing' in name:
        return name[name.index('Bearing'):]
    return name


def condition_group(name: str) -> str:
    """
    将物理轴承名进一步映射为"工况编号"，例如：
    - Bearing1_1 / Bearing1_5 -> '1'
    - Bearing3_2 / Bearing3_5 -> '3'
    这样就可以按工况 1/2/3 来划分。
    """
    def condition_key(name: str) -> str:
        """将各种文件名/前缀还原为"物理轴承名"的工况键"""
        base = normalize_name(name)
        # 优先用正则直接抽取 BearingX_Y
        m = re.search(r'Bearing\d+_\d+', base)
        if m:
            return m.group(0)
        parts = base.split('_')
        return '_'.join(parts[:2]) if len(parts) >= 2 else base
    
    key = condition_key(name)  # e.g. Bearing3_5
    m = re.match(r'Bearing(\d+)_\d+', key)
    if m:
        return m.group(1)
    # 兜底：若不符合模式，则直接返回 key
    return key


def filter_validation_by_condition(validation_bearings: List[str], test_bearings: List[str]) -> List[str]:
    """
    根据测试集的工况筛选验证集，只返回与测试集同工况的验证集轴承
    
    Args:
        validation_bearings: 验证集轴承列表
        test_bearings: 测试集轴承列表
    
    Returns:
        与测试集同工况的验证集轴承列表
    """
    if not validation_bearings or not test_bearings:
        return []
    
    # 获取测试集的工况编号集合
    test_conditions = set()
    for test_bearing in test_bearings:
        condition = condition_group(test_bearing)
        test_conditions.add(condition)
    
    # 筛选出与测试集同工况的验证集轴承
    filtered_validation = []
    for val_bearing in validation_bearings:
        val_condition = condition_group(val_bearing)
        if val_condition in test_conditions:
            filtered_validation.append(val_bearing)
    
    return filtered_validation


def get_logger():
    """获取logger"""
    logger = logging.getLogger('FBTCN_Training')
    if not logger.handlers:
        logger = logging.getLogger(__name__)
    return logger


def predict_on_calibration_dataset(
    model: torch.nn.Module,
    calibration_bearings: List[str],
    train_data_dir: str,
    scaler_dir: str,
    test_bearings: List[str],
    test_datasets_type: str,
    config: Dict,
    device: torch.device,
    load_bearing_data_func,
    test_data_dir: Optional[str] = None,
    use_test_data_dir: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    在校准数据集上进行预测，用于拟合等渗回归校准器
    
    Args:
        model: 训练好的模型
        calibration_bearings: 校准数据集轴承列表（通常是训练集的第一个子集）
        train_data_dir: 训练数据目录
        scaler_dir: scaler目录
        test_bearings: 测试轴承列表（用于确定scaler）
        test_datasets_type: 测试数据集类型
        config: 配置字典
        device: 设备
        load_bearing_data_func: 加载轴承数据的函数
        test_data_dir: 测试数据目录（当use_test_data_dir=True时使用）
        use_test_data_dir: 是否使用测试数据目录加载校准数据（默认False，使用train_data_dir）
    
    Returns:
        val_target: 校准数据集的真实值（反归一化后）
        val_prediction: 校准数据集的预测均值（反归一化后）
        val_pred_std: 校准数据集的预测标准差
    """
    logger = get_logger()
    
    logger.info("\n📊 使用校准数据集进行预测，用于拟合等渗回归校准器...")
    logger.info(f"校准数据集轴承: {calibration_bearings}")
    
    # 根据use_test_data_dir决定从哪个目录加载数据
    data_dir = test_data_dir if use_test_data_dir and test_data_dir is not None else train_data_dir
    if use_test_data_dir:
        logger.info(f"从测试数据目录加载校准数据: {data_dir}")
    else:
        logger.info(f"从训练数据目录加载校准数据: {data_dir}")
    
    # 加载校准数据集
    calibration_set, calibration_label = load_bearing_data_func(calibration_bearings, data_dir)
    if len(calibration_set) == 0:
        logger.warning("⚠️  警告: 校准数据集为空，跳过校准")
        return None, None, None
    
    # 创建校准数据加载器
    calibration_loader = Data.DataLoader(
        dataset=Data.TensorDataset(calibration_set, calibration_label),
        batch_size=config['test_batch_size'], num_workers=0, drop_last=False,
        pin_memory=True, persistent_workers=False
    )
    
    # 在校准数据集上进行预测
    val_target = []
    val_prediction = []
    val_var_list = []
    val_mu_samples_list = []
    
    model.eval()
    with torch.no_grad():
        for data, label in calibration_loader:
            origin_label = label.tolist()
            val_target += origin_label
            
            data = data.to(device)
            label = label.to(device)
            
            mu_list = []
            log_var = []
            for _ in range(config['forward_pass']):
                mu, sigma, kl = model(data)
                mu_list.append(mu.cpu().numpy())
                log_var.append(sigma.cpu().numpy())
            
            mu_samples = np.stack(mu_list, axis=0)
            sigma_samples = np.stack(log_var, axis=0)
            
            mu_mean = np.mean(mu_samples, axis=0)
            sigma_mean = np.mean(sigma_samples, axis=0)
            val_prediction += mu_mean.squeeze(-1).tolist()
            val_var_list += sigma_mean.squeeze(-1).tolist()
            val_mu_samples_list.append(mu_samples.squeeze(-1))
    
    # 反归一化校准数据集预测结果
    # 使用第一个校准轴承的scaler（按照 notebook 中的方法）
    if len(calibration_bearings) > 0:
        first_bearing = calibration_bearings[0]
        bearing_name_for_scaler = first_bearing
        # 根据数据集类型确定 scaler 文件名
        if test_datasets_type == 'xjtu_made' or test_datasets_type == 'xjtu_made_v3':
            bearing_name_for_scaler = first_bearing.replace("_labeled", "_labeled_fpt_scaler")
        elif test_datasets_type == 'femto_made':
            bearing_name_for_scaler = first_bearing.replace("_labeled", "_labeled_fpt_scaler")
        
        scaler_path = os.path.join(scaler_dir, test_datasets_type, bearing_name_for_scaler)
        if not os.path.exists(scaler_path):
            # 尝试不带数据集类型前缀的路径
            scaler_path = os.path.join(scaler_dir, bearing_name_for_scaler)
        
        if os.path.exists(scaler_path):
            scaler = load(scaler_path)
            val_target = scaler.inverse_transform(np.array(val_target).reshape(-1, 1)).reshape(-1)
            val_prediction = scaler.inverse_transform(np.array(val_prediction).reshape(-1, 1)).reshape(-1)
        else:
            logger.warning(f"⚠️  警告: 未找到scaler文件 {scaler_path}，跳过校准")
            return None, None, None
    else:
        logger.warning("⚠️  警告: 没有校准轴承，无法确定scaler，跳过校准")
        return None, None, None
    
    val_au = np.array(val_var_list)
    val_eu = np.var(np.concatenate(val_mu_samples_list, axis=1), axis=0) if len(val_mu_samples_list) > 0 else np.zeros(len(val_target))
    
    # 计算校准数据集的预测标准差（用于校准）
    val_pred_std = np.sqrt(val_au + val_eu)
    
    logger.info(f"校准数据集样本数: {len(val_target)}")
    logger.info(f"校准数据集预测均值范围: [{val_prediction.min():.4f}, {val_prediction.max():.4f}]")
    logger.info(f"校准数据集预测标准差范围: [{val_pred_std.min():.4f}, {val_pred_std.max():.4f}]")
    
    return val_target, val_prediction, val_pred_std


def fit_calibrator(
    val_target: np.ndarray,
    val_prediction: np.ndarray,
    val_pred_std: np.ndarray,
    config: Dict
) -> IsotonicRegressionCalibrator:
    """
    拟合等渗回归校准器
    
    Args:
        val_target: 验证集真实值
        val_prediction: 验证集预测均值
        val_pred_std: 验证集预测标准差
        config: 配置字典
    
    Returns:
        calibrator: 拟合好的校准器
    """
    logger = get_logger()
    
    logger.info("\n🔧 拟合等渗回归校准器...")
    
    alpha = 1 - config.get('ci', 0.95)  # 从配置中获取置信水平
    calibrator = IsotonicRegressionCalibrator(alpha=alpha)
    calibrator.fit(
        y_true=val_target,
        y_pred_mean=val_prediction,
        y_pred_std=val_pred_std
    )
    logger.info("✓ 校准器拟合完成")
    
    return calibrator


def calibrate_test_results(
    calibrator: IsotonicRegressionCalibrator,
    first_test_results: Dict[str, Dict],
    config: Dict,
    res_dir: str
) -> None:
    """
    对测试集进行校准并保存结果
    
    Args:
        calibrator: 拟合好的校准器
        first_test_results: 第一次测试结果字典
        config: 配置字典
        res_dir: 结果保存目录
    """
    logger = get_logger()
    
    logger.info(f"\n{'='*80}")
    logger.info("第二次测试（校准后）")
    logger.info(f"{'='*80}")
    
    for bearing_name, first_result in first_test_results.items():
        target = first_result['target']
        prediction = first_result['prediction']
        origin_prediction = first_result['origin_prediction']
        log_var_list = first_result['log_var_list']
        mu_samples = first_result['mu_samples']
        
        # 计算测试集的预测标准差
        au = log_var_list
        eu = np.var(origin_prediction, axis=0) if origin_prediction.size > 0 else np.zeros(len(target))
        
        # au 是对数方差（log variance），需要转换为方差（variance）
        # eu 已经是方差（variance）
        # 总方差 = variance_au + eu，然后计算标准差
        variance_au = np.exp(np.clip(au, -20, 10))  # 防止溢出，clip到合理范围
        total_variance = variance_au + np.maximum(eu, 0)  # 确保eu非负
        test_pred_std = np.sqrt(np.maximum(total_variance, 1e-8))  # 防止负值或零值导致NaN
        
        # 检查并处理NaN
        nan_mask = np.isnan(test_pred_std) | np.isinf(test_pred_std)
        if np.any(nan_mask):
            logger.warning(f"⚠️  警告: 检测到 {np.sum(nan_mask)} 个NaN/Inf值，使用默认值替换")
            default_std = np.nanmedian(test_pred_std[~nan_mask]) if np.any(~nan_mask) else 1.0
            test_pred_std[nan_mask] = default_std if not np.isnan(default_std) else 1.0
        
        # 对测试集进行校准
        logger.info(f"\n📈 对测试集 {bearing_name} 进行校准...")
        y_lower_calibrated, y_upper_calibrated = calibrator.calibrate(
            y_pred_mean=prediction,
            y_pred_std=test_pred_std
        )
        
        # 从校准后的区间反推出校准后的不确定性
        # 等渗回归校准器使用 z_score * calibrated_uncertainty 来计算区间
        # 所以：calibrated_uncertainty = (y_upper - y_lower) / (2 * z_score)
        from scipy import stats
        alpha = 1 - config.get('ci', 0.95)
        z_score = stats.norm.ppf(1 - alpha / 2)
        calibrated_uncertainty = (y_upper_calibrated - y_lower_calibrated) / (2 * z_score)
        # 转换为方差（ECE函数需要方差）
        calibrated_variance = calibrated_uncertainty ** 2
        
        # 评估校准后的效果
        metrics_after = evaluate_calibration(
            y_true=target,
            y_lower=y_lower_calibrated,
            y_upper=y_upper_calibrated,
            confidence_level=config.get('ci', 0.95)
        )
        
        logger.info(f"\n📊 校准后指标 (轴承: {bearing_name}):")
        logger.info(f"  PICP: {metrics_after['PICP']:.6f}")
        logger.info(f"  Coverage Error: {metrics_after['Coverage_Error']:.6f}")
        logger.info(f"  NMPIW: {metrics_after['NMPIW']:.6f}")
        logger.info(f"  MPIW: {metrics_after['MPIW']:.6f}")
        
        # 保存校准后的结果（文件名添加_calibrated后缀，使用os.path.join确保路径正确）
        calibrated_csv_path = os.path.join(res_dir, f"{bearing_name}_calibrated.csv")
        calibrated_png_path = os.path.join(res_dir, f"{bearing_name}_calibrated.png")
        
        # 重新计算指标（使用校准后的区间）
        y_true = target
        y_pred_mean = prediction
        y_pred_alea = au
        y_pred_epi = eu
        y_pred_std_total = test_pred_std  # 保留校准前的不确定性（用于对比）
        y_pred_std_calibrated = calibrated_uncertainty  # 校准后的不确定性
        
        # 使用校准后的区间
        y_lower = y_lower_calibrated
        y_upper = y_upper_calibrated
        
        # 计算指标
        R = float(y_true.max() - y_true.min()) if y_true.size > 0 and y_true.max() != y_true.min() else 1.0
        metric_values = {}
        metric_values["MAE"] = float(mae(y_true, y_pred_mean))
        metric_values["RMSE"] = float(rmse(y_true, y_pred_mean))
        metric_values["PICP"] = float(picp(y_true, y_lower, y_upper))
        metric_values["NMPIW"] = float(nmpiw(y_lower, y_upper, R))
        # metric_values["MPIW"] = float(np.mean(y_upper - y_lower))
        # metric_values["Coverage_Error"] = abs(metric_values["PICP"] - config.get('ci', 0.95))
        metric_values["CWC"] = float(cwc(metric_values["PICP"], metric_values["NMPIW"]))
        # 使用校准后的不确定性计算ECE（反映校准后的不确定性质量）
        metric_values["ECE"] = float(ece(y_true, y_pred_mean, calibrated_variance))
        metric_values["Sharpness"] = float(sharpness(calibrated_uncertainty, alpha=config.get('ci', 0.95)))
        metric_values["Mean AU"] = float(np.mean(y_pred_alea))
        metric_values["Mean EU"] = float(np.mean(y_pred_epi))
        
        # 保存CSV（包含指标和校准后的预测区间）
        # 创建包含所有数据的DataFrame
        data_dict = {
            'y_true': y_true,
            'y_pred_mean': y_pred_mean,
            'y_lower_calibrated': y_lower_calibrated,
            'y_upper_calibrated': y_upper_calibrated,
            'y_pred_std_total': y_pred_std_total,  # 校准前的不确定性
            'y_pred_std_calibrated': y_pred_std_calibrated,  # 校准后的不确定性
            'y_pred_alea': y_pred_alea,
            'y_pred_epi': y_pred_epi
        }
        data_df = pd.DataFrame(data_dict)
        
        # 保存数据到CSV
        data_df.to_csv(calibrated_csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"✓ 校准后数据已保存到: {calibrated_csv_path}")
        
        # 保存指标到单独的CSV（可选，如果需要单独保存指标）
        metrics_csv_path = os.path.join(res_dir, f"{bearing_name}_calibrated_metrics.csv")
        metrics_df = pd.DataFrame([metric_values])
        metrics_df.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"✓ 校准后指标已保存到: {metrics_csv_path}")
        
        # 绘制并保存图片
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(y_true))
        ax.plot(x, y_true, 'k-', label="True RUL", linewidth=2)
        ax.plot(x, y_pred_mean, 'b--', label="Predicted Mean", linewidth=1.5)
        ax.fill_between(x, y_lower_calibrated, y_upper_calibrated, 
                      color='lime', alpha=0.25, label='Calibrated Interval')
        ax.legend(loc='best')
        ax.set_title(f"Calibrated - {bearing_name} PI")
        ax.set_xlabel("TIME")
        ax.set_ylabel("RUL")
        plt.tight_layout()
        plt.savefig(calibrated_png_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ 校准后图片已保存到: {calibrated_png_path}")


def run_isotonic_calibration(
    model: torch.nn.Module,
    train_bearings: List[str],
    train_data_dir: str,
    scaler_dir: str,
    test_bearings: List[str],
    test_datasets_type: str,
    first_test_results: Dict[str, Dict],
    config: Dict,
    device: torch.device,
    res_dir: str,
    load_bearing_data_func,
    calibration_bearings: Optional[List[str]] = None,
    calibration_mode: str = 'train_first',
    context_bearings: Optional[List[str]] = None,
    use_same_condition_validation: bool = True,
    test_data_dir: Optional[str] = None
) -> None:
    """
    运行完整的等渗回归校准流程
    
    Args:
        model: 训练好的模型
        train_bearings: 训练集轴承列表
        train_data_dir: 训练数据目录
        scaler_dir: scaler目录
        test_bearings: 测试轴承列表
        test_datasets_type: 测试数据集类型
        first_test_results: 第一次测试结果字典
        config: 配置字典
        device: 设备
        res_dir: 结果保存目录
        load_bearing_data_func: 加载轴承数据的函数
        calibration_bearings: 自定义校准数据集轴承列表（如果提供，将优先使用）
        calibration_mode: 校准数据选择模式，可选值：
            - 'train_first': 使用训练集的第一个子集（默认）
            - 'train_all': 使用所有训练集
            - 'context': 使用上下文数据集（需要提供context_bearings）
            - 'custom': 使用自定义轴承列表（需要提供calibration_bearings）
            - 'test': 使用测试集作为校准集（使用test_bearings）
        context_bearings: 上下文轴承列表（当calibration_mode='context'时使用）
        use_same_condition_validation: 是否只使用与测试集同工况的验证集（默认True）
        test_data_dir: 测试数据目录（当calibration_mode='test'时使用）
    """
    logger = get_logger()
    logger.info(f"⚠️  calibration_mode: {calibration_mode}")
    if len(first_test_results) == 0:
        logger.warning("⚠️  警告: 没有第一次测试结果，跳过校准")
        return
    
    logger.info(f"\n{'='*80}")
    logger.info("等渗回归校准")
    logger.info(f"{'='*80}")
    # logger.info(f"first_test_results: {first_test_results}")
    # 根据参数选择校准数据集
    # 优先检查 calibration_mode，如果指定了特定模式，则忽略 calibration_bearings
    if calibration_mode == 'test':
        # 使用测试集作为校准集：每个测试集单独训练一个校准器
        if len(test_bearings) == 0:
            logger.warning("⚠️  警告: 测试集为空，跳过校准")
            return
        
        logger.info(f"使用测试集作为校准集（每个测试集单独训练校准器）")
        logger.info(f"测试集轴承: {test_bearings}")
        
        # 对每个测试集单独处理
        for test_bearing_original in test_bearings:
            # 找到对应的测试结果
            # 首先尝试直接匹配
            matching_key = None
            if test_bearing_original in first_test_results:
                matching_key = test_bearing_original
            else:
                # 尝试匹配bearing_name（可能名称格式不同）
                # 提取基础名称（去掉可能的后缀）
                base_name = test_bearing_original
                # 去掉常见的后缀
                for suffix in ['_labeled', '_labeled_fpt', '_labeled_fpt_scaler', '_fpt', '_fpt_scaler']:
                    if base_name.endswith(suffix):
                        base_name = base_name[:-len(suffix)]
                        break
                
                # 在first_test_results中查找匹配的键
                for key in first_test_results.keys():
                    # 检查是否包含基础名称
                    if base_name in key or key in base_name:
                        matching_key = key
                        break
                    # 也检查去掉后缀后的key
                    key_base = key
                    for suffix in ['_labeled', '_labeled_fpt', '_labeled_fpt_scaler', '_fpt', '_fpt_scaler']:
                        if key_base.endswith(suffix):
                            key_base = key_base[:-len(suffix)]
                            break
                    if base_name == key_base or base_name in key_base or key_base in base_name:
                        matching_key = key
                        break
            
            if matching_key is None:
                logger.warning(f"⚠️  警告: 未找到测试集 {test_bearing_original} 的测试结果，跳过")
                logger.warning(f"  可用的测试结果键: {list(first_test_results.keys())}")
                continue
            
            test_bearing_name = matching_key
            logger.info(f"匹配测试集: {test_bearing_original} -> {test_bearing_name}")
            
            first_result = first_test_results[test_bearing_name]
            target = first_result['target']
            prediction = first_result['prediction']
            origin_prediction = first_result['origin_prediction']
            log_var_list = first_result['log_var_list']
            mu_samples = first_result['mu_samples']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"为测试集 {test_bearing_name} 单独训练校准器")
            logger.info(f"{'='*60}")
            
            # 计算该测试集的预测标准差
            au = log_var_list
            eu = np.var(origin_prediction, axis=0) if origin_prediction.size > 0 else np.zeros(len(target))
            # eu = np.var(np.concatenate(mu_samples, axis=1), axis=0) if len(mu_samples) > 0 else np.zeros(len(target))
            
            # au 是对数方差（log variance），需要转换为方差（variance）
            # eu 已经是方差（variance）
            # 总方差 = variance_au + eu，然后计算标准差
            variance_au = np.exp(np.clip(au, -20, 10))  # 防止溢出，clip到合理范围
            total_variance = variance_au + np.maximum(eu, 0)  # 确保eu非负
            test_pred_std = np.sqrt(np.maximum(total_variance, 1e-8))  # 防止负值或零值导致NaN
            
            # 检查并处理NaN
            nan_mask = np.isnan(test_pred_std) | np.isinf(test_pred_std)
            if np.any(nan_mask):
                logger.warning(f"⚠️  警告: 检测到 {np.sum(nan_mask)} 个NaN/Inf值，使用默认值替换")
                default_std = np.nanmedian(test_pred_std[~nan_mask]) if np.any(~nan_mask) else 1.0
                test_pred_std[nan_mask] = default_std if not np.isnan(default_std) else 1.0
            
            # 使用该测试集的数据训练校准器
            logger.info(f"使用测试集 {test_bearing_name} 的数据训练校准器...")
            calibrator = fit_calibrator(
                val_target=target,
                val_prediction=prediction,
                val_pred_std=test_pred_std,
                config=config
            )
            
            # 使用该校准器对该测试集进行校准
            logger.info(f"使用训练好的校准器对测试集 {test_bearing_name} 进行校准...")
            logger.info(f"prediction.shape: {prediction.shape}")
            logger.info(f"test_pred_std.shape: {test_pred_std.shape}")
            # 对测试集进行校准
            y_lower_calibrated, y_upper_calibrated = calibrator.calibrate(
                y_pred_mean=prediction,
                y_pred_std=test_pred_std
            )
            
            # 从校准后的区间反推出校准后的不确定性
            from scipy import stats
            alpha = 1 - config.get('ci', 0.95)
            # z_score = stats.norm.ppf(1 - alpha / 2)
            calibrated_uncertainty = (y_upper_calibrated - y_lower_calibrated) / (1.96*2)
            calibrated_variance = calibrated_uncertainty ** 2
            
            # 评估校准后的效果
            metrics_after = evaluate_calibration(
                y_true=target,
                y_lower=y_lower_calibrated,
                y_upper=y_upper_calibrated,
                confidence_level=config.get('ci', 0.95)
            )
            
            logger.info(f"\n📊 校准后指标 (轴承: {test_bearing_name}):")
            logger.info(f"  PICP: {metrics_after['PICP']:.6f}")
            logger.info(f"  Coverage Error: {metrics_after['Coverage_Error']:.6f}")
            logger.info(f"  NMPIW: {metrics_after['NMPIW']:.6f}")
            logger.info(f"  MPIW: {metrics_after['MPIW']:.6f}")
            
            # 保存校准后的结果（使用os.path.join确保路径正确）
            calibrated_csv_path = os.path.join(res_dir, f"{test_bearing_name}_calibrated.csv")
            calibrated_png_path = os.path.join(res_dir, f"{test_bearing_name}_calibrated.png")
            
            # 重新计算指标（使用校准后的区间）
            y_true = target
            y_pred_mean = prediction
            y_pred_alea = au
            y_pred_epi = eu
            y_pred_std_total = test_pred_std
            y_pred_std_calibrated = calibrated_uncertainty
            
            # 使用校准后的区间
            y_lower = y_lower_calibrated
            y_upper = y_upper_calibrated
            
            # 计算指标
            R = float(y_true.max() - y_true.min()) if y_true.size > 0 and y_true.max() != y_true.min() else 1.0
            metric_values = {}
            metric_values["MAE"] = float(mae(y_true, y_pred_mean))
            metric_values["RMSE"] = float(rmse(y_true, y_pred_mean))
            metric_values["PICP"] = float(picp(y_true, y_lower, y_upper))
            metric_values["NMPIW"] = float(nmpiw(y_lower, y_upper, R))
            metric_values["CWC"] = float(cwc(metric_values["PICP"], metric_values["NMPIW"], alpha=alpha))
            metric_values["ECE"] = float(ece(y_true, y_pred_mean, calibrated_variance))
            metric_values["Sharpness"] = float(sharpness(calibrated_variance, alpha=alpha))
            metric_values["Aleatoric_Uncertainty"] = float(np.mean(y_pred_alea))
            metric_values["Epistemic_Uncertainty"] = float(np.mean(y_pred_epi))
            
            # 保存CSV
            data_dict = {
                'y_true': y_true,
                'y_pred_mean': y_pred_mean,
                'y_lower_calibrated': y_lower_calibrated,
                'y_upper_calibrated': y_upper_calibrated,
                'y_pred_std_total': y_pred_std_total,
                'y_pred_std_calibrated': y_pred_std_calibrated,
                'y_pred_alea': y_pred_alea,
                'y_pred_epi': y_pred_epi
            }
            data_df = pd.DataFrame(data_dict)
            data_df.to_csv(calibrated_csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"✓ 校准后数据已保存到: {calibrated_csv_path}")
            
            # 保存指标
            metrics_csv_path = os.path.join(res_dir, f"{test_bearing_name}_calibrated_metrics.csv")
            metrics_df = pd.DataFrame([metric_values])
            metrics_df.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"✓ 校准后指标已保存到: {metrics_csv_path}")
            
            # 绘制并保存图片
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(y_true))
            ax.plot(x, y_true, 'k-', label="True RUL", linewidth=2)
            ax.plot(x, y_pred_mean, 'b--', label="Predicted Mean", linewidth=1.5)
            ax.fill_between(x, y_lower_calibrated, y_upper_calibrated, 
                          color='lime', alpha=0.25, label='Calibrated Interval')
            ax.legend(loc='best')
            ax.set_title(f"Calibrated - {test_bearing_name} PI")
            ax.set_xlabel("TIME")
            ax.set_ylabel("RUL")
            plt.tight_layout()
            plt.savefig(calibrated_png_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ 校准后图片已保存到: {calibrated_png_path}")
        
        # 处理完所有测试集后返回
        return
    elif calibration_mode == 'train_first':
        # 使用训练集的第一个子集
        if len(train_bearings) == 0:
            logger.warning("⚠️  警告: 训练集为空，跳过校准")
            return
        selected_calibration_bearings = [train_bearings[0]]
        logger.info(f"使用训练集的第一个子集进行校准: {selected_calibration_bearings}")
    elif calibration_mode == 'train_all':
        # 使用所有训练集
        if len(train_bearings) == 0:
            logger.warning("⚠️  警告: 训练集为空，跳过校准")
            return
        selected_calibration_bearings = train_bearings
        logger.info(f"使用所有训练集进行校准: {selected_calibration_bearings}")
    elif calibration_mode == 'context':
        # 使用上下文数据集
        if context_bearings is None or len(context_bearings) == 0:
            logger.warning("⚠️  警告: 上下文数据集为空，跳过校准")
            return
        selected_calibration_bearings = context_bearings
        logger.info(f"使用上下文数据集进行校准: {selected_calibration_bearings}")
    elif calibration_mode == 'custom':
        # 使用自定义列表（需要提供calibration_bearings）
        if calibration_bearings is None or len(calibration_bearings) == 0:
            logger.warning("⚠️  警告: 自定义校准数据集为空，跳过校准")
            return
        # 如果启用同工况筛选，则只使用与测试集同工况的验证集
        if use_same_condition_validation:
            filtered_calibration = filter_validation_by_condition(calibration_bearings, test_bearings)
            if len(filtered_calibration) > 0:
                selected_calibration_bearings = filtered_calibration
                logger.info(f"使用与测试集同工况的验证集进行校准: {selected_calibration_bearings}")
                logger.info(f"测试集工况: {[condition_group(b) for b in test_bearings]}")
            else:
                logger.warning(f"⚠️  警告: 验证集中没有与测试集同工况的轴承，使用全部验证集: {calibration_bearings}")
                selected_calibration_bearings = calibration_bearings
        else:
            selected_calibration_bearings = calibration_bearings
            logger.info(f"使用自定义校准数据集: {selected_calibration_bearings}")
    elif calibration_bearings is not None and len(calibration_bearings) > 0:
        # 如果提供了calibration_bearings但没有指定模式，则使用自定义模式
        # 如果启用同工况筛选，则只使用与测试集同工况的验证集
        if use_same_condition_validation:
            filtered_calibration = filter_validation_by_condition(calibration_bearings, test_bearings)
            if len(filtered_calibration) > 0:
                selected_calibration_bearings = filtered_calibration
                logger.info(f"使用与测试集同工况的验证集进行校准: {selected_calibration_bearings}")
                logger.info(f"测试集工况: {[condition_group(b) for b in test_bearings]}")
            else:
                logger.warning(f"⚠️  警告: 验证集中没有与测试集同工况的轴承，使用全部验证集: {calibration_bearings}")
                selected_calibration_bearings = calibration_bearings
        else:
            selected_calibration_bearings = calibration_bearings
            logger.info(f"使用自定义校准数据集: {selected_calibration_bearings}")
    else:
        logger.warning(f"⚠️  警告: 未知的校准模式 '{calibration_mode}'，使用默认模式 'train_first'")
        if len(train_bearings) == 0:
            logger.warning("⚠️  警告: 训练集为空，跳过校准")
            return
        selected_calibration_bearings = [train_bearings[0]]
        logger.info(f"使用训练集的第一个子集进行校准: {selected_calibration_bearings}")
    
    print("selected_calibration_bearings", selected_calibration_bearings)
    # 1. 在校准数据集上进行预测
    # 如果使用测试集作为校准集，需要从测试数据目录加载数据
    use_test_data_dir = (calibration_mode == 'test')
    val_target, val_prediction, val_pred_std = predict_on_calibration_dataset(
        model=model,
        calibration_bearings=selected_calibration_bearings,
        train_data_dir=train_data_dir,
        scaler_dir=scaler_dir,
        test_bearings=test_bearings,
        test_datasets_type=test_datasets_type,
        config=config,
        device=device,
        load_bearing_data_func=load_bearing_data_func,
        test_data_dir=test_data_dir,
        use_test_data_dir=use_test_data_dir
    )
    
    if val_target is None or val_prediction is None or val_pred_std is None:
        logger.warning("⚠️  警告: 校准数据集预测失败，跳过校准")
        return
    
    # 2. 拟合等渗回归校准器
    calibrator = fit_calibrator(
        val_target=val_target,
        val_prediction=val_prediction,
        val_pred_std=val_pred_std,
        config=config
    )
    
    # 3. 对测试集进行校准并保存结果
    calibrate_test_results(
        calibrator=calibrator,
        first_test_results=first_test_results,
        config=config,
        res_dir=res_dir
    )

