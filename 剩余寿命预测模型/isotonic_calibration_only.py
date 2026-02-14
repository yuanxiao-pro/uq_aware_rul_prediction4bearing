"""
仅使用保序回归的预测区间校准 (Isotonic Regression Only)

- 校准集规则：
  - XJTU 工况1：使用工况1的 Bearing1_4 作为校准集
  - XJTU 工况2：使用工况1的 Bearing2_3 作为校准集
  - XJTU 工况3：使用工况1的 Bearing3_2 作为校准集
  - FEMTO 工况1：使用工况1的 Bearing1_6 作为校准集
- 输入数据：/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds 下的 *_result.csv
- 结果仅打印到控制台，不写入磁盘。
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.stats import norm
from prediction_interval_calibration import (
    IsotonicRegressionCalibrator,
    evaluate_calibration,
)
from metrics import cwc, ece, sharpness

# 为 True 时，在 ECE 校准时变差的轴承上打印「分位 q vs 经验覆盖率 empirical」便于分析
DEBUG_QUANTILE_CALIBRATION = False
QUANTILE_N_BINS = 10

BASE_DIR = "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds"
# 校准结果与可靠性图输出目录（新建文件夹）
PROJECT_ROOT = "/mnt/uq_aware_rul_prediction4bearing-main"
CALIBRATION_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "calibration_reliability_output")
CALIBRATED_CSV_DIR = os.path.join(CALIBRATION_OUTPUT_DIR, "ir_calibrated")
FIGURES_DIR = os.path.join(CALIBRATION_OUTPUT_DIR, "figures")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
ALPHA = 0.05
CONFIDENCE_LEVEL = 1 - ALPHA

# 校准集：每个（子目录, 条件）对应一个校准用 result 文件名（不含路径、不含后缀）
# XJTU 工况1 -> 工况1的 Bearing1_4；工况2 -> 工况1的 Bearing2_3；工况3 -> 工况1的 Bearing3_2
# FEMTO 工况1 -> 工况1的 Bearing1_6
CALIBRATION_MAP = {
    "xjtu_to_xjtu": {
        "c1": "c1_Bearing1_2_labeled_fpt_scaler_fbtcn_result",
        "c2": "c2_Bearing2_3_labeled_fpt_scaler_fbtcn_result",
        "c3": "c3_Bearing3_4_labeled_fpt_scaler_fbtcn_result",
    },
    "xjtu_to_femto": {
        "c1": "Bearing1_6_labeled_fpt_scaler_fbtcn_result",
    },
}


def _load_result_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 *_result.csv 加载并计算 y_true, y_pred_mean, y_pred_std。"""
    df = pd.read_csv(path)
    y_true = df["y_true"].values
    sample_cols = [c for c in df.columns if c.startswith("y_pred_sample_")]
    samples = df[sample_cols].values  # (n, n_samples)
    y_pred_mean = np.mean(samples, axis=1)
    epi_var = np.var(samples, axis=1)
    alea_var = df["y_var"].values
    y_pred_std = np.sqrt(epi_var + alea_var)
    return y_true, y_pred_mean, y_pred_std


def _result_csv_stem(fname: str) -> str:
    """去掉 _result.csv 后缀得到 stem。"""
    if fname.endswith("_result.csv"):
        return fname[:-len("_result.csv")]
    return fname


def _quantile_empirical_curve(
    y_true: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    与 metrics.ece 一致的分位校准曲线：对每个分位 q 计算经验比例 empirical = mean(y_true <= mu + sigma*z_q)。
    返回 (qs, empirical)，用于调试 IR 后哪些 q 变差。
    """
    y_true = np.asarray(y_true).reshape(-1)
    mu = np.asarray(mu).reshape(-1)
    var = np.asarray(var).reshape(-1)
    sigma = np.sqrt(np.maximum(var, eps))
    qs = np.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins)
    empirical = np.empty_like(qs)
    for j, q in enumerate(qs):
        z = norm.ppf(q)
        thresh = mu + sigma * z
        empirical[j] = np.mean(y_true <= thresh)
    return qs, empirical


def _print_quantile_debug(
    name: str,
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    var_raw: np.ndarray,
    var_cal: np.ndarray,
    n_bins: int = 10,
) -> None:
    """按分位打印 empirical vs q（未校准 vs 校准），标出 IR 后变差的分位。"""
    qs, emp_raw = _quantile_empirical_curve(y_true, y_pred_mean, var_raw, n_bins=n_bins)
    _, emp_cal = _quantile_empirical_curve(y_true, y_pred_mean, var_cal, n_bins=n_bins)
    err_raw = np.abs(emp_raw - qs)
    err_cal = np.abs(emp_cal - qs)
    worse = err_cal > err_raw
    print(f"    [分位调试] {name} (n_bins={n_bins}, 理想 empirical=q)")
    print(f"    {'q':>6} | {'emp_raw':>8} | {'emp_cal':>8} | {'|e_raw-q|':>10} | {'|e_cal-q|':>10} | IR后变差")
    print("    " + "-" * 62)
    for i in range(len(qs)):
        w = "  *" if worse[i] else ""
        print(f"    {qs[i]:.3f} | {emp_raw[i]:.4f}   | {emp_cal[i]:.4f}   | {err_raw[i]:.4f}     | {err_cal[i]:.4f}     |{w}")
    n_worse = np.sum(worse)
    print(f"    -> IR 后变差的分位数: {n_worse}/{n_bins}")
    print()


def run_isotonic_calibration():
    """按校准集规则拟合保序回归，并在各轴承子集上校准，结果打印到控制台。"""
    from scipy import stats

    z_score = stats.norm.ppf(1 - ALPHA / 2)

    for subdir, cond_cal in CALIBRATION_MAP.items():
        subdir_path = os.path.join(BASE_DIR, subdir)
        if not os.path.isdir(subdir_path):
            print(f"[跳过] 目录不存在: {subdir_path}")
            continue

        all_results = [
            f for f in os.listdir(subdir_path)
            if f.endswith("_result.csv") and "calibrated" not in f.lower()
        ]

        for condition, cal_stem in cond_cal.items():
            cal_fname = cal_stem + ".csv" if not cal_stem.endswith(".csv") else cal_stem
            cal_path = os.path.join(subdir_path, cal_fname)

            if not os.path.isfile(cal_path):
                print(f"[跳过] 校准集文件不存在: {cal_path}")
                continue

            # 校准集
            y_true_cal, y_pred_mean_cal, y_pred_std_cal = _load_result_csv(cal_path)
            calibrator = IsotonicRegressionCalibrator(alpha=ALPHA)
            calibrator.fit(y_true_cal, y_pred_mean_cal, y_pred_std_cal)

            # 该条件对应的 result 文件：xjtu_to_xjtu 下为 cX_*；xjtu_to_femto 仅处理工况1（Bearing1_*）
            if subdir == "xjtu_to_xjtu":
                prefix = condition  # c1, c2, c3
                target_files = [f for f in all_results if f.startswith(prefix + "_")]
            else:
                # xjtu_to_femto 只校准工况1，工况2/3不做处理
                target_files = [f for f in all_results if f.startswith("Bearing1_")]

            print()
            print("=" * 70)
            print(f"保序回归校准 | {subdir} | 条件 {condition} | 校准集: {cal_stem}")
            print("=" * 70)

            for fname in sorted(target_files):
                path = os.path.join(subdir_path, fname)
                y_true, y_pred_mean, y_pred_std = _load_result_csv(path)
                name = _result_csv_stem(fname)

                # 未校准区间
                y_lower_raw = y_pred_mean - z_score * y_pred_std
                y_upper_raw = y_pred_mean + z_score * y_pred_std
                m_raw = evaluate_calibration(y_true, y_lower_raw, y_upper_raw, CONFIDENCE_LEVEL)

                # 校准后区间
                y_lower_cal, y_upper_cal = calibrator.calibrate(y_pred_mean, y_pred_std)
                m_cal = evaluate_calibration(y_true, y_lower_cal, y_upper_cal, CONFIDENCE_LEVEL)

                # 校准后的不确定性（用于 ECE、Sharpness）
                calibrated_std = (y_upper_cal - y_lower_cal) / (2 * z_score + 1e-8)
                var_raw = y_pred_std ** 2
                var_cal = calibrated_std ** 2

                cwc_raw = cwc(m_raw["PICP"], m_raw["NMPIW"], alpha=ALPHA)
                cwc_cal = cwc(m_cal["PICP"], m_cal["NMPIW"], alpha=ALPHA)
                ece_raw = ece(y_true, y_pred_mean, var_raw)
                ece_cal = ece(y_true, y_pred_mean, var_cal)
                sharp_raw = sharpness(y_pred_std, alpha=ALPHA)
                sharp_cal = sharpness(calibrated_std, alpha=ALPHA)

                print(f"\n--- {name} ---")
                print(f"  样本数: {len(y_true)}")
                print(f"  未校准  PICP: {m_raw['PICP']:.6f}  覆盖误差: {m_raw['Coverage_Error']:.6f}  NMPIW: {m_raw['NMPIW']:.6f}  MPIW: {m_raw['MPIW']:.6f}")
                print(f"           CWC: {cwc_raw:.6f}  ECE: {ece_raw:.6f}  Sharpness: {sharp_raw:.6f}")
                print(f"  校准后  PICP: {m_cal['PICP']:.6f}  覆盖误差: {m_cal['Coverage_Error']:.6f}  NMPIW: {m_cal['NMPIW']:.6f}  MPIW: {m_cal['MPIW']:.6f}")
                print(f"           CWC: {cwc_cal:.6f}  ECE: {ece_cal:.6f}  Sharpness: {sharp_cal:.6f}")

                # ECE 校准时变差则打印分位 empirical vs q，便于分析哪些 q 被 IR 推偏
                if DEBUG_QUANTILE_CALIBRATION and ece_cal > ece_raw:
                    _print_quantile_debug(
                        name, y_true, y_pred_mean, var_raw, var_cal, n_bins=QUANTILE_N_BINS
                    )

                # 保存校准后 CSV（供 calibration_compare_plot 画可靠性图）
                out_subdir = os.path.join(CALIBRATED_CSV_DIR, subdir)
                os.makedirs(out_subdir, exist_ok=True)
                cal_csv_path = os.path.join(out_subdir, f"{name}_calibrated.csv")
                pd.DataFrame({
                    "y_true": y_true,
                    "y_pred_mean": y_pred_mean,
                    "y_lower_calibrated": y_lower_cal,
                    "y_upper_calibrated": y_upper_cal,
                    "y_pred_std_calibrated": calibrated_std,
                }).to_csv(cal_csv_path, index=False, encoding="utf-8-sig")

            print()

    # 调用 calibration_compare_plot 绘制校准前后可靠性图，保存到新文件夹
    _run_reliability_plots()
    print("保序回归校准（仅控制台输出）完成。")


def _run_reliability_plots() -> None:
    """调用 calibration_compare_plot 绘制校准前后可靠性图，保存到 FIGURES_DIR。"""
    try:
        from calibration_compare_plot import main_before_after
        main_before_after(
            result_base_dir=BASE_DIR,
            calibrated_dir=CALIBRATED_CSV_DIR,
            out_fig_dir=FIGURES_DIR,
            n_bins=20,
        )
    except Exception as e:
        print(f"[警告] 可靠性图绘制失败: {e}")


def fit_isotonic_calibrator(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    alpha: float = 0.05,
) -> IsotonicRegressionCalibrator:
    """在校准集上拟合保序回归校准器。"""
    calibrator = IsotonicRegressionCalibrator(alpha=alpha)
    calibrator.fit(y_true, y_pred_mean, y_pred_std)
    return calibrator


def calibrate_predictions(
    calibrator: IsotonicRegressionCalibrator,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """使用已拟合的保序回归校准器校准预测区间。"""
    return calibrator.calibrate(y_pred_mean, y_pred_std)


if __name__ == "__main__":
    run_isotonic_calibration()
