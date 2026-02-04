"""
从 joblib 保存的 origin / pre / var 三组文件读取未校准结果，使用「温度缩放 + 保序回归」两步校准并保存 CSV。
结果保存到与本 .py 同路径下的 two_stage_calibration_results 文件夹；并计算 MAE、RMSE、PICP、NMPIW、CWC、ECE、Sharpness。
路径在下方直接配置，可直接运行: python run_two_stage_calibration_on_files.py
"""

import json
import numpy as np
import pandas as pd
from joblib import load
from scipy import stats
import os
import sys

# 保证可导入 two_stage_calibration 与 metrics
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from two_stage_calibration import TemperatureThenIsotonicCalibrator, evaluate_calibration
from metrics import mae, rmse, picp, nmpiw, cwc, ece, sharpness


def load_origin_pre_var(origin_path: str, pre_path: str, var_path: str):
    """从 joblib 文件加载 y_true, y_pred_mean, y_var（方差）。返回 y_true, y_pred_mean, y_pred_std。"""
    y_true = np.asarray(load(origin_path)).reshape(-1).astype(np.float64)
    y_pred_mean = np.asarray(load(pre_path)).reshape(-1).astype(np.float64)
    y_var = np.asarray(load(var_path)).reshape(-1).astype(np.float64)
    # 方差 -> 标准差，避免除零和负值
    y_pred_std = np.sqrt(np.maximum(y_var, 1e-12))
    assert len(y_true) == len(y_pred_mean) == len(y_pred_std), (
        "origin / pre / var 长度不一致"
    )
    return y_true, y_pred_mean, y_pred_std


def run_temperature_isotonic_calibration(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    alpha: float = 0.05,
    cal_frac: float = 0.5,
):
    """
    用部分数据拟合「温度缩放 + 保序回归」校准器，再对全部数据做校准。
    cal_frac: 用于拟合校准器的比例（前 cal_frac 作为校准集，全部数据用于最终校准输出）。
    """
    n = len(y_true)
    n_cal = max(1, int(n * cal_frac))
    y_true_cal, y_pred_mean_cal, y_pred_std_cal = (
        y_true[:n_cal],
        y_pred_mean[:n_cal],
        y_pred_std[:n_cal],
    )
    calibrator = TemperatureThenIsotonicCalibrator(alpha=alpha)
    calibrator.fit(y_true_cal, y_pred_mean_cal, y_pred_std_cal)
    y_lower, y_upper = calibrator.calibrate(y_pred_mean, y_pred_std)
    z = stats.norm.ppf(1 - alpha / 2)
    y_pred_std_calibrated = (y_upper - y_lower) / (2 * z + 1e-12)
    return y_lower, y_upper, y_pred_std_calibrated, calibrator


# ---------------------------------------------------------------------------
# 路径配置（直接修改此处）
# ---------------------------------------------------------------------------
_BASE = "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_xjtu"
ORIGIN_PATH = f"{_BASE}/c1_Bearing1_5_labeled_fpt_scaler_fbtcn_origin"
PRE_PATH = f"{_BASE}/c1_Bearing1_5_labeled_fpt_scaler_fbtcn_pre"
VAR_PATH = f"{_BASE}/c1_Bearing1_5_labeled_fpt_scaler_fbtcn_var"
OUT_BASE = "c1_Bearing1_5"  # 输出文件名前缀
# 结果保存到与本 .py 同路径下的 two_stage_calibration_results 文件夹
OUT_DIR = os.path.join(_script_dir, "two_stage_calibration_results")
ALPHA = 0.05
CAL_FRAC = 0.5  # 用于拟合校准器的数据比例 (0~1)


def compute_metrics(y_true, y_pred_mean, y_lower, y_upper, y_pred_std_calibrated, alpha=0.05):
    """计算 MAE、RMSE、PICP、NMPIW、CWC、ECE、Sharpness。"""
    R = float(np.max(y_true) - np.min(y_true))
    if R <= 0:
        R = 1.0
    picp_val = float(picp(y_true, y_lower, y_upper))
    nmpiw_val = float(nmpiw(y_lower, y_upper, R))
    var_cal = np.maximum(y_pred_std_calibrated ** 2, 1e-12)
    return {
        "MAE": float(mae(y_true, y_pred_mean)),
        "RMSE": float(rmse(y_true, y_pred_mean)),
        "PICP": picp_val,
        "NMPIW": nmpiw_val,
        "CWC": float(cwc(picp_val, nmpiw_val, alpha=alpha)),
        "ECE": float(ece(y_true, y_pred_mean, var_cal, n_bins=20)),
        "Sharpness": float(sharpness(y_pred_std_calibrated, alpha=alpha)),
    }


def main():
    y_true, y_pred_mean, y_pred_std = load_origin_pre_var(
        ORIGIN_PATH, PRE_PATH, VAR_PATH
    )

    # 校准前区间与指标
    z = stats.norm.ppf(1 - ALPHA / 2)
    y_lower_uncal = y_pred_mean - z * y_pred_std
    y_upper_uncal = y_pred_mean + z * y_pred_std
    met_uncal = compute_metrics(
        y_true, y_pred_mean, y_lower_uncal, y_upper_uncal, y_pred_std, alpha=ALPHA
    )

    y_lower, y_upper, y_pred_std_calibrated, _ = run_temperature_isotonic_calibration(
        y_true, y_pred_mean, y_pred_std,
        alpha=ALPHA,
        cal_frac=CAL_FRAC,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"{OUT_BASE}_temperature_isotonic_calibrated.csv")

    # 与现有 calibrated.csv 格式对齐；无 alea/epi 时填 0
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred_mean": y_pred_mean,
        "y_lower_calibrated": y_lower,
        "y_upper_calibrated": y_upper,
        "y_pred_std_total": y_pred_std,
        "y_pred_std_calibrated": y_pred_std_calibrated,
        "y_pred_alea": 0.0,
        "y_pred_epi": 0.0,
    })
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"已保存 CSV: {csv_path} (n={len(df)})")

    met = compute_metrics(
        y_true, y_pred_mean, y_lower, y_upper, y_pred_std_calibrated, alpha=ALPHA
    )
    met["target_coverage"] = 1.0 - ALPHA
    met["alpha"] = ALPHA

    json_path = os.path.join(OUT_DIR, f"{OUT_BASE}_temperature_isotonic_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(met, f, indent=2, ensure_ascii=False)
    print(f"已保存指标 JSON: {json_path}")

    txt_path = os.path.join(OUT_DIR, f"{OUT_BASE}_temperature_isotonic_metrics.txt")
    lines = [
        f"指标 (置信水平 {met['target_coverage']*100:.1f}%, alpha={ALPHA})",
        "=" * 50,
        f"MAE       : {met['MAE']:.6f}  (越小越好)",
        f"RMSE      : {met['RMSE']:.6f}  (越小越好)",
        f"PICP      : {met['PICP']:.6f}  (越接近目标覆盖率越好)",
        f"NMPIW     : {met['NMPIW']:.6f}  (越小越好)",
        f"CWC       : {met['CWC']:.6f}  (越小越好)",
        f"ECE       : {met['ECE']:.6f}  (越小越好)",
        f"Sharpness : {met['Sharpness']:.6f}  (越小越好)",
        "=" * 50,
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已保存指标 TXT: {txt_path}")

    print("\n校准前 指标 (PICP, NMPIW, CWC, ECE, Sharpness):")
    for k in ("PICP", "NMPIW", "CWC", "ECE", "Sharpness"):
        print(f"  {k}: {met_uncal[k]:.6f}")

    print("\n校准后 指标摘要:")
    for k in ("MAE", "RMSE", "PICP", "NMPIW", "CWC", "ECE", "Sharpness"):
        print(f"  {k}: {met[k]:.6f}")


if __name__ == "__main__":
    main()
