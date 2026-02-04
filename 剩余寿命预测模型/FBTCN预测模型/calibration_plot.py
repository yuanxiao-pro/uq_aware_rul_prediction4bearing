import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def compute_calibration_curve(y_true: np.ndarray,
                              mu: np.ndarray,
                              std: np.ndarray,
                              n_bins: int = 20,
                              eps: float = 1e-12):
    """
    构造回归可靠性曲线（Gaussian 假设）。
    返回分位数列表、经验覆盖率和 ECE。
    """
    var = np.maximum(std ** 2, eps)
    sigma = np.sqrt(var)

    qs = np.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins)
    empirical = np.empty_like(qs)
    for i, q in enumerate(qs):
        z = norm.ppf(q)
        thresh = mu + sigma * z
        empirical[i] = np.mean(y_true <= thresh)
    ece = float(np.mean(np.abs(empirical - qs)))
    return qs, empirical, ece


def main():
    parser = argparse.ArgumentParser(description="绘制回归校准曲线 (calibration plot)")
    parser.add_argument("--csv", required=True, help="包含 y_true / y_pred_mean / std(or var) 的 CSV 路径")
    parser.add_argument("--y_true_col", default="y_true", help="真实值列名")
    parser.add_argument("--mu_col", default="y_pred_mean", help="预测均值列名")
    parser.add_argument("--std_col", default="y_pred_std_total", help="预测标准差列名（默认用未校准/总不确定性）")
    parser.add_argument("--var_col", default=None, help="可选：预测方差列名（若提供则优先生效）")
    parser.add_argument("--n_bins", type=int, default=20, help="分箱数量（默认20）")
    parser.add_argument("--out", default="calibration_plot.png", help="输出图片路径")
    parser.add_argument("--title", default="Calibration Plot", help="图标题")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if args.var_col and args.var_col in df.columns:
        std = np.sqrt(np.maximum(df[args.var_col].to_numpy(), 0.0))
    else:
        if args.std_col not in df.columns:
            raise ValueError(f"找不到标准差列: {args.std_col}")
        std = df[args.std_col].to_numpy()

    if args.y_true_col not in df.columns:
        raise ValueError(f"找不到真实值列: {args.y_true_col}")
    if args.mu_col not in df.columns:
        raise ValueError(f"找不到均值列: {args.mu_col}")

    y_true = df[args.y_true_col].to_numpy()
    mu = df[args.mu_col].to_numpy()

    qs, empirical, ece = compute_calibration_curve(
        y_true=y_true,
        mu=mu,
        std=std,
        n_bins=args.n_bins
    )

    plt.figure(figsize=(6, 6))
    plt.plot(qs, empirical, marker="o", label="Empirical")
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.xlabel("Nominal quantile")
    plt.ylabel("Empirical CDF")
    plt.title(f"{args.title}\nECE={ece:.4f}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Calibration plot saved to: {args.out}")
    print(f"ECE: {ece:.6f}")


if __name__ == "__main__":
    main()

