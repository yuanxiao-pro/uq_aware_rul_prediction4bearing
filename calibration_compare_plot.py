import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import norm
import joblib
import os

# 配置字体：中文为宋体，西文为 Times New Roman
def setup_fonts():
    """设置 matplotlib 字体：中文宋体，西文 Times New Roman"""
    # 清除 matplotlib 字体缓存（确保加载最新字体）
    try:
        cache_dir = matplotlib.get_cachedir()
        for f in os.listdir(cache_dir):
            if 'font' in f.lower() and f.endswith('.json'):
                try:
                    os.remove(os.path.join(cache_dir, f))
                except:
                    pass
    except:
        pass
    
    # 添加字体文件（如果存在）
    simsun_path = os.path.expanduser("~/.local/share/fonts/simsun.ttc")
    times_path = os.path.expanduser("~/.local/share/fonts/times.ttf")
    noto_serif_path = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
    noto_sans_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    
    font_paths = [
        (simsun_path, "SimSun"),
        (times_path, "Times New Roman"),
        (noto_serif_path, "Noto Serif CJK SC"),
        (noto_sans_path, "Noto Sans CJK SC"),
    ]
    
    for font_path, font_name in font_paths:
        if os.path.exists(font_path):
            try:
                fm.fontManager.addfont(font_path)
            except:
                pass
    
    # 重新初始化字体管理器以识别新添加的字体
    try:
        fm.fontManager.__init__()
    except:
        pass
    
    # 查找可用的字体名称
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 设置中文字体
    # 优先使用 Noto Serif CJK（TTC 文件包含所有 CJK 变体，包括 SC，可以正确显示简体中文）
    chinese_font = None
    
    # 优先查找 Noto Serif CJK（即使显示为 JP，TTC 文件包含所有字符）
    if os.path.exists(noto_serif_path):
        noto_fonts = [f for f in available_fonts if 'Noto' in f and 'Serif' in f and 'CJK' in f]
        if noto_fonts:
            chinese_font = noto_fonts[0]  # 使用第一个（TTC 文件包含所有 CJK 变体）
        else:
            # 如果不在列表中，尝试直接使用 SC 名称
            chinese_font = 'Noto Serif CJK SC'
    
    # 如果没有 Noto，尝试使用 SimSun
    if chinese_font is None:
        for font_name in ['SimSun', 'NSimSun', 'SimSun-ExtB']:
            if font_name in available_fonts:
                chinese_font = font_name
                break
    
    # 设置西文字体（Times New Roman）
    western_font = None
    if 'Times New Roman' in available_fonts:
        western_font = 'Times New Roman'
    elif 'Times' in available_fonts:
        western_font = 'Times'
    
    # 配置 matplotlib
    if chinese_font:
        # 将中文字体放在最前面
        current_fonts = [f for f in plt.rcParams['font.sans-serif'] if f != chinese_font]
        plt.rcParams['font.sans-serif'] = [chinese_font] + current_fonts
        
        # 如果使用 Noto Serif CJK，确保使用正确的字体文件
        # TTC 文件包含所有 CJK 变体，matplotlib 会自动选择正确的字符
        if 'Noto' in chinese_font and os.path.exists(noto_serif_path):
            # 确保字体文件已加载
            try:
                # 通过设置字体属性来确保使用正确的字体文件
                from matplotlib.font_manager import FontProperties
                # 创建一个全局的字体属性对象（可选）
                pass
            except:
                pass
    else:
        # 如果都没找到，尝试直接使用 Noto Serif CJK（即使名称不在列表中）
        if os.path.exists(noto_serif_path):
            chinese_font = 'Noto Serif CJK SC'  # 直接使用，让 matplotlib 尝试匹配
            current_fonts = [f for f in plt.rcParams['font.sans-serif'] if f != chinese_font]
            plt.rcParams['font.sans-serif'] = [chinese_font] + current_fonts
    
    if western_font:
        current_serif = [f for f in plt.rcParams['font.serif'] if f != western_font]
        plt.rcParams['font.serif'] = [western_font] + current_serif
        plt.rcParams['mathtext.fontset'] = 'stix'
    
    plt.rcParams['axes.unicode_minus'] = False

# 初始化字体配置
setup_fonts()


def compute_calibration_curve(y_true: np.ndarray,
                              mu: np.ndarray,
                              var: np.ndarray,
                              n_bins: int = 20,
                              eps: float = 1e-12):
    """
    构造回归可靠性曲线（Gaussian 假设）。
    返回分位数列表、经验覆盖率和 ECE。
    """
    sigma = np.sqrt(np.maximum(var, eps))
    qs = np.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins)
    empirical = np.empty_like(qs)
    for i, q in enumerate(qs):
        z = norm.ppf(q)
        thresh = mu + sigma * z
        empirical[i] = np.mean(y_true <= thresh)
    ece = float(np.mean(np.abs(empirical - qs)))
    return qs, empirical, ece


def load_pre_from_joblib(origin_path: str, pre_path: str, var_path: str):
    """
    读取校准前数据（joblib 保存的 numpy 数组）：
    - origin_path: y_true
    - pre_path: 预测均值
    - var_path: 预测方差
    """
    y_true = joblib.load(origin_path)
    mu = joblib.load(pre_path)
    var = joblib.load(var_path)
    return np.asarray(y_true), np.asarray(mu), np.asarray(var)


def load_pre_calibration(path: str):
    """
    读取校准前 CSV（保留兼容性）：
    - y_true 列名为 y_true
    - 预测样本列名形如 y_pred_sample_*
    - 方差列名 y_var
    """
    df = pd.read_csv(path)
    if "y_true" not in df.columns:
        raise ValueError("pre csv 缺少列 y_true")
    y_true = df["y_true"].to_numpy()

    sample_cols = [c for c in df.columns if c.startswith("y_pred_sample_")]
    if len(sample_cols) == 0:
        raise ValueError("pre csv 未找到 y_pred_sample_* 列")
    mu = df[sample_cols].to_numpy().mean(axis=1)

    if "y_var" in df.columns:
        var = df["y_var"].to_numpy()
    else:
        var = df[sample_cols].to_numpy().var(axis=1)
    return y_true, mu, var


def load_post_calibration(path: str):
    """
    读取校准后文件：
    - y_true 列名 y_true
    - 预测均值列名 y_pred_mean
    - 预测方差优先使用 y_pred_std_calibrated^2，其次 y_pred_alea + y_pred_epi
    """
    df = pd.read_csv(path)
    for col in ["y_true", "y_pred_mean"]:
        if col not in df.columns:
            raise ValueError(f"post csv 缺少列 {col}")
    y_true = df["y_true"].to_numpy()
    mu = df["y_pred_mean"].to_numpy()

    if "y_pred_std_calibrated" in df.columns:
        var = np.square(df["y_pred_std_calibrated"].to_numpy())
    elif "y_pred_alea" in df.columns and "y_pred_epi" in df.columns:
        var = df["y_pred_alea"].to_numpy() + df["y_pred_epi"].to_numpy()
    else:
        raise ValueError("post csv 未找到预测方差信息 (y_pred_std_calibrated 或 y_pred_alea + y_pred_epi)")
    
    return y_true, mu, var


def main():
    """
    批量绘制校准前后可靠性曲线。
    将多个输入路径放入 entries 数组，每个元素包含 origin/pre/var/post 的路径和可选标题。
    所有子图输出到同一个 svg 文件。
    """
    entries = [
        # 可按需追加更多条目，例如：
        # {
        #     "origin": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_xjtu/c1_Bearing1_2_labeled_fpt_scaler_fbtcn_origin",
        #     "pre": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_xjtu/c1_Bearing1_2_labeled_fpt_scaler_fbtcn_origin",
        #     "var": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_xjtu/c1_Bearing1_2_labeled_fpt_scaler_fbtcn_origin",
        #     "post": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_xjtu/c1_Bearing1_2_calibrated.csv",
        #     "title": "校准前后可靠性曲线对比",
        #     "name": "c1_Bearing1_2",
        # },
        {
            "origin": "/mnt/uq_aware_rul_prediction4bearing-main/auto_myexp_result/xjtu_to_xjtu/c2_Bearing2_2_labeled_fpt_scaler_fbtcn_origin",
            "pre": "/mnt/uq_aware_rul_prediction4bearing-main/auto_myexp_result/xjtu_to_xjtu/c2_Bearing2_2_labeled_fpt_scaler_fbtcn_pre",
            "var": "/mnt/uq_aware_rul_prediction4bearing-main/auto_myexp_result/xjtu_to_xjtu/c2_Bearing2_2_labeled_fpt_scaler_fbtcn_var",
            "middle_origin": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir_no_rds/xjtu_to_xjtu/c2_Bearing2_2_labeled_fpt_scaler_fbtcn_origin",
            "middle_pre": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir_no_rds/xjtu_to_xjtu/c2_Bearing2_2_labeled_fpt_scaler_fbtcn_pre",
            "middle_var": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir_no_rds/xjtu_to_xjtu/c2_Bearing2_2_labeled_fpt_scaler_fbtcn_var",
            "post": "/mnt/uq_aware_rul_prediction4bearing-main/auto_myexp_result/xjtu_to_xjtu/c2_Bearing2_2_calibrated.csv",
            "title": "X2-Bearing2_2",
            "name": "c2_Bearing2_2",
        },
        {
            "origin": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_xjtu/c3_Bearing3_5_labeled_fpt_scaler_fbtcn_origin",
            "pre": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_xjtu/c3_Bearing3_5_labeled_fpt_scaler_fbtcn_pre",
            "var": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_xjtu/c3_Bearing3_5_labeled_fpt_scaler_fbtcn_var",
            "middle_origin": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir/xjtu_to_xjtu/c3_Bearing3_5_labeled_fpt_scaler_fbtcn_origin",
            "middle_pre": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir/xjtu_to_xjtu/c3_Bearing3_5_labeled_fpt_scaler_fbtcn_pre",
            "middle_var": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir/xjtu_to_xjtu/c3_Bearing3_5_labeled_fpt_scaler_fbtcn_var",
            "post": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_xjtu/c3_Bearing3_5_calibrated.csv",
            "title": "X3-Bearing3_5",
            "name": "c3_Bearing3_5",
        },
        {
            "origin": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_femto/Bearing1_3_labeled_fpt_scaler_fbtcn_origin",
            "pre": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_femto/Bearing1_3_labeled_fpt_scaler_fbtcn_pre",
            "var": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_femto/Bearing1_3_labeled_fpt_scaler_fbtcn_var",
            "middle_origin": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir_no_rds/xjtu_to_femto/Bearing1_3_labeled_fpt_scaler_fbtcn_origin",
            "middle_pre": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir_no_rds/xjtu_to_femto/Bearing1_3_labeled_fpt_scaler_fbtcn_pre",
            "middle_var": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/E_no_crps_ir_no_rds/xjtu_to_femto/Bearing1_3_labeled_fpt_scaler_fbtcn_var",
            "post": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_femto/Bearing1_3_calibrated.csv",
            "title": "F1-Bearing1_3",
            "name": "Bearing1_3",
        },

        # {
        #     "origin": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_xjtu/c1_Bearing1_3_labeled_fpt_scaler_fbtcn_origin",
        #     "pre": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_xjtu/c1_Bearing1_3_labeled_fpt_scaler_fbtcn_pre",
        #     "var": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_xjtu/c1_Bearing1_3_labeled_fpt_scaler_fbtcn_var",
        #     "post": "/mnt/uq_aware_rul_prediction4bearing-main/auto_ablation_result/A_no_rds/xjtu_to_xjtu/c1_Bearing1_3_calibrated.csv",
        #     "title": "Bearing1_3可靠性图",
        #     "name": "c1_Bearing1_3",
        # },
    ]

    n_bins = 20
    out_path = "framework_calib_compare_batch.svg"

    # 创建字体属性对象，确保使用正确的字体
    from matplotlib.font_manager import FontProperties

    # 中文字体（必须使用宋体）
    # 优先使用标准 SimSun（不是 ExtB）
    simsun_standard_path = os.path.expanduser("~/.local/share/fonts/simsun_standard.ttc")
    simsun_path = os.path.expanduser("~/.local/share/fonts/simsun.ttc")
    noto_serif_path = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"

    chinese_font_prop = None
    font_warning = None

    # 优先使用标准 SimSun
    if os.path.exists(simsun_standard_path):
        chinese_font_prop = FontProperties(fname=simsun_standard_path)
        print("使用标准 SimSun 字体")
    elif os.path.exists(simsun_path):
        try:
            from fontTools.ttLib import TTFont
            font = TTFont(simsun_path)
            is_extb = False
            if 'name' in font:
                name_table = font['name']
                for record in name_table.names:
                    if record.nameID == 4:
                        try:
                            name_str = record.toUnicode()
                            if 'ExtB' in name_str:
                                is_extb = True
                                break
                        except:
                            pass
            if is_extb:
                font_warning = "警告: 当前 SimSun-ExtB 缺少常用中文字符，可能显示为方框。请安装标准 SimSun 字体。"
                print(font_warning)
            chinese_font_prop = FontProperties(fname=simsun_path)
        except:
            chinese_font_prop = FontProperties(fname=simsun_path)
            font_warning = "警告: 无法验证 SimSun 字体版本，如果中文显示为方框，请安装标准 SimSun 字体。"
            print(font_warning)
    else:
        raise ValueError("未找到 SimSun 字体。请安装标准 SimSun 字体（不是 ExtB 版本）。\n"
                        "可以从 Windows 系统复制: C:\\Windows\\Fonts\\simsun.ttc\n"
                        "或运行: python install_standard_simsun.py")
    # 西文字体（Times New Roman）
    times_path = os.path.expanduser("~/.local/share/fonts/times.ttf")
    western_font_prop = None
    if os.path.exists(times_path):
        western_font_prop = FontProperties(fname=times_path)

    # # 全局字体尺寸放大
    # import matplotlib as mpl
    # mpl.rcParams.update({
    #     "axes.titlesize": 26,
    #     "axes.labelsize": 22,
    #     "legend.fontsize": 22,
    #     "xtick.labelsize": 20,
    #     "ytick.labelsize": 20,
    # })

    # 子图布局：1 行 n 列
    n = len(entries)
    ncols = n if n > 0 else 1
    nrows = 1
    # 增大高度，避免标题和图例被挤压
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, item in enumerate(entries):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        # 读取数据（校准前、中间均为 joblib 三组文件；校准后为 CSV）
        y_true_pre, mu_pre, var_pre = load_pre_from_joblib(item["origin"], item["pre"], item["var"])
        y_true_mid, mu_mid, var_mid = load_pre_from_joblib(item["middle_origin"], item["middle_pre"], item["middle_var"])
        y_true_post, mu_post, var_post = load_post_calibration(item["post"])

        # 计算校准曲线（三组）
        qs_pre, empirical_pre, ece_pre = compute_calibration_curve(
            y_true=y_true_pre, mu=mu_pre, var=var_pre, n_bins=n_bins
        )
        qs_mid, empirical_mid, ece_mid = compute_calibration_curve(
            y_true=y_true_mid, mu=mu_mid, var=var_mid, n_bins=n_bins
        )
        qs_post, empirical_post, ece_post = compute_calibration_curve(
            y_true=y_true_post, mu=mu_post, var=var_post, n_bins=n_bins
        )

        # 计算 sharpness（方差的平均值）
        sharp_pre = float(np.mean(np.maximum(var_pre, 1e-12)))
        sharp_mid = float(np.mean(np.maximum(var_mid, 1e-12)))
        sharp_post = float(np.mean(np.maximum(var_post, 1e-12)))

        # 绘制曲线并保存颜色（校准前、中间、校准后）
        line2 = ax.plot(qs_mid, empirical_mid, marker="^", label=f"w/o CRPS&IR (ECE={ece_mid:.3f})", linewidth=1.5, markersize=3)
        color_mid = line2[0].get_color()

        line1 = ax.plot(qs_pre, empirical_pre, marker="o", label=f"w/o IR (ECE={ece_pre:.3f})", linewidth=1.5, markersize=4)
        color_pre = line1[0].get_color()

        line3 = ax.plot(qs_post, empirical_post, marker="s", label=f"所提方法 (ECE={ece_post:.3f})", linewidth=1.5, markersize=3)
        color_post = line3[0].get_color()

        # 为校准前曲线添加起点和终点标记，并将其与理想曲线的起止点相连
        ax.plot(0, 0, marker="o", markersize=6, color=color_pre,
                markeredgecolor='white', markeredgewidth=0.5, zorder=5, label='_nolegend_')
        ax.plot(1, 1, marker="o", markersize=6, color=color_pre,
                markeredgecolor='white', markeredgewidth=0.5, zorder=5, label='_nolegend_')
        ax.plot([qs_pre[0], 0], [empirical_pre[0], 0],
                color=color_pre, linewidth=1.2, linestyle='-', alpha=1, zorder=3, label='_nolegend_')
        ax.plot([qs_pre[-1], 1], [empirical_pre[-1], 1],
                color=color_pre, linewidth=1.2, linestyle='-', alpha=1, zorder=3, label='_nolegend_')

        # 为中间结果曲线添加起点和终点标记，并将其与理想曲线的起止点相连
        ax.plot(0, 0, marker="^", markersize=5, color=color_mid,
                markeredgecolor='white', markeredgewidth=0.1, zorder=5, label='_nolegend_')
        ax.plot(1, 1, marker="^", markersize=5, color=color_mid,
                markeredgecolor='white', markeredgewidth=0.1, zorder=5, label='_nolegend_')
        ax.plot([qs_mid[0], 0], [empirical_mid[0], 0],
                color=color_mid, linewidth=1.2, linestyle='-', alpha=1, zorder=3, label='_nolegend_')
        ax.plot([qs_mid[-1], 1], [empirical_mid[-1], 1],
                color=color_mid, linewidth=1.2, linestyle='-', alpha=1, zorder=3, label='_nolegend_')

        # 为校准后曲线添加起点和终点标记，并将其与理想曲线的起止点相连
        ax.plot(0, 0, marker="s", markersize=4, color=color_post,
                markeredgecolor='white', markeredgewidth=0.1, zorder=5, label='_nolegend_')
        ax.plot(1, 1, marker="s", markersize=4, color=color_post,
                markeredgecolor='white', markeredgewidth=0.1, zorder=5, label='_nolegend_')
        ax.plot([qs_post[0], 0], [empirical_post[0], 0],
                color=color_post, linewidth=1.2, linestyle='-', alpha=1, zorder=3, label='_nolegend_')
        ax.plot([qs_post[-1], 1], [empirical_post[-1], 1],
                color=color_post, linewidth=1.2, linestyle='-', alpha=1, zorder=3, label='_nolegend_')

        # 理想曲线：黑色虚线，不加端点标记，最后绘制以确保显示在最上层
        ax.plot([0, 1], [0, 1], color='black', linestyle='--', label="理想曲线", linewidth=1.5, zorder=4)

        # 使用字体属性对象设置标签和标题，标题加序号 (a),(b)...
        raw_title = item.get("title", "校准前后可靠性曲线对比")
        prefix = f"({chr(ord('a') + idx)}) "
        title = prefix + raw_title
        if chinese_font_prop:
            label_fp = chinese_font_prop.copy()
            label_fp.set_size(18)

            title_fp = chinese_font_prop.copy()
            title_fp.set_size(22)
            title_fp.set_weight("bold")

            legend_fp = chinese_font_prop.copy()
            legend_fp.set_size(16)

            ax.set_xlabel("名义分位数", fontproperties=label_fp)
            ax.set_ylabel("经验累积分布函数", fontproperties=label_fp)
            ax.set_title(title, fontproperties=title_fp)
            legend = ax.legend(prop=legend_fp)
        else:
            ax.set_xlabel("名义分位数", fontsize=18)
            ax.set_ylabel("经验累积分布函数", fontsize=18)
            ax.set_title(title, fontsize=30, fontweight='bold')
            ax.legend(fontsize=16)

        # 调大坐标轴刻度字号
        ax.tick_params(axis='both', which='major', labelsize=16)

        # 控制台打印（三组）
        name = item.get("name", f"entry_{idx}")
        print(f"[{name}] ECE 校准前   : {ece_pre:.5f}")
        print(f"[{name}] ECE 中间结果 : {ece_mid:.5f}")
        print(f"[{name}] ECE 校准后   : {ece_post:.5f}")
        print(f"[{name}] Sharpness 校准前   : {sharp_pre:.5f}")
        print(f"[{name}] Sharpness 中间结果 : {sharp_mid:.5f}")
        print(f"[{name}] Sharpness 校准后   : {sharp_post:.5f}")

        ax.grid(True, alpha=0.3)

    # 隐藏多余子图（当条目数小于 ncols 时）
    for j in range(len(entries), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    # 预留顶部空间给标题/图例
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=300, bbox_inches="tight", format='svg')
    print(f"Calibration compare plot saved to: {out_path}")
    print(f"Total figures: {len(entries)}")
if __name__ == "__main__":
    main()

