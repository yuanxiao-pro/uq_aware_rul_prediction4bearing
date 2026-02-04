"""
Matplotlib中文显示配置工具
Matplotlib Chinese Font Configuration Utility

使用方法：
    from matplotlib_chinese_config import setup_chinese_font
    setup_chinese_font()
    
或者在导入matplotlib后直接调用：
    import matplotlib_chinese_config  # 会自动配置
"""

import matplotlib.pyplot as plt
import matplotlib

def setup_chinese_font(chinese_font_name='SimSun', western_font_name='Times New Roman'):
    """
    配置matplotlib以支持中文显示，并设置中文字体和西文字体
    
    Args:
        chinese_font_name: 中文字体名称，默认为'SimSun'（宋体）
        western_font_name: 西文字体名称，默认为'Times New Roman'
    
    Returns:
        tuple: (成功设置的中文字体名称, 成功设置的西文字体名称)
    """
    # 获取系统可用字体
    try:
        available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    except:
        available_fonts = []
    
    # 中文字体备选列表（按优先级）
    chinese_font_list = [
        'SimSun',           # 宋体（Windows）
        'NSimSun',          # 新宋体（Windows）
        'STSong',           # 华文宋体（Mac）
        'Songti SC',        # 宋体-简（Mac）
        'Noto Serif CJK SC', # Noto宋体（Linux）
        'WenQuanYi Micro Hei',  # 文泉驿微米黑（Linux备选）
    ]
    
    # 西文字体备选列表（按优先级）
    western_font_list = [
        'Times New Roman',  # Times New Roman（Windows/Mac）
        'Times',            # Times（Linux）
        'DejaVu Serif',     # DejaVu Serif（Linux备选）
        'Liberation Serif', # Liberation Serif（Linux备选）
    ]
    
    # 查找中文字体
    chinese_font = None
    if chinese_font_name in available_fonts:
        chinese_font = chinese_font_name
    else:
        # 尝试备选字体
        for font in chinese_font_list:
            if font in available_fonts:
                chinese_font = font
                break
    
    # 查找西文字体
    western_font = None
    if western_font_name in available_fonts:
        western_font = western_font_name
    else:
        # 尝试备选字体
        for font in western_font_list:
            if font in available_fonts:
                western_font = font
                break
    
    # 设置字体
    if chinese_font:
        # 设置中文字体（sans-serif用于中文）
        plt.rcParams['font.sans-serif'] = [chinese_font] + plt.rcParams['font.sans-serif']
        print(f"✓ 中文字体已设置: {chinese_font}")
    else:
        print("⚠️  警告: 未找到中文字体，中文可能显示为方框")
        if available_fonts:
            print(f"   可用字体示例: {available_fonts[:10]}")
    
    if western_font:
        # 设置西文字体（serif用于西文）
        plt.rcParams['font.serif'] = [western_font] + plt.rcParams['font.serif']
        # 同时设置数学字体
        plt.rcParams['mathtext.fontset'] = 'stix'  # 使用STIX字体（包含Times风格）
        plt.rcParams['mathtext.default'] = 'rm'  # 默认使用罗马字体
        print(f"✓ 西文字体已设置: {western_font}")
    else:
        print("⚠️  警告: 未找到西文字体 Times New Roman")
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    return chinese_font, western_font


def list_available_chinese_fonts():
    """
    列出系统中可用的中文字体
    
    Returns:
        list: 可用的中文字体列表
    """
    try:
        available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
        # 常见中文字体关键词
        chinese_keywords = ['SimHei', 'YaHei', 'WenQuanYi', 'Noto', 'STHeiti', 'CJK', 'SC', 'CN']
        chinese_fonts = []
        for font in available_fonts:
            for keyword in chinese_keywords:
                if keyword in font:
                    chinese_fonts.append(font)
                    break
        return sorted(set(chinese_fonts))
    except:
        return []


# 如果直接导入此模块，自动配置中文字体（宋体）和西文字体（Times New Roman）
if __name__ != "__main__":
    setup_chinese_font(chinese_font_name='SimSun', western_font_name='Times New Roman')


if __name__ == "__main__":
    # 测试中文字体配置
    print("=" * 60)
    print("Matplotlib字体配置测试")
    print("=" * 60)
    
    chinese_font, western_font = setup_chinese_font(chinese_font_name='SimSun', western_font_name='Times New Roman')
    print(f"\n当前设置的中文字体: {chinese_font}")
    print(f"当前设置的西文字体: {western_font}")
    
    print("\n可用的中文字体:")
    chinese_fonts = list_available_chinese_fonts()
    if chinese_fonts:
        for f in chinese_fonts:
            print(f"  - {f}")
    else:
        print("  未找到中文字体")
    
    # 测试中英文显示
    print("\n测试中英文显示...")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 2, 3, 5]
    ax.plot(x, y, 'o-', label='Test Data / 测试数据', linewidth=2, markersize=8)
    ax.set_title('Font Test: Chinese (宋体) & English (Times New Roman)', fontsize=16)
    ax.set_xlabel('X Axis Label / 横轴标签', fontsize=12)
    ax.set_ylabel('Y Axis Label / 纵轴标签', fontsize=12)
    ax.text(2.5, 4.5, 'Chinese: 中文显示测试\nEnglish: Times New Roman Font', 
             fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('matplotlib_font_test.png', dpi=150, bbox_inches='tight')
    print("✓ 测试图片已保存: matplotlib_font_test.png")
    plt.close()
    
    print("\n" + "=" * 60)
    print("配置完成！")

