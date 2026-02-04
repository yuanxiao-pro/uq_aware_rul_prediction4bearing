#!/usr/bin/env python3
"""
最终版本的 matplotlib 中文字体配置
支持 Noto Serif CJK SC（宋体替代）和 Liberation Serif（Times New Roman 替代）
"""

import os
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

def setup_chinese_font(chinese_font_name='Noto Serif CJK SC', western_font_name='Liberation Serif'):
    """
    配置 matplotlib 以支持中文显示（宋体替代）和西文（Times New Roman 替代）
    
    Args:
        chinese_font_name: 中文字体名称，默认为 'Noto Serif CJK SC'
        western_font_name: 西文字体名称，默认为 'Liberation Serif'
    
    Returns:
        tuple: (chinese_font, western_font) 实际使用的字体名称
    """
    # 中文字体文件路径
    noto_serif_sc = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
    noto_sans_sc = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    
    # 添加字体文件到 matplotlib（如果存在）
    chinese_font_path = None
    if os.path.exists(noto_serif_sc):
        chinese_font_path = noto_serif_sc
    elif os.path.exists(noto_sans_sc):
        chinese_font_path = noto_sans_sc
    
    if chinese_font_path:
        try:
            fm.fontManager.addfont(chinese_font_path)
        except Exception:
            pass  # 如果已经添加过，会报错，忽略即可
    
    # 检查字体是否可用
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 查找中文字体
    chinese_font = None
    if chinese_font_name in available_fonts:
        chinese_font = chinese_font_name
    else:
        # 尝试查找包含 "Noto" 和 "CJK" 的字体
        for font in available_fonts:
            if 'Noto' in font and 'CJK' in font and ('SC' in font or 'Serif' in font):
                chinese_font = font
                break
    
    # 查找西文字体
    western_font = None
    if western_font_name in available_fonts:
        western_font = western_font_name
    else:
        # 尝试查找替代字体
        for font in available_fonts:
            if 'Liberation' in font and 'Serif' in font:
                western_font = font
                break
            elif 'Times' in font:
                western_font = font
                break
    
    # 配置 matplotlib
    if chinese_font:
        # 将中文字体放在最前面
        plt.rcParams['font.sans-serif'] = [chinese_font] + [f for f in plt.rcParams['font.sans-serif'] if f != chinese_font]
        print(f"✓ 中文字体已设置: {chinese_font}")
    else:
        print("⚠️  警告: 未找到中文字体，中文可能显示为方框")
    
    if western_font:
        plt.rcParams['font.serif'] = [western_font] + [f for f in plt.rcParams['font.serif'] if f != western_font]
        plt.rcParams['mathtext.fontset'] = 'stix'
        print(f"✓ 西文字体已设置: {western_font}")
    else:
        print("⚠️  警告: 未找到西文字体 Times New Roman")
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    return chinese_font, western_font

# 如果直接导入此模块，自动配置字体
if __name__ != "__main__":
    setup_chinese_font()

if __name__ == "__main__":
    # 测试字体配置
    print("=" * 60)
    print("Matplotlib 字体配置测试")
    print("=" * 60)
    
    chinese_font, western_font = setup_chinese_font()
    print(f"\n当前设置的中文字体: {chinese_font}")
    print(f"当前设置的西文字体: {western_font}")
    
    # 测试中英文显示
    print("\n测试中英文显示...")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 2, 3, 5]
    ax.plot(x, y, 'o-', label='Test Data / 测试数据', linewidth=2, markersize=8)
    ax.set_title('Font Test: Chinese (宋体替代) & English (Times New Roman 替代)', fontsize=16)
    ax.set_xlabel('X Axis Label / 横轴标签', fontsize=12)
    ax.set_ylabel('Y Axis Label / 纵轴标签', fontsize=12)
    ax.text(2.5, 4.5, 'Chinese: 中文显示测试\nEnglish: Liberation Serif Font', 
             fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('matplotlib_font_test_final.png', dpi=150, bbox_inches='tight')
    print("✓ 测试图片已保存: matplotlib_font_test_final.png")
    plt.close()
    
    print("\n" + "=" * 60)
    print("配置完成！")
    print("=" * 60)

