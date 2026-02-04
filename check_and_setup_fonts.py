#!/usr/bin/env python3
"""
检查和配置 matplotlib 字体（宋体和 Times New Roman）
"""

import os
import sys
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

def check_font_available(font_name):
    """检查字体是否可用"""
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    return font_name in available_fonts

def find_similar_fonts(keywords):
    """查找包含关键词的字体"""
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    matches = []
    for font in available_fonts:
        for keyword in keywords:
            if keyword.lower() in font.lower():
                matches.append(font)
                break
    return sorted(set(matches))

def setup_fonts():
    """设置 matplotlib 字体"""
    print("=" * 60)
    print("字体检查和配置")
    print("=" * 60)
    
    # 检查中文字体（宋体）
    chinese_fonts = ['SimSun', 'NSimSun', 'STSong', 'Songti SC', 'Noto Sans CJK SC', 'Noto Serif CJK SC', 'Source Han Serif SC']
    chinese_font = None
    print("\n检查中文字体（宋体）：")
    for font in chinese_fonts:
        if check_font_available(font):
            chinese_font = font
            print(f"  ✓ 找到字体: {font}")
            break
        else:
            print(f"  ✗ 未找到: {font}")
    
    if chinese_font is None:
        print("\n  未找到宋体，查找替代中文字体：")
        alternatives = find_similar_fonts(['song', 'simsun', 'noto', 'source', 'han', 'cjk', 'serif', 'sans'])
        if alternatives:
            # 优先选择 Noto Serif CJK SC（最接近宋体）
            for alt in alternatives:
                if 'Noto Serif CJK SC' in alt or 'Noto Sans CJK SC' in alt:
                    chinese_font = alt
                    break
            if chinese_font is None:
                chinese_font = alternatives[0]
            print(f"  使用替代字体: {chinese_font}")
        else:
            print("  ⚠️  警告: 未找到合适的中文字体")
    
    # 检查西文字体（Times New Roman）
    western_fonts = ['Times New Roman', 'Times', 'Liberation Serif', 'TeX Gyre Termes', 'DejaVu Serif']
    western_font = None
    print("\n检查西文字体（Times New Roman）：")
    for font in western_fonts:
        if check_font_available(font):
            western_font = font
            print(f"  ✓ 找到字体: {font}")
            break
        else:
            print(f"  ✗ 未找到: {font}")
    
    if western_font is None:
        print("\n  未找到 Times New Roman，查找替代西文字体：")
        alternatives = find_similar_fonts(['times', 'liberation', 'serif', 'tex', 'gyre'])
        if alternatives:
            western_font = alternatives[0]
            print(f"  使用替代字体: {western_font}")
        else:
            print("  ⚠️  警告: 未找到合适的西文字体")
            western_font = 'DejaVu Serif'  # 默认字体
    
    # 配置 matplotlib
    print("\n配置 matplotlib 字体：")
    if chinese_font:
        plt.rcParams['font.sans-serif'] = [chinese_font] + plt.rcParams['font.sans-serif']
        print(f"  中文字体: {chinese_font}")
    if western_font:
        plt.rcParams['font.serif'] = [western_font] + plt.rcParams['font.serif']
        plt.rcParams['mathtext.fontset'] = 'stix'
        print(f"  西文字体: {western_font}")
    
    plt.rcParams['axes.unicode_minus'] = False
    
    # 测试显示
    print("\n测试字体显示：")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.7, '中文字体测试 / Chinese Font Test', 
                fontsize=16, ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.5, 'Times New Roman Style Font', 
                fontsize=14, ha='center', transform=ax.transAxes, family='serif')
        ax.text(0.5, 0.3, '数学公式: $y = ax^2 + bx + c$', 
                fontsize=14, ha='center', transform=ax.transAxes)
        ax.set_title('字体测试 / Font Test', fontsize=18)
        ax.axis('off')
        
        test_file = 'font_test.png'
        plt.savefig(test_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ 测试图片已保存: {test_file}")
        plt.close()
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("配置完成！")
    print("=" * 60)
    
    return chinese_font, western_font

if __name__ == "__main__":
    chinese_font, western_font = setup_fonts()
    
    # 提供安装建议
    if chinese_font is None or 'Noto' not in chinese_font and 'SimSun' not in chinese_font:
        print("\n建议安装中文字体：")
        print("  Ubuntu/Debian: sudo apt-get install fonts-noto-cjk")
        print("  CentOS/RHEL: sudo yum install google-noto-cjk-fonts")
        print("  或从 Windows 复制 simsun.ttc 到 ~/.local/share/fonts/")
    
    if 'Times' not in western_font and 'Liberation' not in western_font:
        print("\n建议安装 Times New Roman 替代字体：")
        print("  Ubuntu/Debian: sudo apt-get install fonts-liberation")
        print("  CentOS/RHEL: sudo yum install liberation-serif-fonts")
        print("  或从 Windows 复制 times.ttf 到 ~/.local/share/fonts/")

