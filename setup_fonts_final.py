#!/usr/bin/env python3
"""
最终字体配置脚本 - 直接使用字体文件路径
"""

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import os

def setup_fonts():
    """配置 matplotlib 字体"""
    print("=" * 60)
    print("配置 matplotlib 字体")
    print("=" * 60)
    
    # 中文字体文件路径
    noto_serif_sc = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
    noto_sans_sc = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    
    # 检查字体文件是否存在
    chinese_font_path = None
    if os.path.exists(noto_serif_sc):
        chinese_font_path = noto_serif_sc
        print(f"✓ 找到中文字体文件: {noto_serif_sc}")
    elif os.path.exists(noto_sans_sc):
        chinese_font_path = noto_sans_sc
        print(f"✓ 找到中文字体文件: {noto_sans_sc}")
    else:
        print("✗ 未找到 Noto CJK 字体文件")
    
    # 添加字体到 matplotlib
    if chinese_font_path:
        try:
            # 直接添加字体文件
            fm.fontManager.addfont(chinese_font_path)
            print("✓ 已添加中文字体到 matplotlib")
            
            # 使用 fc-query 获取字体名称（优先选择 SC）
            import subprocess
            result = subprocess.run(['fc-query', '--format=%{family}\n', chinese_font_path], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                # 按行分割，去除空行
                families = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
                # 优先选择 SC（简体中文）
                chinese_font_name = 'Noto Serif CJK SC'  # 默认值
                for family in families:
                    if 'SC' in family and 'Noto' in family:
                        chinese_font_name = family.strip()
                        break
                if chinese_font_name == 'Noto Serif CJK SC' and families:
                    # 如果没有找到 SC，使用第一个
                    chinese_font_name = families[0].strip()
            else:
                chinese_font_name = 'Noto Serif CJK SC'
            print(f"  字体名称: {chinese_font_name}")
        except Exception as e:
            print(f"  添加字体时出错: {e}")
            chinese_font_name = 'Noto Serif CJK SC'
    else:
        chinese_font_name = None
    
    # 西文字体（Times New Roman 或替代）
    western_fonts = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    western_font = None
    for font in western_fonts:
        if font in [f.name for f in fm.fontManager.ttflist]:
            western_font = font
            print(f"✓ 找到西文字体: {western_font}")
            break
    
    if western_font is None:
        western_font = 'Liberation Serif'
        print(f"  使用默认西文字体: {western_font}")
    
    # 配置 matplotlib
    print("\n配置 matplotlib 字体设置...")
    if chinese_font_name:
        plt.rcParams['font.sans-serif'] = [chinese_font_name] + plt.rcParams['font.sans-serif']
        print(f"  中文字体: {chinese_font_name}")
    if western_font:
        plt.rcParams['font.serif'] = [western_font] + plt.rcParams['font.serif']
        plt.rcParams['mathtext.fontset'] = 'stix'
        print(f"  西文字体: {western_font}")
    
    plt.rcParams['axes.unicode_minus'] = False
    
    # 测试
    print("\n测试字体显示...")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.7, '中文字体测试：宋体样式', 
                fontsize=16, ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.5, 'Times New Roman Style Font', 
                fontsize=14, ha='center', transform=ax.transAxes, family='serif')
        ax.text(0.5, 0.3, '数学公式: $y = ax^2 + bx + c$', 
                fontsize=14, ha='center', transform=ax.transAxes)
        ax.set_title('字体测试 / Font Test', fontsize=18)
        ax.axis('off')
        
        test_file = 'font_test_final.png'
        plt.savefig(test_file, dpi=150, bbox_inches='tight')
        print(f"✓ 测试图片已保存: {test_file}")
        plt.close()
    except Exception as e:
        print(f"✗ 测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("配置完成！")
    print("=" * 60)
    
    return chinese_font_name, western_font

if __name__ == "__main__":
    setup_fonts()

