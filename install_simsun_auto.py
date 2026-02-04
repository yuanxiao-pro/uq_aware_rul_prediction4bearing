#!/usr/bin/env python3
"""
自动安装 Windows 宋体（SimSun）字体
支持多种安装方式
"""

import os
import shutil
import subprocess
import sys

def check_font_installed():
    """检查字体是否已安装"""
    try:
        result = subprocess.run(['fc-list'], capture_output=True, text=True)
        if 'simsun' in result.stdout.lower() or '宋体' in result.stdout:
            return True
    except:
        pass
    return False

def find_windows_fonts():
    """查找 Windows 字体目录"""
    possible_paths = [
        '/mnt/c/Windows/Fonts/simsun.ttc',  # WSL
        '/mnt/c/Windows/Fonts/SimSun.ttf',
        '/media/*/Windows/Fonts/simsun.ttc',  # 挂载的 Windows 分区
        '/media/*/Windows/Fonts/SimSun.ttf',
    ]
    
    for path_pattern in possible_paths:
        if '*' in path_pattern:
            import glob
            matches = glob.glob(path_pattern)
            if matches:
                return matches[0]
        elif os.path.exists(path_pattern):
            return path_pattern
    
    return None

def install_font_from_path(source_path, dest_dir):
    """从指定路径安装字体"""
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, 'simsun.ttc')
    
    try:
        shutil.copy2(source_path, dest_path)
        print(f"✓ 字体文件已复制到: {dest_path}")
        return True
    except Exception as e:
        print(f"✗ 复制失败: {e}")
        return False

def refresh_font_cache():
    """刷新字体缓存"""
    try:
        subprocess.run(['fc-cache', '-fv'], capture_output=True, check=True)
        print("✓ 字体缓存已刷新")
        return True
    except Exception as e:
        print(f"⚠️  刷新字体缓存失败: {e}")
        return False

def main():
    print("=" * 60)
    print("自动安装 Windows 宋体（SimSun）字体")
    print("=" * 60)
    
    # 检查是否已安装
    if check_font_installed():
        print("✓ 宋体字体已安装")
        return
    
    # 查找 Windows 字体目录
    print("\n正在查找 Windows 字体文件...")
    windows_font_path = find_windows_fonts()
    
    font_dir = os.path.expanduser('~/.local/share/fonts')
    
    if windows_font_path and os.path.exists(windows_font_path):
        print(f"✓ 找到字体文件: {windows_font_path}")
        print(f"  正在复制到: {font_dir}/")
        
        if install_font_from_path(windows_font_path, font_dir):
            refresh_font_cache()
            if check_font_installed():
                print("\n✓ 宋体字体安装成功！")
                return
            else:
                print("\n⚠️  字体文件已安装，但可能需要重启应用程序")
                return
    
    # 如果找不到，提供手动安装指导
    print("\n" + "=" * 60)
    print("未找到 Windows 字体文件")
    print("=" * 60)
    print("\n请选择以下方式之一安装宋体：")
    print("\n方法1：从 Windows 系统复制")
    print("  1. 在 Windows 系统中找到: C:\\Windows\\Fonts\\simsun.ttc")
    print("  2. 复制到 Linux 系统")
    print("  3. 运行以下命令：")
    print(f"     cp /path/to/simsun.ttc {font_dir}/")
    print("     fc-cache -fv")
    
    print("\n方法2：使用安装脚本")
    print("  运行: bash install_simsun_font.sh")
    
    print("\n方法3：使用替代字体（已安装）")
    print("  Noto Serif CJK SC 可以作为宋体的替代")
    print("  已在系统中安装，可以直接使用")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

