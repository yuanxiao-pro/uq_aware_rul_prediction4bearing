#!/usr/bin/env python3
"""
安装标准 SimSun 字体（不是 ExtB 版本）

标准 SimSun 字体通常来自 Windows 系统：
C:\Windows\Fonts\simsun.ttc

如果用户有 Windows 系统，可以从那里复制。
如果用户没有 Windows 系统，可以尝试从其他来源获取。
"""

import os
import shutil
import sys

def check_and_install_standard_simsun():
    """检查并安装标准 SimSun 字体"""
    
    # 可能的来源路径
    possible_sources = [
        "/mnt/c/Windows/Fonts/simsun.ttc",  # WSL
        "/mnt/windows/Windows/Fonts/simsun.ttc",  # 其他挂载点
        os.path.expanduser("~/simsun.ttc"),  # 用户主目录
        "/tmp/simsun.ttc",  # 临时目录
    ]
    
    target_dir = os.path.expanduser("~/.local/share/fonts")
    target_path = os.path.join(target_dir, "simsun_standard.ttc")
    
    # 检查当前字体
    current_simsun = os.path.expanduser("~/.local/share/fonts/simsun.ttc")
    if os.path.exists(current_simsun):
        print(f"当前 SimSun 字体: {current_simsun}")
        print("正在检查是否为标准 SimSun...")
        
        # 检查是否是 ExtB
        try:
            from fontTools.ttLib import TTFont
            font = TTFont(current_simsun)
            if 'name' in font:
                name_table = font['name']
                for record in name_table.names:
                    if record.nameID == 4:  # Full name
                        try:
                            name_str = record.toUnicode()
                            if 'ExtB' in name_str:
                                print(f"  ✗ 当前字体是 SimSun-ExtB（缺少常用字）")
                                print(f"  需要标准 SimSun 字体")
                                break
                            elif 'SimSun' in name_str and 'ExtB' not in name_str:
                                print(f"  ✓ 当前字体是标准 SimSun")
                                return True
                        except:
                            pass
        except ImportError:
            print("  需要安装 fonttools: pip install fonttools")
        except Exception as e:
            print(f"  检查字体时出错: {e}")
    
    # 尝试从可能的来源复制
    print("\n正在查找标准 SimSun 字体...")
    for source in possible_sources:
        if os.path.exists(source):
            print(f"  找到字体文件: {source}")
            try:
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy2(source, target_path)
                print(f"  ✓ 已复制到: {target_path}")
                
                # 验证是否是标准 SimSun
                try:
                    from fontTools.ttLib import TTFont
                    font = TTFont(target_path)
                    if 'name' in font:
                        name_table = font['name']
                        for record in name_table.names:
                            if record.nameID == 4:  # Full name
                                try:
                                    name_str = record.toUnicode()
                                    if 'SimSun' in name_str and 'ExtB' not in name_str:
                                        print(f"  ✓ 确认是标准 SimSun")
                                        # 更新字体缓存
                                        os.system("fc-cache -fv ~/.local/share/fonts")
                                        return True
                                except:
                                    pass
                except:
                    pass
                
                return True
            except Exception as e:
                print(f"  ✗ 复制失败: {e}")
    
    print("\n未找到标准 SimSun 字体文件")
    print("\n解决方案：")
    print("1. 如果您有 Windows 系统，请从以下位置复制 simsun.ttc：")
    print("   C:\\Windows\\Fonts\\simsun.ttc")
    print("   然后运行：")
    print(f"   cp /path/to/simsun.ttc {target_path}")
    print("\n2. 或者，如果您在 WSL 中，可以：")
    print("   cp /mnt/c/Windows/Fonts/simsun.ttc ~/.local/share/fonts/simsun_standard.ttc")
    print("\n3. 安装后运行：")
    print("   fc-cache -fv ~/.local/share/fonts")
    
    return False

if __name__ == "__main__":
    success = check_and_install_standard_simsun()
    sys.exit(0 if success else 1)

