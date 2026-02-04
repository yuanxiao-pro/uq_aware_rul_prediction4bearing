# 安装 Windows 宋体（SimSun）字体指南

## 方法1：从 Windows 系统复制（推荐）

### 步骤：

1. **在 Windows 系统中找到字体文件**
   - 路径：`C:\Windows\Fonts\simsun.ttc`
   - 或者：`C:\Windows\Fonts\SimSun.ttf`

2. **复制到 Linux 系统**

   **如果使用 WSL：**
   ```bash
   cp /mnt/c/Windows/Fonts/simsun.ttc ~/.local/share/fonts/
   ```

   **如果使用 scp（从远程 Windows）：**
   ```bash
   scp user@windows_host:/mnt/c/Windows/Fonts/simsun.ttc ~/.local/share/fonts/
   ```

   **如果手动复制：**
   ```bash
   # 创建字体目录
   mkdir -p ~/.local/share/fonts
   
   # 复制字体文件（替换 /path/to/ 为实际路径）
   cp /path/to/simsun.ttc ~/.local/share/fonts/
   ```

3. **刷新字体缓存**
   ```bash
   fc-cache -fv
   ```

4. **验证安装**
   ```bash
   fc-list | grep -i simsun
   ```

## 方法2：使用安装脚本

运行交互式安装脚本：
```bash
chmod +x install_simsun_font.sh
./install_simsun_font.sh
```

或者运行自动检测脚本：
```bash
python3 install_simsun_auto.py
```

## 方法3：使用替代字体（已安装）

如果无法获取 Windows 宋体，系统已安装 **Noto Serif CJK SC**，可以作为宋体的替代：

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Serif CJK SC'] + plt.rcParams['font.sans-serif']
```

## 快速安装命令（如果您已有字体文件）

```bash
# 假设字体文件在当前目录或指定路径
FONT_FILE="/path/to/simsun.ttc"  # 替换为实际路径
FONT_DIR="$HOME/.local/share/fonts"

mkdir -p "$FONT_DIR"
cp "$FONT_FILE" "$FONT_DIR/simsun.ttc"
fc-cache -fv
fc-list | grep -i simsun
```

## 验证安装

运行以下 Python 脚本验证：

```python
import matplotlib.font_manager as fm

fonts = [f.name for f in fm.fontManager.ttflist]
if any('simsun' in f.lower() or '宋体' in f for f in fonts):
    print("✓ 宋体字体已安装")
    for f in fonts:
        if 'simsun' in f.lower() or '宋体' in f:
            print(f"  字体名称: {f}")
else:
    print("✗ 未找到宋体字体")
```

## 注意事项

1. **版权**：Windows 字体受版权保护，请确保您有合法授权使用
2. **文件格式**：SimSun 字体通常是 `.ttc`（TrueType Collection）格式
3. **权限**：确保字体目录有读取权限
4. **缓存**：安装后需要刷新字体缓存才能生效

