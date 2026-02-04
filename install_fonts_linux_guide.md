# Linux系统安装SimSun和Times New Roman字体指南

## 方法1: 从Windows系统复制字体（推荐）

### 步骤1: 获取字体文件

在Windows系统中找到以下字体文件：
- **SimSun（宋体）**: `C:\Windows\Fonts\simsun.ttc` 或 `simsun.ttf`
- **Times New Roman**: `C:\Windows\Fonts\times.ttf` 或 `timesnr.ttf`

### 步骤2: 复制到Linux系统

将字体文件复制到Linux系统的当前目录。

### 步骤3: 运行安装脚本

```bash
# 给脚本添加执行权限
chmod +x install_fonts_linux.sh

# 运行安装脚本（需要root权限）
sudo bash install_fonts_linux.sh
```

### 步骤4: 验证安装

```bash
# 检查SimSun字体
fc-list | grep -i simsun

# 检查Times New Roman字体
fc-list | grep -i times
```

如果看到字体名称，说明安装成功。

---

## 方法2: 手动安装字体

### 步骤1: 创建字体目录

```bash
sudo mkdir -p /usr/share/fonts/truetype/simsun
sudo mkdir -p /usr/share/fonts/truetype/times-new-roman
```

### 步骤2: 复制字体文件

```bash
# 复制SimSun字体（假设字体文件在当前目录）
sudo cp simsun.ttc /usr/share/fonts/truetype/simsun/
# 或
sudo cp simsun.ttf /usr/share/fonts/truetype/simsun/

# 复制Times New Roman字体
sudo cp times.ttf /usr/share/fonts/truetype/times-new-roman/
# 或
sudo cp timesnr.ttf /usr/share/fonts/truetype/times-new-roman/
```

### 步骤3: 设置字体权限

```bash
sudo chmod 644 /usr/share/fonts/truetype/simsun/*
sudo chmod 644 /usr/share/fonts/truetype/times-new-roman/*
```

### 步骤4: 更新字体缓存

```bash
sudo fc-cache -fv
```

---

## 方法3: 使用包管理器安装替代字体

如果无法获取Windows字体文件，可以使用Linux发行版的字体包：

### Ubuntu/Debian系统

```bash
# 安装中文字体（Noto CJK包含宋体风格）
sudo apt-get update
sudo apt-get install fonts-noto-cjk fonts-noto-cjk-extra

# 安装Times New Roman替代字体（Liberation Serif）
sudo apt-get install fonts-liberation

# 更新字体缓存
sudo fc-cache -fv
```

### CentOS/RHEL系统

```bash
# 安装中文字体
sudo yum install google-noto-cjk-fonts

# 安装Times New Roman替代字体
sudo yum install liberation-fonts

# 更新字体缓存
sudo fc-cache -fv
```

### Fedora系统

```bash
# 安装中文字体
sudo dnf install google-noto-cjk-fonts

# 安装Times New Roman替代字体
sudo dnf install liberation-fonts

# 更新字体缓存
sudo fc-cache -fv
```

---

## 方法4: 用户级安装（不需要root权限）

如果不想使用root权限，可以安装到用户目录：

```bash
# 创建用户字体目录
mkdir -p ~/.fonts/simsun
mkdir -p ~/.fonts/times-new-roman

# 复制字体文件
cp simsun.ttc ~/.fonts/simsun/
cp times.ttf ~/.fonts/times-new-roman/

# 更新字体缓存
fc-cache -fv
```

---

## 验证字体安装

### 方法1: 使用fc-list命令

```bash
# 列出所有SimSun相关字体
fc-list | grep -i simsun

# 列出所有Times相关字体
fc-list | grep -i times
```

### 方法2: 使用Python测试

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 列出所有可用字体
fonts = [f.name for f in fm.fontManager.ttflist]
print("SimSun字体:", [f for f in fonts if 'simsun' in f.lower() or 'song' in f.lower()])
print("Times字体:", [f for f in fonts if 'times' in f.lower()])

# 测试显示
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.figure(figsize=(8, 6))
plt.text(0.5, 0.5, '中文测试 / Times New Roman', 
         fontsize=20, ha='center', va='center')
plt.title('字体测试')
plt.savefig('font_test.png')
print("测试图片已保存: font_test.png")
```

---

## 常见问题

### Q1: 安装后Python仍然无法识别字体

**解决方案：**
1. 确保已运行 `fc-cache -fv` 更新字体缓存
2. 重启Python程序
3. 清除matplotlib缓存：
   ```bash
   rm -rf ~/.cache/matplotlib
   ```
4. 在Python中重新加载matplotlib：
   ```python
   import matplotlib
   matplotlib.font_manager._rebuild()
   ```

### Q2: 找不到SimSun字体文件

**解决方案：**
- SimSun是Windows专有字体，需要从Windows系统复制
- 或者使用Linux替代字体，如Noto Serif CJK SC

### Q3: Times New Roman显示为其他字体

**解决方案：**
- 检查字体名称是否正确：`fc-list | grep -i times`
- 可能需要使用 `Times` 而不是 `Times New Roman`
- 或者使用Liberation Serif作为替代

### Q4: 字体安装后需要重启系统吗？

**不需要。** 只需：
1. 运行 `fc-cache -fv` 更新字体缓存
2. 重启Python程序
3. 清除matplotlib缓存（可选）

---

## 推荐的字体组合

如果无法安装Windows字体，可以使用以下Linux替代方案：

| Windows字体 | Linux替代字体 |
|------------|--------------|
| SimSun（宋体） | Noto Serif CJK SC, STSong |
| Times New Roman | Liberation Serif, Times, DejaVu Serif |

在 `matplotlib_chinese_config.py` 中，这些替代字体已经在备选列表中，会自动使用。

---

## 快速安装命令（Ubuntu/Debian）

```bash
# 一键安装所有需要的字体
sudo apt-get update && \
sudo apt-get install -y fonts-noto-cjk fonts-liberation && \
sudo fc-cache -fv && \
echo "字体安装完成！"
```

然后修改 `matplotlib_chinese_config.py` 中的字体名称：
- 中文字体：使用 `Noto Serif CJK SC` 或 `STSong`
- 西文字体：使用 `Liberation Serif` 或 `Times`

