#!/bin/bash
# Linux字体安装脚本
# 安装SimSun（宋体）和Times New Roman字体

echo "=========================================="
echo "Linux字体安装脚本"
echo "安装 SimSun（宋体）和 Times New Roman"
echo "=========================================="

# 检查是否有root权限
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  需要root权限来安装系统字体"
    echo "   请使用: sudo bash install_fonts_linux.sh"
    exit 1
fi

# 创建字体目录
FONT_DIR="/usr/share/fonts/truetype"
SIMSON_DIR="$FONT_DIR/simsun"
TIMES_DIR="$FONT_DIR/times-new-roman"

echo ""
echo "1. 创建字体目录..."
mkdir -p "$SIMSON_DIR"
mkdir -p "$TIMES_DIR"

echo ""
echo "2. 检查字体文件..."
echo "   请确保以下字体文件存在："
echo "   - simsun.ttc 或 simsun.ttf (SimSun宋体)"
echo "   - times.ttf 或 timesnr.ttf (Times New Roman)"
echo ""

# 检查是否有字体文件在当前目录
HAS_SIMSUN=false
HAS_TIMES=false

if [ -f "simsun.ttc" ] || [ -f "simsun.ttf" ]; then
    HAS_SIMSUN=true
    echo "   ✓ 找到SimSun字体文件"
fi

if [ -f "times.ttf" ] || [ -f "timesnr.ttf" ] || [ -f "times-new-roman.ttf" ]; then
    HAS_TIMES=true
    echo "   ✓ 找到Times New Roman字体文件"
fi

if [ "$HAS_SIMSUN" = false ] || [ "$HAS_TIMES" = false ]; then
    echo ""
    echo "⚠️  未找到字体文件，请按照以下步骤操作："
    echo ""
    echo "方法1: 从Windows系统复制字体（推荐）"
    echo "   1. 在Windows系统中找到以下字体文件："
    echo "      - C:/Windows/Fonts/simsun.ttc (SimSun宋体)"
    echo "      - C:/Windows/Fonts/times.ttf 或 timesnr.ttf (Times New Roman)"
    echo "   2. 将字体文件复制到当前目录"
    echo "   3. 重新运行此脚本"
    echo ""
    echo "方法2: 使用包管理器安装（部分Linux发行版）"
    echo "   Ubuntu/Debian:"
    echo "     sudo apt-get update"
    echo "     sudo apt-get install fonts-noto-cjk fonts-liberation"
    echo ""
    echo "   CentOS/RHEL:"
    echo "     sudo yum install google-noto-cjk-fonts liberation-fonts"
    echo ""
    echo "方法3: 手动下载字体"
    echo "   - SimSun: 需要从Windows系统获取"
    echo "   - Times New Roman: 可以尝试安装 liberation-serif 作为替代"
    echo ""
    read -p "是否继续安装已找到的字体？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 复制SimSun字体
if [ "$HAS_SIMSUN" = true ]; then
    echo ""
    echo "3. 安装SimSun字体..."
    if [ -f "simsun.ttc" ]; then
        cp simsun.ttc "$SIMSON_DIR/"
        echo "   ✓ 已复制 simsun.ttc"
    elif [ -f "simsun.ttf" ]; then
        cp simsun.ttf "$SIMSON_DIR/"
        echo "   ✓ 已复制 simsun.ttf"
    fi
fi

# 复制Times New Roman字体
if [ "$HAS_TIMES" = true ]; then
    echo ""
    echo "4. 安装Times New Roman字体..."
    if [ -f "times.ttf" ]; then
        cp times.ttf "$TIMES_DIR/"
        echo "   ✓ 已复制 times.ttf"
    elif [ -f "timesnr.ttf" ]; then
        cp timesnr.ttf "$TIMES_DIR/"
        echo "   ✓ 已复制 timesnr.ttf"
    elif [ -f "times-new-roman.ttf" ]; then
        cp times-new-roman.ttf "$TIMES_DIR/"
        echo "   ✓ 已复制 times-new-roman.ttf"
    fi
fi

# 更新字体缓存
echo ""
echo "5. 更新字体缓存..."
fc-cache -fv

echo ""
echo "=========================================="
echo "字体安装完成！"
echo "=========================================="
echo ""
echo "验证字体是否安装成功："
echo "  fc-list | grep -i simsun"
echo "  fc-list | grep -i times"
echo ""
echo "如果看到字体名称，说明安装成功。"
echo "可能需要重启Python程序或重新加载matplotlib才能使用新字体。"
echo ""

