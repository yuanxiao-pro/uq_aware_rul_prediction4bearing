#!/bin/bash
# 安装中文字体（宋体）和西文字体（Times New Roman）的脚本

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}开始安装字体...${NC}"

# 创建用户字体目录
FONT_DIR="$HOME/.local/share/fonts"
mkdir -p "$FONT_DIR"

# 检查是否已有字体文件
SIMSUM_FOUND=false
TIMES_FOUND=false

# 检查 SimSun（宋体）
if [ -f "$FONT_DIR/simsun.ttc" ] || [ -f "$FONT_DIR/SimSun.ttf" ] || [ -f "$FONT_DIR/simsun.ttf" ]; then
    echo -e "${GREEN}✓ 宋体字体已存在${NC}"
    SIMSUM_FOUND=true
fi

# 检查 Times New Roman
if [ -f "$FONT_DIR/times.ttf" ] || [ -f "$FONT_DIR/TimesNewRoman.ttf" ] || [ -f "$FONT_DIR/timesnr.ttf" ]; then
    echo -e "${GREEN}✓ Times New Roman 字体已存在${NC}"
    TIMES_FOUND=true
fi

# 如果字体不存在，提供安装说明
if [ "$SIMSUM_FOUND" = false ] || [ "$TIMES_FOUND" = false ]; then
    echo -e "${YELLOW}需要安装字体文件。请按照以下步骤操作：${NC}"
    echo ""
    
    if [ "$SIMSUM_FOUND" = false ]; then
        echo -e "${YELLOW}1. 安装宋体（SimSun）：${NC}"
        echo "   方法1：从 Windows 系统复制"
        echo "   - 在 Windows 系统中找到：C:\\Windows\\Fonts\\simsun.ttc"
        echo "   - 复制到当前系统的字体目录：$FONT_DIR/"
        echo "   - 或者使用以下命令（如果有 Windows 系统访问权限）："
        echo "     cp /path/to/windows/fonts/simsun.ttc $FONT_DIR/"
        echo ""
        echo "   方法2：使用开源替代字体"
        echo "   - 可以安装 Noto Sans CJK SC 或 Source Han Serif SC"
        echo "   - sudo apt-get install fonts-noto-cjk  # Ubuntu/Debian"
        echo "   - sudo yum install google-noto-cjk-fonts  # CentOS/RHEL"
        echo ""
    fi
    
    if [ "$TIMES_FOUND" = false ]; then
        echo -e "${YELLOW}2. 安装 Times New Roman：${NC}"
        echo "   方法1：从 Windows 系统复制"
        echo "   - 在 Windows 系统中找到：C:\\Windows\\Fonts\\times.ttf 或 timesnr.ttf"
        echo "   - 复制到当前系统的字体目录：$FONT_DIR/"
        echo ""
        echo "   方法2：使用开源替代字体"
        echo "   - 可以安装 Liberation Serif 或 TeX Gyre Termes"
        echo "   - sudo apt-get install fonts-liberation  # Ubuntu/Debian"
        echo "   - sudo yum install liberation-serif-fonts  # CentOS/RHEL"
        echo ""
    fi
    
    echo -e "${YELLOW}3. 安装字体后，运行以下命令刷新字体缓存：${NC}"
    echo "   fc-cache -fv"
    echo ""
    echo -e "${YELLOW}4. 验证字体是否安装成功：${NC}"
    echo "   fc-list | grep -i 'simsun\\|宋体'"
    echo "   fc-list | grep -i 'times'"
    echo ""
    
    # 尝试安装开源替代字体
    echo -e "${YELLOW}是否要安装开源替代字体？(y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        # 检测系统类型
        if command -v apt-get &> /dev/null; then
            echo "检测到 Debian/Ubuntu 系统"
            if [ "$SIMSUM_FOUND" = false ]; then
                echo "安装 Noto Sans CJK SC（中文字体替代）..."
                sudo apt-get update
                sudo apt-get install -y fonts-noto-cjk
            fi
            if [ "$TIMES_FOUND" = false ]; then
                echo "安装 Liberation Serif（Times New Roman 替代）..."
                sudo apt-get install -y fonts-liberation
            fi
        elif command -v yum &> /dev/null; then
            echo "检测到 CentOS/RHEL 系统"
            if [ "$SIMSUM_FOUND" = false ]; then
                echo "安装 Noto Sans CJK SC（中文字体替代）..."
                sudo yum install -y google-noto-cjk-fonts
            fi
            if [ "$TIMES_FOUND" = false ]; then
                echo "安装 Liberation Serif（Times New Roman 替代）..."
                sudo yum install -y liberation-serif-fonts
            fi
        else
            echo -e "${RED}未检测到支持的包管理器，请手动安装字体${NC}"
        fi
    fi
fi

# 刷新字体缓存
echo ""
echo -e "${GREEN}刷新字体缓存...${NC}"
fc-cache -fv

# 验证字体
echo ""
echo -e "${GREEN}验证已安装的字体：${NC}"
echo "中文字体："
fc-list :lang=zh | grep -i "simsun\|宋体\|noto\|source" | head -5
echo ""
echo "Times New Roman 或替代字体："
fc-list | grep -i "times\|liberation\|tex.*gyre.*termes" | head -5

echo ""
echo -e "${GREEN}字体安装完成！${NC}"

