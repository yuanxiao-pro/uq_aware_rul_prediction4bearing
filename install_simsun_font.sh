#!/bin/bash
# 安装 Windows 宋体（SimSun）字体脚本

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}安装 Windows 宋体（SimSun）字体${NC}"
echo -e "${GREEN}========================================${NC}"

# 创建用户字体目录
FONT_DIR="$HOME/.local/share/fonts"
mkdir -p "$FONT_DIR"

# 检查是否已存在
if [ -f "$FONT_DIR/simsun.ttc" ] || [ -f "$FONT_DIR/SimSun.ttf" ] || [ -f "$FONT_DIR/simsun.ttf" ]; then
    echo -e "${GREEN}✓ 宋体字体已存在${NC}"
    if [ -f "$FONT_DIR/simsun.ttc" ]; then
        echo "  文件: $FONT_DIR/simsun.ttc"
    elif [ -f "$FONT_DIR/SimSun.ttf" ]; then
        echo "  文件: $FONT_DIR/SimSun.ttf"
    else
        echo "  文件: $FONT_DIR/simsun.ttf"
    fi
    echo ""
    echo "刷新字体缓存..."
    fc-cache -fv > /dev/null 2>&1
    echo -e "${GREEN}✓ 字体已就绪${NC}"
    exit 0
fi

echo -e "${YELLOW}未找到宋体字体文件，请选择安装方式：${NC}"
echo ""
echo "1. 从 Windows 系统复制（推荐）"
echo "2. 从网络下载（需要合法授权）"
echo "3. 手动指定字体文件路径"
echo ""
read -p "请选择 (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo -e "${BLUE}方法1：从 Windows 系统复制${NC}"
        echo "请按照以下步骤操作："
        echo "1. 在 Windows 系统中找到字体文件："
        echo "   C:\\Windows\\Fonts\\simsun.ttc"
        echo ""
        echo "2. 将 simsun.ttc 文件复制到 Linux 系统"
        echo ""
        echo "3. 使用以下命令复制到字体目录："
        echo "   cp /path/to/simsun.ttc $FONT_DIR/"
        echo ""
        echo "或者使用 scp 从远程 Windows 系统复制："
        echo "   scp user@windows_host:/mnt/c/Windows/Fonts/simsun.ttc $FONT_DIR/"
        echo ""
        read -p "请输入 simsun.ttc 文件的完整路径: " font_path
        if [ -f "$font_path" ]; then
            cp "$font_path" "$FONT_DIR/simsun.ttc"
            echo -e "${GREEN}✓ 字体文件已复制到 $FONT_DIR/simsun.ttc${NC}"
        else
            echo -e "${RED}✗ 文件不存在: $font_path${NC}"
            exit 1
        fi
        ;;
    2)
        echo ""
        echo -e "${BLUE}方法2：从网络下载${NC}"
        echo -e "${YELLOW}注意：请确保您有合法授权下载和使用 Windows 字体${NC}"
        echo ""
        echo "可以使用以下方式下载："
        echo "1. 从 Microsoft 官方获取（如果有授权）"
        echo "2. 从合法字体分发网站下载"
        echo ""
        read -p "请输入下载的字体文件路径: " font_path
        if [ -f "$font_path" ]; then
            cp "$font_path" "$FONT_DIR/simsun.ttc"
            echo -e "${GREEN}✓ 字体文件已复制到 $FONT_DIR/simsun.ttc${NC}"
        else
            echo -e "${RED}✗ 文件不存在: $font_path${NC}"
            exit 1
        fi
        ;;
    3)
        echo ""
        echo -e "${BLUE}方法3：手动指定字体文件路径${NC}"
        read -p "请输入字体文件的完整路径: " font_path
        if [ -f "$font_path" ]; then
            # 检查文件类型（如果 file 命令可用）
            if command -v file >/dev/null 2>&1; then
                file_type=$(file -b "$font_path" 2>/dev/null)
                if [[ "$file_type" == *"TrueType"* ]] || [[ "$file_type" == *"OpenType"* ]] || [[ "$file_type" == *"font"* ]]; then
                    cp "$font_path" "$FONT_DIR/simsun.ttc"
                    echo -e "${GREEN}✓ 字体文件已复制到 $FONT_DIR/simsun.ttc${NC}"
                else
                    echo -e "${YELLOW}⚠️  警告: 文件可能不是有效的字体文件${NC}"
                    read -p "是否继续? (y/n): " confirm
                    if [[ "$confirm" =~ ^[Yy]$ ]]; then
                        cp "$font_path" "$FONT_DIR/simsun.ttc"
                        echo -e "${GREEN}✓ 字体文件已复制${NC}"
                    else
                        exit 1
                    fi
                fi
            else
                # 如果 file 命令不可用，直接复制（根据文件扩展名判断）
                if [[ "$font_path" =~ \.(ttf|ttc|otf)$ ]]; then
                    cp "$font_path" "$FONT_DIR/simsun.ttc"
                    echo -e "${GREEN}✓ 字体文件已复制到 $FONT_DIR/simsun.ttc${NC}"
                else
                    echo -e "${YELLOW}⚠️  警告: 文件扩展名不是常见的字体格式 (.ttf, .ttc, .otf)${NC}"
                    read -p "是否继续? (y/n): " confirm
                    if [[ "$confirm" =~ ^[Yy]$ ]]; then
                        cp "$font_path" "$FONT_DIR/simsun.ttc"
                        echo -e "${GREEN}✓ 字体文件已复制${NC}"
                    else
                        exit 1
                    fi
                fi
            fi
        else
            echo -e "${RED}✗ 文件不存在: $font_path${NC}"
            exit 1
        fi
        ;;
    *)
        echo -e "${RED}✗ 无效的选择${NC}"
        exit 1
        ;;
esac

# 刷新字体缓存
echo ""
echo "刷新字体缓存..."
fc-cache -fv > /dev/null 2>&1

# 验证安装
echo ""
echo "验证字体安装..."
if fc-list | grep -qi "simsun\|宋体"; then
    echo -e "${GREEN}✓ 宋体字体安装成功！${NC}"
    fc-list | grep -i "simsun\|宋体" | head -3
else
    echo -e "${YELLOW}⚠️  字体文件已复制，但可能需要重启应用程序才能生效${NC}"
    echo "可以尝试运行: fc-cache -fv"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}安装完成！${NC}"
echo -e "${GREEN}========================================${NC}"

