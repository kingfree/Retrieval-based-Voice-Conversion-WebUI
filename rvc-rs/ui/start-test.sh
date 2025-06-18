#!/bin/bash

# RVC-RS UI 修复验证启动脚本
# 用于快速启动和测试修复的功能

echo "🚀 RVC-RS UI 修复验证启动脚本"
echo "=================================="

# 检查当前目录
CURRENT_DIR=$(pwd)
UI_DIR="Retrieval-based-Voice-Conversion-WebUI/rvc-rs/ui"

# 导航到正确的目录
if [[ $CURRENT_DIR == *"$UI_DIR" ]]; then
    echo "✅ 当前已在UI目录中"
elif [ -d "$UI_DIR" ]; then
    echo "📂 导航到UI目录..."
    cd "$UI_DIR"
else
    echo "❌ 错误: 找不到UI目录 $UI_DIR"
    echo "请确保在项目根目录中运行此脚本"
    exit 1
fi

# 检查Node.js是否安装
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 未安装或不在PATH中"
    echo "请安装Node.js后重试"
    exit 1
fi

# 检查npm是否安装
if ! command -v npm &> /dev/null; then
    echo "❌ npm 未安装或不在PATH中"
    echo "请安装npm后重试"
    exit 1
fi

echo "✅ Node.js版本: $(node --version)"
echo "✅ npm版本: $(npm --version)"

# 检查package.json是否存在
if [ ! -f "package.json" ]; then
    echo "❌ package.json 文件不存在"
    echo "请确保在正确的UI目录中"
    exit 1
fi

echo "✅ package.json 文件存在"

# 安装依赖
echo ""
echo "📦 检查并安装依赖..."
if [ ! -d "node_modules" ]; then
    echo "🔄 首次安装依赖..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ 依赖安装失败"
        exit 1
    fi
else
    echo "✅ node_modules 目录存在，跳过安装"
fi

# 运行修复验证测试
echo ""
echo "🧪 运行修复验证测试..."
if [ -f "test-fixes.js" ]; then
    node test-fixes.js
    echo ""
else
    echo "⚠️ 测试文件 test-fixes.js 不存在，跳过测试"
fi

# 检查Tauri CLI
echo "🔧 检查Tauri环境..."
if command -v cargo &> /dev/null; then
    echo "✅ Rust/Cargo 已安装"

    # 检查Tauri CLI
    if cargo tauri --version &> /dev/null; then
        echo "✅ Tauri CLI 已安装"
        TAURI_AVAILABLE=true
    else
        echo "⚠️ Tauri CLI 未安装，将仅启动Web版本"
        TAURI_AVAILABLE=false
    fi
else
    echo "⚠️ Rust/Cargo 未安装，将仅启动Web版本"
    TAURI_AVAILABLE=false
fi

# 提供启动选项
echo ""
echo "🎯 启动选项:"
echo "1. 开发服务器 (Web版本) - npm run dev"
if [ "$TAURI_AVAILABLE" = true ]; then
    echo "2. Tauri开发版本 - npm run tauri dev"
    echo "3. 构建Tauri应用 - npm run tauri build"
fi
echo "4. 退出"

echo ""
read -p "请选择启动方式 (1-4): " choice

case $choice in
    1)
        echo "🌐 启动Web开发服务器..."
        echo "⚠️ 注意: Web版本无法使用Tauri特定功能（如文件系统访问）"
        echo "🔗 将在 http://localhost:5173 启动"
        echo ""
        npm run dev
        ;;
    2)
        if [ "$TAURI_AVAILABLE" = true ]; then
            echo "🖥️ 启动Tauri开发版本..."
            echo "📱 这将启动一个原生桌面应用"
            echo ""
            npm run tauri dev
        else
            echo "❌ Tauri不可用，请安装Rust和Tauri CLI"
            echo "安装命令:"
            echo "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
            echo "cargo install @tauri-apps/cli"
        fi
        ;;
    3)
        if [ "$TAURI_AVAILABLE" = true ]; then
            echo "🏗️ 构建Tauri应用..."
            echo "这将创建生产版本的桌面应用"
            echo ""
            npm run tauri build
        else
            echo "❌ Tauri不可用，请安装Rust和Tauri CLI"
        fi
        ;;
    4)
        echo "👋 退出脚本"
        exit 0
        ;;
    *)
        echo "❌ 无效选择，启动Web开发服务器..."
        npm run dev
        ;;
esac

# 显示修复说明
echo ""
echo "📋 修复说明:"
echo "1. ✅ 修复了 stopAudioStream 未定义错误"
echo "2. ✅ 优化了800x600窗口显示"
echo "3. ✅ 统一了控件样式"
echo "4. ✅ 改进了错误处理"
echo "5. ✅ 添加了启动诊断"
echo ""
echo "📖 更多信息请查看 FIXES.md 文件"
echo ""
echo "🎉 享受使用修复后的RVC-RS UI！"
