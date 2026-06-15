#!/bin/bash
# LSTM v9 环境确认脚本
# 检查依赖安装情况, 缺失则安装
set -e

echo "🔍 检查 v9 依赖..."

# 检测 Python
PYTHON=$(which python3 || which python)
echo "   Python: $($PYTHON --version)"

# 检测虚拟环境
if [ -n "$VIRTUAL_ENV" ]; then
    echo "   venv: $VIRTUAL_ENV"
else
    echo "   ⚠️ 未激活虚拟环境, 建议: source .venv/bin/activate"
fi

# 检查 PyTorch
$PYTHON -c "
import torch
print(f'   PyTorch: {torch.__version__}')
if torch.backends.mps.is_available():
    print('   🚀 MPS (Apple GPU) 可用')
elif torch.cuda.is_available():
    print('   🚀 CUDA 可用')
else:
    print('   ⚠️ 仅 CPU 模式 (M4 Pro 上 MPS 可用, 请确认 PyTorch >= 2.0)')
" 2>/dev/null || {
    echo "   ❌ PyTorch 未安装, 正在安装..."
    $PYTHON -m pip install torch --quiet
    echo "   ✅ PyTorch 安装完成"
}

# 检查其他关键包
for pkg in numpy pandas lightgbm scikit-learn scipy akshare joblib; do
    $PYTHON -c "import $pkg" 2>/dev/null || {
        echo "   ❌ $pkg 未安装"
        MISSING="$MISSING $pkg"
    }
done

if [ -n "$MISSING" ]; then
    echo "   正在安装缺失包: $MISSING"
    $PYTHON -m pip install -r "$(dirname "$0")/../requirements.txt" --quiet
    echo "   ✅ 安装完成"
fi

echo ""
echo "✅ 所有依赖就绪, 可以开始训练:"
echo ""
echo "   # 1. 训练 LSTM 编码器 (30-60分钟)"
echo "   python strategy/lstm_encoder.py"
echo ""
echo "   # 2. 训练 LGBM (含 LSTM 特征)"
echo "   python strategy/train.py --model daily"
echo ""