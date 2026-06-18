#!/bin/bash
# 并行实验运行脚本 (macOS + Linux 兼容)
# 用法: bash run_experiments.sh [--quick]
#   --quick: 快速模式 (csi300)
#   --parallel N: 并行跑 N 个任务 (默认串行)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/experiments"
mkdir -p "$RESULTS_DIR"

QUICK=""
PARALLEL=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK="--quick"; shift ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        *) shift ;;
    esac
done

MODELS=("LightGBM")
HORIZONS=(1 3 5)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/results_${TIMESTAMP}.csv"
echo "model,horizon,n_feat,ic,rank_ic,icir,rank_icir,train_s,sharpe,ret" > "$RESULT_FILE"

echo "============================================"
echo " 并行实验"
echo "  模型: ${MODELS[*]}"
echo "  预测周期: ${HORIZONS[*]}"
echo "  快速模式: ${QUICK:-否}"
echo "  结果: $RESULT_FILE"
echo "============================================"

# 激活虚拟环境
if [ -f "$PROJECT_DIR/../.venv/bin/activate" ]; then
    source "$PROJECT_DIR/../.venv/bin/activate"
elif [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

cd "$PROJECT_DIR"

# 用 Python 解析日志 (macOS grep 不支持 -P)
parse_log() {
    python -c "
import re, sys
with open('$1') as f:
    text = f.read()
m = re.findall(r\"'IC':\s*np\.float64\(([^)]+)\)\", text)
ic = m[-1] if m else '?'
m = re.findall(r\"'Rank IC':\s*np\.float64\(([^)]+)\)\", text)
rank_ic = m[-1] if m else '?'
m = re.findall(r\"'ICIR':\s*np\.float64\(([^)]+)\)\", text)
icir = m[-1] if m else '?'
m = re.findall(r\"'Rank ICIR':\s*np\.float64\(([^)]+)\)\", text)
rank_icir = m[-1] if m else '?'
m = re.findall(r'训练耗时:\s*(\d+)', text)
train_s = m[-1] if m else '?'
m = re.findall(r'信号夏普:\s*([-\d.]+)', text)
sharpe = m[-1] if m else '?'
m = re.findall(r'累计复合:\s*([-\d.]+)', text)
ret = m[-1] if m else '?'
m = re.findall(r'特征:\s*(\d+)', text)
n_feat = m[-1] if m else '?'
print(f'{n_feat},{ic},{rank_ic},{icir},{rank_icir},{train_s},{sharpe},{ret}')
"
}

run_exp() {
    local model=$1
    local horizon=$2
    local log_file="$RESULTS_DIR/${model}_h${horizon}_${TIMESTAMP}.log"
    
    echo ""
    echo "▶ [$model h=$horizon] 开始... ($(date +%H:%M:%S))"
    
    python qlib_pipeline/train.py --model "$model" --horizon "$horizon" $QUICK --quiet \
        > "$log_file" 2>&1 || true
    
    local parsed=$(parse_log "$log_file")
    local n_feat=$(echo "$parsed" | cut -d, -f1)
    local ic=$(echo "$parsed" | cut -d, -f2)
    local rank_ic=$(echo "$parsed" | cut -d, -f3)
    local sharpe=$(echo "$parsed" | cut -d, -f8)
    local train_s=$(echo "$parsed" | cut -d, -f7)
    
    echo "$model,$horizon,$parsed" >> "$RESULT_FILE"
    echo "  ✅ [$model h=$horizon] IC=$ic RankIC=$rank_ic 夏普=$sharpe (${train_s}s)"
}

if [ "$PARALLEL" -gt 1 ]; then
    echo "⚠️ 并行模式需要每个进程独立内存, 注意 OOM"
    for model in "${MODELS[@]}"; do
        for h in "${HORIZONS[@]}"; do
            run_exp "$model" "$h" &
            while [ $(jobs -r | wc -l) -ge "$PARALLEL" ]; do
                sleep 5
            done
        done
    done
    wait
else
    for model in "${MODELS[@]}"; do
        for h in "${HORIZONS[@]}"; do
            run_exp "$model" "$h"
        done
    done
fi

echo ""
echo "============================================"
echo " 实验结果汇总"
echo "============================================"
column -t -s, "$RESULT_FILE"
echo ""
echo "结果: $RESULT_FILE"