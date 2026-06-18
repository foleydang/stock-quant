#!/bin/bash
# 并行实验运行脚本
# 用法: bash run_experiments.sh [--quick]
#   --quick: 快速模式 (csi300, 50只股票)
#   --parallel N: 并行跑 N 个任务 (默认串行)
#
# 输出: experiments/results.csv

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

# 实验配置
MODELS=("LightGBM")
HORIZONS=(1 3 5)
# 如果 GPU 可用, 可以加: GRU LSTM

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/results_${TIMESTAMP}.csv"
echo "model,horizon,n_feat,ic,rank_ic,icir,rank_icir,train_s,sharpe,ret" > "$RESULT_FILE"

echo "============================================"
echo " 🧪 并行实验运行"
echo "  模型: ${MODELS[*]}"
echo "  预测周期: ${HORIZONS[*]}"
echo "  快速模式: ${QUICK:-否}"
echo "  并行数: $PARALLEL"
echo "  结果: $RESULT_FILE"
echo "============================================"

# 激活虚拟环境
if [ -f "$PROJECT_DIR/../.venv/bin/activate" ]; then
    source "$PROJECT_DIR/../.venv/bin/activate"
elif [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

cd "$PROJECT_DIR"

run_exp() {
    local model=$1
    local horizon=$2
    local log_file="$RESULTS_DIR/${model}_h${horizon}_${TIMESTAMP}.log"
    
    echo ""
    echo "▶ [$model h=$horizon] 开始... ($(date +%H:%M:%S))"
    
    python qlib_pipeline/train.py --model "$model" --horizon "$horizon" $QUICK --quiet \
        > "$log_file" 2>&1
    
    # 解析结果
    local ic=$(grep "'IC':" "$log_file" | tail -1 | grep -oP "np\.float64\([^)]+\)" | head -1 | grep -oP '[-0-9.]+' | head -1)
    local rank_ic=$(grep "'Rank IC':" "$log_file" | tail -1 | grep -oP "np\.float64\([^)]+\)" | head -1 | grep -oP '[-0-9.]+' | head -1)
    local icir=$(grep "'ICIR':" "$log_file" | tail -1 | grep -oP "np\.float64\([^)]+\)" | head -1 | grep -oP '[-0-9.]+' | head -1)
    local rank_icir=$(grep "'Rank ICIR':" "$log_file" | tail -1 | grep -oP "np\.float64\([^)]+\)" | head -1 | grep -oP '[-0-9.]+' | head -1)
    local train_s=$(grep "训练耗时:" "$log_file" | tail -1 | grep -oP '[0-9]+')
    local sharpe=$(grep "信号夏普:" "$log_file" | tail -1 | grep -oP '[-0-9.]+')
    local ret=$(grep "累计复合:" "$log_file" | tail -1 | grep -oP '[-0-9.]+')
    local n_feat=$(grep "| 📊 特征:" "$log_file" | tail -1 | grep -oP '[0-9]+')
    
    echo "$model,$horizon,${n_feat:-?},${ic:-?},${rank_ic:-?},${icir:-?},${rank_icir:-?},${train_s:-?},${sharpe:-?},${ret:-?}" >> "$RESULT_FILE"
    
    echo "  ✅ [$model h=$horizon] IC=${ic:-?} RankIC=${rank_ic:-?} 夏普=${sharpe:-?} (${train_s:-?}s)"
}

# 运行实验
if [ "$PARALLEL" -gt 1 ]; then
    echo "⚠️ 并行模式需要每个进程独立内存, 注意 OOM"
    for model in "${MODELS[@]}"; do
        for h in "${HORIZONS[@]}"; do
            run_exp "$model" "$h" &
            # 限制并行数
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
echo " 📊 实验结果汇总"
echo "============================================"
column -t -s, "$RESULT_FILE"
echo ""
echo "结果已保存: $RESULT_FILE"