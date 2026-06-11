#!/bin/bash
# 每30分钟累积实时K线

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PYTHON_DIR"
python3 data/kline_accumulator.py >> logs/kline_accumulate.log 2>&1