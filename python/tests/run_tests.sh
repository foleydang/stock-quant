#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PYTHON_DIR"
python3 -m pytest tests/ -v