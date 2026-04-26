#!/usr/bin/env bash
set -e

INPUT_TEXT="${1:-}"

echo "============================================================"
echo "  Banking Intent Classification — Inference"
echo "============================================================"

if [ -n "$INPUT_TEXT" ]; then
    echo "  Mode: Single input"
    echo ""
    python scripts/inference.py --config configs/inference.yaml --text "$INPUT_TEXT"
else
    echo "  Mode: Demo (nhiều ví dụ)"
    echo ""
    python scripts/inference.py --config configs/inference.yaml
fi
