#!/usr/bin/env bash
set -e  # Dừng lại ngay nếu có lỗi

# Nếu có đối số dòng lệnh → chạy 1 câu duy nhất
# Nếu không có → chạy chế độ demo nhiều ví dụ
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
