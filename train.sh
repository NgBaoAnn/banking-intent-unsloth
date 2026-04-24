#!/usr/bin/env bash
set -e  # Dừng lại ngay nếu có lỗi

echo "============================================================"
echo "  Banking Intent Classification — Training Pipeline"
echo "============================================================"

# Step 1: Tiền xử lý dữ liệu
echo ""
echo "[Step 1/2] Tiền xử lý dữ liệu BANKING77..."
python scripts/preprocess_data.py

# Step 2: Fine-tuning + Evaluation
echo ""
echo "[Step 2/2] Fine-tuning mô hình với Unsloth..."
python scripts/train.py \
    --config configs/train.yaml \
    --eval_config configs/evaluation.yaml

echo ""
echo "✓ Pipeline hoàn tất!"
