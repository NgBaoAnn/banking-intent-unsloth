"""
preprocess_data.py
------------------
Tải tập dữ liệu BANKING77 từ HuggingFace, lấy mẫu con (subset),
thực hiện tiền xử lý văn bản, và chia thành tập train/test dưới dạng CSV.
"""

import os
import re
from datasets import load_dataset


def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản: lowercase, loại bỏ ký tự thừa, chuẩn hóa khoảng trắng."""
    text = text.strip()
    text = text.lower()
    text = re.sub(r"[^\w\s\.\,\?\!\'\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def main():
    print("Loading BANKING77 dataset...")
    dataset = load_dataset("mteb/banking77")

    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()

    print(f"Train: {len(df_train)} samples, Test: {len(df_test)} samples")

    # Apply text normalization
    df_train["text"] = df_train["text"].apply(normalize_text)
    df_test["text"] = df_test["text"].apply(normalize_text)

    print("Text normalization applied.")


if __name__ == "__main__":
    main()
