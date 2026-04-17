"""
preprocess_data.py
------------------
Tải tập dữ liệu BANKING77 từ HuggingFace, lấy mẫu con (subset),
thực hiện tiền xử lý văn bản, và chia thành tập train/test dưới dạng CSV.
"""

import os
import re
import argparse
import pandas as pd
from datasets import load_dataset


def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản: lowercase, loại bỏ ký tự thừa, chuẩn hóa khoảng trắng."""
    text = text.strip()
    text = text.lower()
    text = re.sub(r"[^\w\s\.\,\?\!\'\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def main():
    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu BANKING77")
    parser.add_argument("--samples_per_class_train", type=int, default=20)
    parser.add_argument("--samples_per_class_test", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading BANKING77 dataset...")
    dataset = load_dataset("mteb/banking77")

    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()

    # Stratified sampling
    sample_train = (
        df_train.groupby("label_text", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), args.samples_per_class_train), random_state=args.seed))
        .reset_index(drop=True)
    )
    sample_test = (
        df_test.groupby("label_text", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), args.samples_per_class_test), random_state=args.seed))
        .reset_index(drop=True)
    )

    # Text normalization
    sample_train["text"] = sample_train["text"].apply(normalize_text)
    sample_test["text"] = sample_test["text"].apply(normalize_text)

    print(f"Sampled train: {len(sample_train)}, test: {len(sample_test)}")


if __name__ == "__main__":
    main()
