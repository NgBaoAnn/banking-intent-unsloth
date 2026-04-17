"""
preprocess_data.py
------------------
Tải tập dữ liệu BANKING77 từ HuggingFace, lấy mẫu con (subset),
thực hiện tiền xử lý văn bản, và chia thành tập train/test dưới dạng CSV.
"""

import os
from datasets import load_dataset


def main():
    print("Loading BANKING77 dataset...")
    dataset = load_dataset("mteb/banking77")

    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()

    print(f"Train: {len(df_train)} samples, Test: {len(df_test)} samples")
    print(f"Number of labels: {df_train['label_text'].nunique()}")


if __name__ == "__main__":
    main()
