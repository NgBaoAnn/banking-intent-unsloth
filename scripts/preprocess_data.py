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


# ---------------------------------------------------------------------------
# Text normalization & basic cleaning  (Yêu cầu 2.1)
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản: lowercase, loại bỏ ký tự thừa, chuẩn hóa khoảng trắng."""
    text = text.strip()
    text = text.lower()
    # Loại bỏ ký tự đặc biệt không cần thiết (giữ lại chữ, số, dấu câu cơ bản)
    text = re.sub(r"[^\w\s\.\,\?\!\'\-]", "", text)
    # Chuẩn hóa nhiều dấu cách thành 1
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu BANKING77")
    parser.add_argument(
        "--samples_per_class_train", type=int, default=30,
        help="Số mẫu mỗi lớp cho tập train (mặc định: 30)"
    )
    parser.add_argument(
        "--samples_per_class_valid", type=int, default=10,
        help="Số mẫu mỗi lớp cho tập validation (mặc định: 10)"
    )
    parser.add_argument(
        "--samples_per_class_test", type=int, default=15,
        help="Số mẫu mỗi lớp cho tập test (mặc định: 15)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="sample_data",
        help="Thư mục lưu kết quả (mặc định: sample_data)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed cho reproducibility (mặc định: 42)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  BANKING77 — Tiền xử lý dữ liệu")
    print("=" * 60)

    # 1. Tải dataset -----------------------------------------------------------
    print("\n[1/4] Đang tải tập dữ liệu mteb/banking77 từ HuggingFace...")
    dataset = load_dataset("mteb/banking77")

    df_train_full = dataset["train"].to_pandas()
    df_test_full = dataset["test"].to_pandas()
    print(f"  → Tập train gốc : {len(df_train_full)} mẫu")
    print(f"  → Tập test  gốc : {len(df_test_full)} mẫu")
    print(f"  → Số lớp (labels): {df_train_full['label_text'].nunique()}")

    # 2. Sampling --------------------------------------------------------------
    print(f"\n[2/4] Trích xuất mẫu: {args.samples_per_class_train}/class (train), "
          f"{args.samples_per_class_valid}/class (valid), "
          f"{args.samples_per_class_test}/class (test)...")

    # Lấy mẫu từ tập train gốc cho cả Train và Valid
    def split_train_valid(x):
        total_needed = args.samples_per_class_train + args.samples_per_class_valid
        # Lấy đủ mẫu nếu có, hoặc lấy vửa đủ
        sampled = x.sample(min(len(x), total_needed), random_state=args.seed)
        
        valid_size = min(len(sampled), args.samples_per_class_valid)
        train_size = len(sampled) - valid_size
        
        valid_set = sampled.iloc[:valid_size]
        train_set = sampled.iloc[valid_size:]
        return train_set, valid_set

    train_valid_splits = df_train_full.groupby("label_text", group_keys=False).apply(split_train_valid)
    
    # Kết hợp lại các tập đã split
    sample_train = pd.concat([res[0] for res in train_valid_splits]).reset_index(drop=True)
    sample_valid = pd.concat([res[1] for res in train_valid_splits]).reset_index(drop=True)

    # Lấy mẫu test
    sample_test = (
        df_test_full
        .groupby("label_text", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), args.samples_per_class_test),
                                   random_state=args.seed))
        .reset_index(drop=True)
    )
    
    print(f"  → Tập train sau sampling: {len(sample_train)} mẫu")
    print(f"  → Tập valid sau sampling: {len(sample_valid)} mẫu")
    print(f"  → Tập test  sau sampling: {len(sample_test)} mẫu")

    # 3. Text normalization ----------------------------------------------------
    print("\n[3/4] Chuẩn hóa văn bản...")
    sample_train["text"] = sample_train["text"].apply(normalize_text)
    sample_valid["text"] = sample_valid["text"].apply(normalize_text)
    sample_test["text"] = sample_test["text"].apply(normalize_text)

    # Tạo label mapping (label_text → label_id)
    all_labels = sorted(
        set(sample_train["label_text"].unique()) | set(sample_test["label_text"].unique())
    )
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    sample_train["label_id"] = sample_train["label_text"].map(label2id)
    sample_valid["label_id"] = sample_valid["label_text"].map(label2id)
    sample_test["label_id"] = sample_test["label_text"].map(label2id)

    # 4. Lưu file (CHỈ lưu text + label, KHÔNG lưu formatted prompt) ----------
    # Prompt sẽ được format trực tiếp trong train.py với EOS token chính xác
    print(f"\n[4/4] Lưu kết quả vào thư mục '{args.output_dir}'...")
    os.makedirs(args.output_dir, exist_ok=True)

    cols_to_save = ["text", "label", "label_text", "label_id"]
    sample_train[cols_to_save].to_csv(
        os.path.join(args.output_dir, "train.csv"), index=False
    )
    sample_valid[cols_to_save].to_csv(
        os.path.join(args.output_dir, "valid.csv"), index=False
    )
    sample_test[cols_to_save].to_csv(
        os.path.join(args.output_dir, "test.csv"), index=False
    )

    # Lưu label mapping
    label_map_df = pd.DataFrame(list(label2id.items()), columns=["label_text", "label_id"])
    label_map_df.to_csv(os.path.join(args.output_dir, "label_map.csv"), index=False)

    print(f"  ✓ train.csv     : {len(sample_train)} mẫu")
    print(f"  ✓ valid.csv     : {len(sample_valid)} mẫu")
    print(f"  ✓ test.csv      : {len(sample_test)} mẫu")
    print(f"  ✓ label_map.csv : {len(label2id)} nhãn")
    print("\nHoàn tất tiền xử lý dữ liệu!")


if __name__ == "__main__":
    main()
