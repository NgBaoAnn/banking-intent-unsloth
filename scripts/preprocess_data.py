import os
import re
import argparse
import pandas as pd
from datasets import load_dataset


def normalize_text(text: str) -> str:
    """Lowercase, remove special characters, normalize whitespace."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s\.\,\?\!\'\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def main():
    parser = argparse.ArgumentParser(description="Preprocess BANKING77 dataset")
    parser.add_argument("--samples_per_class_train", type=int, default=80)
    parser.add_argument("--samples_per_class_valid", type=int, default=10)
    parser.add_argument("--samples_per_class_test", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="sample_data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  BANKING77 — Preprocessing")
    print("=" * 60)

    print("\n[1/4] Loading mteb/banking77 from HuggingFace...")
    dataset = load_dataset("mteb/banking77")

    df_train_full = dataset["train"].to_pandas()
    df_test_full = dataset["test"].to_pandas()
    print(f"  → Train: {len(df_train_full)} samples, Test: {len(df_test_full)} samples")
    print(f"  → Classes: {df_train_full['label_text'].nunique()}")

    print(f"\n[2/4] Sampling {args.samples_per_class_train}/class (train), "
          f"{args.samples_per_class_valid}/class (valid), "
          f"{args.samples_per_class_test}/class (test)...")

    def split_train_valid(x):
        total_needed = args.samples_per_class_train + args.samples_per_class_valid
        sampled = x.sample(min(len(x), total_needed), random_state=args.seed)
        valid_size = min(len(sampled), args.samples_per_class_valid)
        return sampled.iloc[valid_size:], sampled.iloc[:valid_size]

    splits = df_train_full.groupby("label_text", group_keys=False).apply(split_train_valid)
    sample_train = pd.concat([r[0] for r in splits]).reset_index(drop=True)
    sample_valid = pd.concat([r[1] for r in splits]).reset_index(drop=True)
    sample_test = (
        df_test_full
        .groupby("label_text", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), args.samples_per_class_test), random_state=args.seed))
        .reset_index(drop=True)
    )

    print(f"  → Train: {len(sample_train)}, Valid: {len(sample_valid)}, Test: {len(sample_test)}")

    print("\n[3/4] Normalizing text...")
    for df in [sample_train, sample_valid, sample_test]:
        df["text"] = df["text"].apply(normalize_text)

    all_labels = sorted(set(sample_train["label_text"].unique()) | set(sample_test["label_text"].unique()))
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    for df in [sample_train, sample_valid, sample_test]:
        df["label_id"] = df["label_text"].map(label2id)

    print(f"\n[4/4] Saving to '{args.output_dir}'...")
    os.makedirs(args.output_dir, exist_ok=True)

    cols = ["text", "label", "label_text", "label_id"]
    sample_train[cols].to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    sample_valid[cols].to_csv(os.path.join(args.output_dir, "valid.csv"), index=False)
    sample_test[cols].to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

    label_map_df = pd.DataFrame(list(label2id.items()), columns=["label_text", "label_id"])
    label_map_df.to_csv(os.path.join(args.output_dir, "label_map.csv"), index=False)

    print(f"  ✓ train.csv ({len(sample_train)}), valid.csv ({len(sample_valid)}), "
          f"test.csv ({len(sample_test)}), label_map.csv ({len(label2id)} labels)")
    print("\nDone!")


if __name__ == "__main__":
    main()
