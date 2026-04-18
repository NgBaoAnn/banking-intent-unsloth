"""
train.py
--------
Fine-tune mô hình ngôn ngữ bằng Unsloth + LoRA (PEFT)
cho bài toán phân loại ý định ngân hàng.
"""

import yaml
import argparse
from unsloth import FastLanguageModel


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model với Unsloth")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config.get("model_name", "unsloth/Qwen2.5-1.5B")
    max_seq_length = config.get("max_seq_length", 512)

    print(f"Loading model: {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print("Model loaded successfully.")


if __name__ == "__main__":
    main()
