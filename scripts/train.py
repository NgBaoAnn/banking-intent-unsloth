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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.get("lora_r", 16),
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print("LoRA adapter configured.")


if __name__ == "__main__":
    main()
