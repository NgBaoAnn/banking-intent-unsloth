"""
train.py
--------
Fine-tune mô hình ngôn ngữ bằng Unsloth + LoRA (PEFT)
cho bài toán phân loại ý định ngân hàng.
"""

import yaml
import argparse
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments

from utils import format_prompt


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model với Unsloth")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config.get("model_name", "unsloth/Qwen2.5-1.5B")
    max_seq_length = config.get("max_seq_length", 512)
    dataset_path = config.get("dataset_path", "./sample_data")
    output_dir = config.get("output_dir", "./outputs/intent-model")

    # Print hyperparameters
    print("=" * 60)
    print("  HYPERPARAMETERS")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key:<30s}: {value}")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.get("lora_r", 16),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Load and format dataset
    train_df = pd.read_csv(f"{dataset_path}/train.csv")
    eos = tokenizer.eos_token or ""
    train_df["formatted_text"] = train_df.apply(
        lambda row: format_prompt(text=row["text"], label=row["label_text"], eos_token=eos),
        axis=1,
    )
    dataset = Dataset.from_pandas(train_df[["formatted_text"]])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="formatted_text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=config.get("batch_size", 2),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            warmup_steps=config.get("warmup_steps", 10),
            num_train_epochs=config.get("num_train_epochs", 3),
            learning_rate=config.get("learning_rate", 2e-4),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim=config.get("optimizer", "adamw_8bit"),
            weight_decay=config.get("weight_decay", 0.01),
            lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
            seed=3407,
            output_dir="outputs",
            report_to="none",
        ),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving checkpoint to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
