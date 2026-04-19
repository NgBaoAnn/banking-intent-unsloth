"""
train.py
--------
Fine-tune mô hình ngôn ngữ (Qwen2.5 / LLaMA3) bằng Unsloth + LoRA (PEFT)
cho bài toán phân loại ý định ngân hàng (Banking Intent Classification).
Sau khi huấn luyện, đánh giá accuracy + classification report trên tập test.
"""

import os
import yaml
import argparse
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from sklearn.metrics import classification_report, accuracy_score

from utils import format_prompt


# ---------------------------------------------------------------------------
# Evaluation on test set  (Yêu cầu 1: "Đánh giá hiệu suất trên tập test")
# ---------------------------------------------------------------------------
def evaluate_on_test(model, tokenizer, test_csv_path: str, output_dir: str):
    """Chạy inference trên tập test, in accuracy + classification report."""
    print("\n" + "=" * 60)
    print("  ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print("=" * 60)

    FastLanguageModel.for_inference(model)
    test_df = pd.read_csv(test_csv_path)

    y_true = []
    y_pred = []

    for idx, row in test_df.iterrows():
        prompt = format_prompt(text=row["text"])  # Không có label → inference mode
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(
            outputs[:, input_length:], skip_special_tokens=True
        )[0]
        predicted = decoded.strip().split("\n")[0].strip()
        expected = str(row["label_text"]).strip()

        y_true.append(expected)
        y_pred.append(predicted)

        # In một vài ví dụ đầu
        if idx < 5:
            status = "✓" if predicted == expected else "✗"
            print(f"  [{status}] Input : {row['text'][:80]}...")
            print(f"       Expected : {expected}")
            print(f"       Predicted: {predicted}\n")

        # Progress
        if (idx + 1) % 50 == 0:
            print(f"  ... đã đánh giá {idx + 1}/{len(test_df)} mẫu")

    # --- Metrics ---
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "-" * 60)
    print(f"  ACCURACY: {acc:.4f} ({sum(1 for a, b in zip(y_true, y_pred) if a == b)}/{len(y_true)})")
    print("-" * 60)

    # Classification report (precision, recall, F1 mỗi class)
    # Chỉ report các label thực sự xuất hiện trong predictions
    unique_labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(
        y_true, y_pred,
        labels=unique_labels,
        zero_division=0
    )
    print("\n  CLASSIFICATION REPORT:")
    print(report)

    # --- Lưu kết quả ra file ---
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "evaluation_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"  → Kết quả đã lưu tại: {result_path}")

    return acc


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune model với Unsloth")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    args = parser.parse_args()

    # --- Load config ----------------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config.get("model_name", "unsloth/Qwen2.5-1.5B")
    max_seq_length = config.get("max_seq_length", 512)
    dataset_path = config.get("dataset_path", "./sample_data")
    output_dir = config.get("output_dir", "./outputs/intent-model")

    # --- Print hyperparameters (Yêu cầu 2.2: ghi tài liệu rõ ràng) ----------
    print("=" * 60)
    print("  HYPERPARAMETERS")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key:<30s}: {value}")
    print("=" * 60)

    # --- Load model -----------------------------------------------------------
    print(f"\n[1/5] Đang tải mô hình: {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,       # Auto detect: Float16 cho T4/V100, BFloat16 cho Ampere+
        load_in_4bit=True,
    )

    # Đảm bảo tokenizer có pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Apply LoRA adapter ---------------------------------------------------
    print("[2/5] Cấu hình LoRA adapter...")
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

    # --- Load dataset & format prompts ----------------------------------------
    print("[3/5] Nạp dữ liệu huấn luyện...")
    train_df = pd.read_csv(f"{dataset_path}/train.csv")

    # Format prompt trực tiếp với EOS token thật từ tokenizer (không dùng placeholder)
    eos = tokenizer.eos_token or ""
    train_df["formatted_text"] = train_df.apply(
        lambda row: format_prompt(
            text=row["text"],
            label=row["label_text"],
            eos_token=eos,
        ),
        axis=1,
    )

    dataset = Dataset.from_pandas(train_df[["formatted_text"]])
    print(f"  → Số mẫu train: {len(dataset)}")

    # --- SFTTrainer -----------------------------------------------------------
    print("[4/5] Bắt đầu huấn luyện...\n")
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

    trainer.train()

    # --- Save checkpoint  (Yêu cầu 2.2) --------------------------------------
    print(f"\n[5/5] Huấn luyện hoàn tất! Lưu checkpoint → {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # --- Evaluate  (Yêu cầu 1: đánh giá trên tập test) -----------------------
    test_csv = f"{dataset_path}/test.csv"
    evaluate_on_test(model, tokenizer, test_csv, output_dir)

    print("\n✓ Pipeline hoàn tất!")


if __name__ == "__main__":
    main()
