import os
import yaml
import argparse
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import classification_report, accuracy_score

from utils import format_prompt


def evaluate_on_test(model, tokenizer, eval_config: dict, output_dir: str):
    test_csv_path = eval_config.get("test_dataset", "./sample_data/test.csv")
    results_filename = eval_config.get("output", {}).get("results_file", "evaluation_results.txt")

    print("\n" + "=" * 60)
    print(f"  EVALUATING ON: {test_csv_path}")
    print("=" * 60)

    FastLanguageModel.for_inference(model)
    test_df = pd.read_csv(test_csv_path)

    y_true, y_pred = [], []

    for idx, row in test_df.iterrows():
        prompt = format_prompt(text=row["text"])
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

        if idx < 5:
            status = "✓" if predicted == expected else "✗"
            print(f"  [{status}] Input : {row['text'][:80]}...")
            print(f"       Expected : {expected}")
            print(f"       Predicted: {predicted}\n")

        if (idx + 1) % 50 == 0:
            print(f"  ... evaluated {idx + 1}/{len(test_df)} samples")

    acc = accuracy_score(y_true, y_pred)
    print("\n" + "-" * 60)
    print(f"  ACCURACY: {acc:.4f} ({sum(1 for a, b in zip(y_true, y_pred) if a == b)}/{len(y_true)})")
    print("-" * 60)

    unique_labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=unique_labels, zero_division=0)
    print("\n  CLASSIFICATION REPORT:")
    print(report)

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, results_filename)
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"  → Results saved to: {result_path}")

    return acc


def main():
    parser = argparse.ArgumentParser(description="Fine-tune with Unsloth")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--eval_config", type=str, default=None, help="Path to evaluation config YAML")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config.get("model_name", "unsloth/Qwen2.5-1.5B")
    max_seq_length = config.get("max_seq_length", 512)
    dataset_path = config.get("dataset_path", "./sample_data")
    output_dir = config.get("output_dir", "./outputs/intent-model")

    eval_config = {}
    if args.eval_config:
        with open(args.eval_config, "r") as f:
            eval_config = yaml.safe_load(f)

    print("=" * 60)
    print("  HYPERPARAMETERS")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key:<30s}: {value}")
    print("=" * 60)

    print(f"\n[1/5] Loading model: {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model.config.return_dict = True

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/5] Configuring LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.get("lora_r", 16),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print("[3/5] Loading datasets...")
    train_df = pd.read_csv(f"{dataset_path}/train.csv")
    valid_df = pd.read_csv(f"{dataset_path}/valid.csv")

    eos = tokenizer.eos_token or ""
    train_df["formatted_text"] = train_df.apply(
        lambda row: format_prompt(text=row["text"], label=row["label_text"], eos_token=eos), axis=1
    )
    valid_df["formatted_text"] = valid_df.apply(
        lambda row: format_prompt(text=row["text"], label=row["label_text"], eos_token=eos), axis=1
    )

    dataset = Dataset.from_pandas(train_df[["formatted_text"]])
    eval_dataset = Dataset.from_pandas(valid_df[["formatted_text"]])
    print(f"  → Train: {len(dataset)} samples, Valid: {len(eval_dataset)} samples")

    print("[4/5] Training...\n")

    def formatting_func(examples):
        texts = examples["formatted_text"]
        return [texts] if isinstance(texts, str) else list(texts)

    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        data_collator=data_collator,
        args=SFTConfig(
            max_seq_length=max_seq_length,
            packing=False,
            per_device_train_batch_size=config.get("batch_size", 2),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            warmup_steps=config.get("warmup_steps", 10),
            num_train_epochs=config.get("num_train_epochs", 3),
            learning_rate=config.get("learning_rate", 2e-4),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=20,
            optim=config.get("optimizer", "adamw_8bit"),
            weight_decay=config.get("weight_decay", 0.01),
            lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
            seed=3407,
            output_dir="outputs",
            report_to="none",
            average_tokens_across_devices=False,
        ),
    )

    trainer.train()

    print(f"\n[5/5] Training complete! Saving checkpoint → {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    evaluate_on_test(model, tokenizer, eval_config, output_dir)

    print("\n✓ Pipeline complete!")


if __name__ == "__main__":
    main()
