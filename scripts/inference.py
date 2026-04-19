"""
inference.py
------------
Module cung cấp class IntentClassification.

Yêu cầu 2.3:
  - __init__(model_path): nạp config, tokenizer, checkpoint.
  - __call__(message): nhận tin nhắn → trả về predicted_label.
"""

import yaml
import torch
from unsloth import FastLanguageModel

from utils import format_prompt


class IntentClassification:
    """Phân loại ý định khách hàng ngân hàng bằng mô hình đã tinh chỉnh."""

    def __init__(self, model_path: str):
        with open(model_path, "r") as f:
            config = yaml.safe_load(f)

        real_model_path = config.get("model_path", "./outputs/intent-model")
        max_seq_length = config.get("max_seq_length", 512)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=real_model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

    def __call__(self, message: str) -> str:
        prompt = format_prompt(text=message)

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        decoded = self.tokenizer.batch_decode(
            outputs[:, input_length:], skip_special_tokens=True
        )[0]

        predicted_label = decoded.strip().split("\n")[0].strip()
        return predicted_label
