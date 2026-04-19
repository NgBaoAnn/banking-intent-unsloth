"""
inference.py
------------
Module cung cấp class IntentClassification dùng để dự đoán
nhãn ý định (intent label) từ một tin nhắn ngân hàng.

Yêu cầu 2.3:
  - __init__(model_path): nạp config, tokenizer, checkpoint.
  - __call__(message): nhận tin nhắn → trả về predicted_label.

Ví dụ sử dụng (sau khi huấn luyện):
    >>> from scripts.inference import IntentClassification
    >>> classifier = IntentClassification("configs/inference.yaml")
    >>> label = classifier("I am still waiting on my card?")
    >>> print(label)  # ví dụ: "card_arrival"
"""

import yaml
import torch
import argparse
from unsloth import FastLanguageModel

from utils import format_prompt


class IntentClassification:
    """Phân loại ý định khách hàng ngân hàng bằng mô hình đã tinh chỉnh."""

    def __init__(self, model_path: str):
        """
        Khởi tạo mô hình.

        Parameters
        ----------
        model_path : str
            Đường dẫn tới file cấu hình YAML chứa ít nhất key 'model_path'
            trỏ đến thư mục checkpoint đã lưu.
        """
        with open(model_path, "r") as f:
            config = yaml.safe_load(f)

        real_model_path = config.get("model_path", "./outputs/intent-model")
        max_seq_length = config.get("max_seq_length", 512)

        # Tự động phát hiện device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=real_model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        # Bật chế độ inference nhanh
        FastLanguageModel.for_inference(self.model)

    def __call__(self, message: str) -> str:
        """
        Dự đoán nhãn ý định cho một tin nhắn đầu vào.

        Parameters
        ----------
        message : str
            Tin nhắn/câu hỏi của khách hàng ngân hàng.

        Returns
        -------
        predicted_label : str
            Nhãn ý định được dự đoán (ví dụ: 'card_arrival').
        """
        # Tạo prompt inference (không có label, không có EOS)
        prompt = format_prompt(text=message)

        inputs = self.tokenizer(
            [prompt], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Chỉ decode phần mới sinh ra (bỏ phần prompt)
        input_length = inputs["input_ids"].shape[1]
        decoded = self.tokenizer.batch_decode(
            outputs[:, input_length:], skip_special_tokens=True
        )[0]

        # Lấy dòng đầu tiên làm predicted label
        predicted_label = decoded.strip().split("\n")[0].strip()
        return predicted_label


# ---------------------------------------------------------------------------
# CLI entry point & ví dụ sử dụng
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intent Classification Inference")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Đường dẫn tới file config YAML (inference.yaml)"
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Tin nhắn đầu vào cần phân loại"
    )
    args = parser.parse_args()

    # Khởi tạo classifier
    classifier = IntentClassification(model_path=args.config)

    # Nếu có --text thì chỉ chạy 1 câu
    if args.text:
        print("\n" + "=" * 50)
        print("  INFERENCE")
        print("=" * 50)
        print(f"  Message : {args.text}")
        print(f"  Intent  : {classifier(args.text)}")
        print("=" * 50)
    else:
        # Ví dụ sử dụng (Yêu cầu 2.3: cung cấp usage example)
        examples = [
            "I am still waiting on my card?",
            "How do I change my pin number?",
            "Why was I charged a fee?",
            "I want to make a transfer to my friend",
            "Is my card compatible with Apple Pay?",
        ]
        print("\n" + "=" * 50)
        print("  INFERENCE — VÍ DỤ SỬ DỤNG")
        print("=" * 50)
        for msg in examples:
            label = classifier(msg)
            print(f"  Input  : {msg}")
            print(f"  Intent : {label}\n")
