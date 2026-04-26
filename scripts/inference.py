import yaml
import torch
import argparse
from unsloth import FastLanguageModel

from utils import format_prompt


class IntentClassification:

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

        return decoded.strip().split("\n")[0].strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Banking Intent Classification Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to inference.yaml")
    parser.add_argument("--text", type=str, default=None, help="Input message to classify")
    args = parser.parse_args()

    classifier = IntentClassification(model_path=args.config)

    if args.text:
        print("\n" + "=" * 50)
        print(f"  Message : {args.text}")
        print(f"  Intent  : {classifier(args.text)}")
        print("=" * 50)
    else:
        examples = [
            "I am still waiting on my card?",
            "How do I change my pin number?",
            "Why was I charged a fee?",
            "I want to make a transfer to my friend",
            "Is my card compatible with Apple Pay?",
        ]
        print("\n" + "=" * 50)
        print("  INFERENCE DEMO")
        print("=" * 50)
        for msg in examples:
            print(f"  Input  : {msg}")
            print(f"  Intent : {classifier(msg)}\n")
