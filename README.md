# Banking Intent Classification với Unsloth — BANKING77

Fine-tune mô hình ngôn ngữ nhỏ (**Qwen2.5-1.5B**) cho bài toán phân loại ý định khách hàng ngân hàng (77 intents) bằng **Unsloth** + **LoRA** (PEFT).

Tập dữ liệu: [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77)

---

## 1. Yêu cầu Môi trường

> **Lưu ý:** Unsloth yêu cầu GPU NVIDIA (CUDA). Nếu bạn không có GPU phù hợp, hãy sử dụng **Google Colab** (Runtime → T4 GPU).

### Cài đặt

```bash
git clone <YOUR_REPO_URL>
cd banking-intent-unsloth
pip install -r requirements.txt
```

---

## 2. Tiền xử lý Dữ liệu

Tự động tải `BANKING77`, chuẩn hóa văn bản (lowercase, loại ký tự đặc biệt), lấy mẫu 20 câu/class (train) + 5 câu/class (test):

```bash
python scripts/preprocess_data.py
```

Kết quả lưu tại `sample_data/`:
- `train.csv` — ~1540 mẫu (text, label, label_text, label_id)
- `test.csv` — ~385 mẫu
- `label_map.csv` — mapping 77 nhãn

---

## 3. Huấn luyện

### Hyperparameters

| Tham số | Giá trị |
|---|---|
| Model | `unsloth/Qwen2.5-1.5B` |
| Max sequence length | 512 |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 (optimized by Unsloth) |
| Batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Learning rate | 2e-4 |
| Optimizer | AdamW 8-bit |
| Epochs | 3 |
| Warmup steps | 10 |
| Weight decay (L2 reg.) | 0.01 |
| LR scheduler | Linear |
| Quantization | QLoRA (4-bit) |
| Gradient checkpointing | Unsloth optimized |

### Chạy huấn luyện

```bash
# Cách 1: Script tự động (tiền xử lý + train + evaluate)
bash train.sh

# Cách 2: Chạy thủ công từng bước
python scripts/preprocess_data.py
python scripts/train.py --config configs/train.yaml
```

Checkpoint → `./outputs/intent-model/`

Sau khi train xong, script sẽ tự động **đánh giá trên tập test** và in ra:
- **Accuracy** tổng thể
- **Classification Report** (Precision / Recall / F1 cho từng intent class)
- Kết quả được lưu tại `./outputs/intent-model/evaluation_results.txt`

---

## 4. Inference (Suy luận)

### Sử dụng qua Shell Script

```bash
# Chạy 1 câu
bash inference.sh "I am still waiting on my card?"

# Chạy chế độ demo (nhiều ví dụ)
bash inference.sh
```

### Sử dụng trong Python

```python
from scripts.inference import IntentClassification

# Khởi tạo — model_path trỏ tới file config YAML
classifier = IntentClassification("configs/inference.yaml")

# Dự đoán
label = classifier("I am still waiting on my card?")
print(label)  # → "card_arrival"

label = classifier("How do I change my pin number?")
print(label)  # → "change_pin"
```

---

## 5. Cấu trúc Source Code

```text
banking-intent-unsloth/
├── scripts/
│   ├── utils.py              # Hằng số & hàm tiện ích dùng chung
│   ├── preprocess_data.py    # Tải & tiền xử lý dữ liệu
│   ├── train.py              # Fine-tuning + đánh giá
│   └── inference.py          # Class IntentClassification
├── configs/
│   ├── train.yaml            # Cấu hình hyperparameters
│   └── inference.yaml        # Cấu hình đường dẫn checkpoint
├── sample_data/              # (sinh ra sau preprocess)
│   ├── train.csv
│   ├── test.csv
│   └── label_map.csv
├── outputs/                  # (sinh ra sau training)
│   └── intent-model/
│       └── evaluation_results.txt
├── train.sh
├── inference.sh
├── requirements.txt
└── README.md
```

---

## 6. Video Demo

> 📹 [Google Drive — Video Demo Inference](<CHÈN_LINK_GOOGLE_DRIVE_CỦA_BẠN_VÀO_ĐÂY>)
>
> *(Đảm bảo link ở chế độ **công khai** — Anyone with the link can view)*
