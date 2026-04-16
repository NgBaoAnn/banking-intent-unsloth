"""
utils.py
--------
Module chứa các hằng số và hàm tiện ích dùng chung cho toàn bộ pipeline:
preprocess_data.py, train.py, inference.py.

Mục đích: tránh trùng lặp (DRY), đảm bảo prompt format luôn nhất quán.
"""


# ---------------------------------------------------------------------------
# Prompt template  — Alpaca-style instruction tuning
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Below is an instruction that describes a banking customer's request. "
    "Identify the specific intent of the query."
)


def format_prompt(text: str, label: str = "", eos_token: str = "") -> str:
    """
    Tạo prompt dạng Alpaca cho mô hình generative.

    Parameters
    ----------
    text : str
        Tin nhắn khách hàng ngân hàng.
    label : str
        Nhãn ý định (chỉ thêm vào khi training). Để trống khi inference.
    eos_token : str
        Token kết thúc chuỗi từ tokenizer (chỉ thêm khi training).

    Returns
    -------
    str
        Chuỗi prompt đã format.
    """
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"### Instruction:\n"
        f"Input: {text}\n\n"
        f"### Response:\n"
    )
    if label:
        prompt += f"{label}{eos_token}"
    return prompt
