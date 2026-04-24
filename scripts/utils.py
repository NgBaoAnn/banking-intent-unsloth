# Alpaca-style prompt template for banking intent classification
SYSTEM_PROMPT = (
    "Below is an instruction that describes a banking customer's request. "
    "Identify the specific intent of the query."
)


def format_prompt(text: str, label: str = "", eos_token: str = "") -> str:
    """
    Build an Alpaca-style prompt.

    Args:
        text: Customer message.
        label: Intent label — include only during training, leave empty for inference.
        eos_token: EOS token from the tokenizer — include only during training.
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
