SYSTEM_PROMPT = (
    "Below is an instruction that describes a banking customer's request. "
    "Identify the specific intent of the query."
)


def format_prompt(text: str, label: str = "", eos_token: str = "") -> str:
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"### Instruction:\n"
        f"Input: {text}\n\n"
        f"### Response:\n"
    )
    if label:
        prompt += f"{label}{eos_token}"
    return prompt
