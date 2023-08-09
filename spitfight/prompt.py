"""An abstraction layer for prompting different models."""

from __future__ import annotations

import enum

from fastchat.model.model_adapter import get_conversation_template


class Task(enum.Enum):
    """Different system prompt styles."""

    CHAT = "chat"
    CHAT_CONCISE = "chat-concise"
    INSTRUCT = "instruct"
    INSTRUCT_CONCISE = "instruct-concise"


SYSTEM_PROMPTS = {
    Task.CHAT: (
        "A chat between a human user (prompter) and an artificial intelligence (AI) assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    ),
    Task.CHAT_CONCISE: (
        "A chat between a human user (prompter) and an artificial intelligence (AI) assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "The assistant's answers are very concise. "
    ),
    Task.INSTRUCT: (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request. "
    ),
    Task.INSTRUCT_CONCISE: (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request. "
        "The response should be very concise. "
    ),
}

def get_system_prompt(task: Task | str) -> str:
    """Get the system prompt for a given task."""
    if isinstance(task, str):
        task = Task(task)
    return SYSTEM_PROMPTS[task]


def apply_model_characteristics(
    system_prompt: str,
    prompt: str,
    model_name: str,
) -> tuple[str, str | None, list[int]]:
    """Apply and return model-specific differences."""
    conv = get_conversation_template(model_name)

    if "llama-2" in model_name.lower():
        conv.system = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    elif "stablelm" in model_name.lower():
        conv.system = f"""<|SYSTEM|># {system_prompt}\n"""
    else:
        conv.system = system_prompt
    conv.messages = []
    conv.offset = 0

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")

    stop_str = None if conv.stop_str is None or not conv.stop_str else conv.stop_str

    return conv.get_prompt(), stop_str, (conv.stop_token_ids or [])
