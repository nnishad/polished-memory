"""Normalize direct conversation evidence without inventing attachment contents."""
from __future__ import annotations

from typing import Any


def text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    return "\n".join(part["text"] for part in value
                     if isinstance(part, dict) and isinstance(part.get("text"), str)
                     and part.get("type") in {"text", "input_text", "output_text"})


def transcript(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for position, message in enumerate(messages or []):
        role = message.get("role")
        if role not in {"user", "assistant"} or message.get("_compressed_summary"):
            continue
        text = text_content(message.get("content"))
        if not text.strip():
            continue
        item = {"role": role, "text": text, "position": position}
        for key in ("author", "timestamp", "_row_id"):
            if message.get(key) is not None:
                item[key] = message[key]
        result.append(item)
    return result
