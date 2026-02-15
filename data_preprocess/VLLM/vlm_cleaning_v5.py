#!/usr/bin/env python3
"""Shared structural cleaning for tinyLLM VLM continual SFT v5.

The policy is intentionally structural: normalize roles and image placeholders,
reject malformed conversations, and leave ordinary speculative wording or
incidental OCR content untouched.
"""
from __future__ import annotations

import re
from typing import Any, Iterable


IMG_TOKEN = "<img>"

# Known dataset placeholders. These are removed everywhere before one canonical
# marker is inserted into the first user turn.
IMAGE_MARKER_RE = re.compile(
    r"(?:"
    r"<\s*/?\s*(?:image(?:_\d+)?|img|IMG_CONTEXT|图像|圖像|图片|圖片|图|圖|影像|画像|图象)\s*>"
    r"|\[\s*(?:image|img)\s*\]"
    r")",
    re.IGNORECASE,
)
CHAT_CONTROL_RE = re.compile(r"<\|(?:im|thought)_(?:start|end)\|>", re.IGNORECASE)
HTML_RE = re.compile(r"</?(?:b|br|p|div|span|strong|em|h[1-6])(?:\s[^<>]*)?/?>", re.IGNORECASE)
# A few sources contain stray XML-like pseudo-emotion tags such as <惊讶>.
PSEUDO_TAG_RE = re.compile(r"</?[A-Za-z_\u4e00-\u9fff][^<>\n]{0,31}>")
LOOP_RE = re.compile(r"(.{12,80})(?:\1){3,}", re.DOTALL)

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "question": "user",
    "instruction": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "answer": "assistant",
    "response": "assistant",
}


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = CHAT_CONTROL_RE.sub(" ", text)
    text = IMAGE_MARKER_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = PSEUDO_TAG_RE.sub(" ", text)
    text = text.replace("\x00", " ")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def normalize_role(value: Any) -> str | None:
    return ROLE_MAP.get(str(value or "").strip().lower())


def normalize_messages(
    turns: Iterable[Any],
    *,
    default_user_prompt: str,
) -> tuple[list[dict[str, str]] | None, str | None]:
    """Return canonical messages and an optional rejection reason."""
    messages: list[dict[str, str]] = []
    for raw in turns:
        if not isinstance(raw, dict):
            return None, "turn_not_object"
        role = normalize_role(raw.get("role") or raw.get("from"))
        if role is None:
            return None, "unknown_role"
        content = clean_text(raw.get("content") if "content" in raw else raw.get("value"))
        if not content:
            # A user turn that only contained <image> is meaningful; replace it
            # with a neutral request. Empty assistant turns remain invalid.
            if role == "user":
                content = default_user_prompt
            else:
                return None, "empty_assistant"
        messages.append({"role": role, "content": content})

    if len(messages) < 2:
        return None, "too_few_turns"
    if messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
        return None, "bad_end_roles"
    if any(messages[i]["role"] == messages[i - 1]["role"] for i in range(1, len(messages))):
        return None, "non_alternating"
    if not any(row["role"] == "assistant" for row in messages):
        return None, "no_assistant"
    if any(LOOP_RE.search(row["content"]) for row in messages if row["role"] == "assistant"):
        return None, "loop"

    # All source markers have already been removed. Insert exactly one canonical
    # textual marker in the first user turn, matching local inference.
    messages[0]["content"] = f"{IMG_TOKEN}\n{messages[0]['content']}"
    if sum(row["content"].count(IMG_TOKEN) for row in messages) != 1:
        return None, "image_marker_count"
    if IMG_TOKEN in "\n".join(row["content"] for row in messages if row["role"] == "assistant"):
        return None, "assistant_image_marker"
    return messages, None


def caption_messages(caption: Any, *, language: str) -> tuple[list[dict[str, str]] | None, str | None]:
    prompt = "请简洁描述这张图片。" if language == "zh" else "Describe this image briefly."
    return normalize_messages(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": caption},
        ],
        default_user_prompt=prompt,
    )


def qa_messages(pairs: Iterable[tuple[Any, Any]], *, language: str) -> tuple[list[dict[str, str]] | None, str | None]:
    prompt = "请描述这张图片。" if language == "zh" else "Describe this image."
    turns: list[dict[str, Any]] = []
    for question, answer in pairs:
        turns.append({"role": "user", "content": question})
        turns.append({"role": "assistant", "content": answer})
    return normalize_messages(turns, default_user_prompt=prompt)


def render_and_validate(tokenizer, messages: list[dict[str, str]], max_tokens: int) -> tuple[str | None, int, str | None]:
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    except Exception:
        return None, 0, "template"
    if rendered.count(IMG_TOKEN) != 1:
        return None, len(token_ids), "rendered_image_marker_count"
    if len(token_ids) > int(max_tokens):
        return None, len(token_ids), "too_long"
    return rendered, len(token_ids), None

