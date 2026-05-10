"""Vendor-neutral LLM backends. Model IDs and vendors come from environment only."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LlmSlotConfig:
    slot: str
    vendor: str
    model: str
    api_key: str
    openai_base_url: str | None


class ChatBackend(ABC):
    def __init__(self, config: LlmSlotConfig) -> None:
        self.config = config

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 300,
        json_object: bool = False,
    ) -> str:
        raise NotImplementedError


class OpenAIStyleBackend(ChatBackend):
    def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 300,
        json_object: bool = False,
    ) -> str:
        import openai

        kwargs: dict[str, Any] = {"api_key": self.config.api_key}
        if self.config.openai_base_url:
            kwargs["base_url"] = self.config.openai_base_url
        client = openai.OpenAI(**kwargs)
        create_kw: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        if json_object:
            create_kw["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**create_kw)
        msg = resp.choices[0].message
        return (msg.content or "").strip()


class AnthropicBackend(ChatBackend):
    def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 300,
        json_object: bool = False,
    ) -> str:
        import anthropic

        _ = json_object
        client = anthropic.Anthropic(api_key=self.config.api_key)
        resp = client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        if not resp.content:
            return ""
        block = resp.content[0]
        return block.text.strip() if hasattr(block, "text") else str(block).strip()


def build_chat_backend(config: LlmSlotConfig) -> ChatBackend:
    v = config.vendor.lower().strip()
    if v in ("openai", "openai_compatible"):
        if v == "openai_compatible" and not (config.openai_base_url or "").strip():
            raise ValueError(
                "openai_compatible requires BIAS_LLM_<SLOT>_OPENAI_BASE_URL "
                "or OPENAI_COMPATIBLE_BASE_URL."
            )
        return OpenAIStyleBackend(config)
    if v == "anthropic":
        return AnthropicBackend(config)
    raise ValueError(
        f"Unknown vendor {config.vendor!r} for slot {config.slot!r}. "
        "Use: openai | anthropic | openai_compatible"
    )


def _env_first(*keys: str) -> str:
    for k in keys:
        v = os.environ.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def resolve_slot_config(slot: str) -> LlmSlotConfig:
    """Resolve slot ``A``, ``B``, ``AUDITOR``, ``VERIFIER``, etc. **No defaults** for vendor/model."""
    su = slot.upper().replace("-", "_")
    vendor = _env_first(f"BIAS_LLM_{su}_VENDOR")
    model = _env_first(f"BIAS_LLM_{su}_MODEL")
    if not vendor or not model:
        raise ValueError(
            f"Set BIAS_LLM_{su}_VENDOR and BIAS_LLM_{su}_MODEL (no hard-coded defaults)."
        )

    key_direct = _env_first(f"BIAS_LLM_{su}_API_KEY")
    if vendor.lower() == "anthropic":
        api_key = key_direct or _env_first("ANTHROPIC_API_KEY")
    else:
        api_key = key_direct or _env_first("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            f"Missing API key for slot {slot!r} (vendor={vendor}). "
            f"Set BIAS_LLM_{su}_API_KEY or the provider default env var."
        )

    base_url = _env_first(f"BIAS_LLM_{su}_OPENAI_BASE_URL") or None
    if not base_url and vendor.lower() == "openai":
        base_url = _env_first("OPENAI_BASE_URL") or None
    if vendor.lower() == "openai_compatible" and not base_url:
        base_url = _env_first("OPENAI_COMPATIBLE_BASE_URL") or None

    return LlmSlotConfig(
        slot=su,
        vendor=vendor.lower(),
        model=model,
        api_key=api_key,
        openai_base_url=base_url,
    )
