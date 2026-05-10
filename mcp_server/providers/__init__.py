"""Model-agnostic LLM integration: env slots, vendor backends, packaged prompts."""

from mcp_server.providers.chat_providers import (
    AnthropicBackend,
    ChatBackend,
    LlmSlotConfig,
    OpenAIStyleBackend,
    build_chat_backend,
    resolve_slot_config,
)
from mcp_server.providers.prompt_loader import load_llm_prompt

__all__ = [
    "AnthropicBackend",
    "ChatBackend",
    "LlmSlotConfig",
    "OpenAIStyleBackend",
    "build_chat_backend",
    "load_llm_prompt",
    "resolve_slot_config",
]
