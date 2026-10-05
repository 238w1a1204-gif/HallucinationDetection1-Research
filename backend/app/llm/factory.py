"""Picks the correct LLMProvider implementation based on config.

If LLM_PROVIDER is set to "gemini" or "openai" but the corresponding key
or package is missing, we log a warning and fall back to the local
Hugging Face provider so the application keeps working.
"""
from __future__ import annotations

import logging

from app.config import get_settings
from app.llm.base import LLMProvider
from app.llm.huggingface_provider import HuggingFaceProvider

logger = logging.getLogger(__name__)


def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    provider_name = settings.LLM_PROVIDER.lower().strip()

    if provider_name == "gemini":
        try:
            from app.llm.gemini_provider import GeminiProvider

            return GeminiProvider()
        except Exception as exc:
            logger.warning("Falling back to HuggingFaceProvider: Gemini unavailable (%s)", exc)
            return HuggingFaceProvider()

    if provider_name == "openai":
        try:
            from app.llm.openai_provider import OpenAIProvider

            return OpenAIProvider()
        except Exception as exc:
            logger.warning("Falling back to HuggingFaceProvider: OpenAI unavailable (%s)", exc)
            return HuggingFaceProvider()

    # default / "huggingface"
    return HuggingFaceProvider()
