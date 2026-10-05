"""Abstract interface every LLM provider must implement.

The rest of the pipeline only ever calls `generate_answer(question)`.
This means swapping Hugging Face for Gemini/OpenAI later requires adding
one new class here, never touching claim extraction, retrieval, or
verification code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate_answer(self, question: str) -> str:
        """Return a free-text answer to `question`."""
        raise NotImplementedError

    @abstractmethod
    def generate_correction(self, original_answer: str, evidence_summary: str) -> str:
        """Return a corrected version of `original_answer` given evidence text.

        Implementations that cannot reasonably rewrite text (e.g. very small
        models) may fall back to a simple templated correction; this is
        documented per-provider.
        """
        raise NotImplementedError
