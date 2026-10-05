"""Local/Hugging Face based LLM provider.

Uses a small instruction-tuned seq2seq model (default: google/flan-t5-base)
via the `transformers` pipeline API. This runs fully offline after the
first download and requires no API key, which is why it is the default
provider for this project.

Limitation: flan-t5-base is a small model. Its answers are shorter and
less fluent than a large commercial LLM. This is acceptable for a student
project whose focus is the *verification pipeline*, not the quality of
the generator itself.
"""
from __future__ import annotations

import logging
import threading

from app.config import get_settings
from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)
settings = get_settings()

_lock = threading.Lock()
_pipeline = None


def _get_pipeline():
    """Lazily load the HF pipeline (expensive) only once, on first use."""
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                from transformers import pipeline

                logger.info("Loading Hugging Face generation model: %s", settings.HF_GENERATION_MODEL)
                _pipeline = pipeline("text2text-generation", model=settings.HF_GENERATION_MODEL)
    return _pipeline


class HuggingFaceProvider(LLMProvider):
    def generate_answer(self, question: str) -> str:
        try:
            pipe = _get_pipeline()
            result = pipe(
                f"Answer the question factually and concisely: {question}",
                max_new_tokens=128,
                do_sample=False,
            )
            return result[0]["generated_text"].strip()
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("Hugging Face generation failed: %s", exc)
            return (
                "I could not generate an answer locally due to a model error. "
                "Please try again or check that the model downloaded correctly."
            )

    def generate_correction(self, original_answer: str, evidence_summary: str) -> str:
        try:
            pipe = _get_pipeline()
            prompt = (
                "Rewrite the following answer so it only contains information "
                "confirmed by the evidence. Do not add new facts.\n"
                f"Original answer: {original_answer}\n"
                f"Evidence: {evidence_summary}\n"
                "Corrected answer:"
            )
            result = pipe(prompt, max_new_tokens=160, do_sample=False)
            return result[0]["generated_text"].strip()
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("Hugging Face correction failed: %s", exc)
            return original_answer
