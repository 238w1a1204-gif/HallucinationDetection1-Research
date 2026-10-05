"""Optional Gemini provider.

Only used when LLM_PROVIDER=gemini and GEMINI_API_KEY is set. Requires the
`google-generativeai` package (not installed by default since this project
targets the free local Hugging Face path). Install it manually if you want
to use Gemini:

    pip install google-generativeai
"""
from __future__ import annotations

import logging

from app.config import get_settings
from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)
settings = get_settings()


class GeminiProvider(LLMProvider):
    def __init__(self) -> None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file to use the Gemini provider."
            )
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "google-generativeai is not installed. Run `pip install google-generativeai`."
            ) from exc

        genai.configure(api_key=settings.GEMINI_API_KEY)
        self._model = genai.GenerativeModel(settings.GEMINI_MODEL)

    def generate_answer(self, question: str) -> str:
        response = self._model.generate_content(
            f"Answer the question factually and concisely: {question}"
        )
        return response.text.strip()

    def generate_correction(self, original_answer: str, evidence_summary: str) -> str:
        prompt = (
            "Rewrite the following answer so it only contains information "
            "confirmed by the evidence. Do not add new facts.\n"
            f"Original answer: {original_answer}\n"
            f"Evidence: {evidence_summary}\n"
            "Corrected answer:"
        )
        response = self._model.generate_content(prompt)
        return response.text.strip()
