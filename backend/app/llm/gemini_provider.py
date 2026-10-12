"""Gemini provider using Google's maintained google-genai SDK."""
from __future__ import annotations

from google import genai

from app.config import get_settings
from app.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self) -> None:
        settings = get_settings()

        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is missing from backend/.env"
            )

        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._model = settings.GEMINI_MODEL

    def _generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        text = response.text

        if not text or not text.strip():
            raise RuntimeError("Gemini returned an empty text response.")

        return text.strip()

    def generate_answer(self, question: str) -> str:
        prompt = (
            "Answer the question factually and concisely. "
            "If uncertain, say so rather than guessing.\n"
            f"Question: {question}"
        )
        return self._generate(prompt)

    def generate_correction(
        self, original_answer: str, evidence_summary: str
    ) -> str:
        prompt = (
            "Correct the answer using only information supported by the "
            "provided evidence. Do not add unsupported facts. If the "
            "evidence is insufficient, say so.\n\n"
            f"Original answer: {original_answer}\n"
            f"Evidence: {evidence_summary}\n"
            "Corrected answer:"
        )
        return self._generate(prompt)
