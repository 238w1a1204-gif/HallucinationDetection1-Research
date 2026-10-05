"""Optional OpenAI provider.

Only used when LLM_PROVIDER=openai and OPENAI_API_KEY is set. Requires the
`openai` package (not installed by default). Install manually if needed:

    pip install openai
"""
from __future__ import annotations

import logging

from app.config import get_settings
from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)
settings = get_settings()


class OpenAIProvider(LLMProvider):
    def __init__(self) -> None:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file to use the OpenAI provider."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package is not installed. Run `pip install openai`.") from exc

        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.OPENAI_MODEL

    def generate_answer(self, question: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "Answer the question factually and concisely."},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content.strip()

    def generate_correction(self, original_answer: str, evidence_summary: str) -> str:
        prompt = (
            "Rewrite the following answer so it only contains information "
            "confirmed by the evidence. Do not add new facts.\n"
            f"Original answer: {original_answer}\n"
            f"Evidence: {evidence_summary}\n"
            "Corrected answer:"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
