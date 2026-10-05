"""
Central configuration for the Hallucination Detection backend.

All tunable values (model names, thresholds, weights) live here so that
no module hard-codes a "magic number". Values are read from environment
variables (or a `.env` file) with sensible defaults that work out of the
box using only the free local Hugging Face provider.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- LLM provider ----
    LLM_PROVIDER: str = "huggingface"  # "huggingface" | "gemini" | "openai"
    HF_GENERATION_MODEL: str = "google/flan-t5-base"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # ---- NLP models ----
    SPACY_MODEL: str = "en_core_web_sm"
    SENTENCE_TRANSFORMER_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    NLI_MODEL: str = "facebook/bart-large-mnli"

    # ---- Verification thresholds ----
    # NOTE: these are practical starting points based on common usage of
    # MiniLM-style sentence embeddings, NOT scientifically calibrated
    # constants. They should be tuned during the Phase 12 dataset
    # evaluation described in the README.
    SIMILARITY_STRONG_THRESHOLD: float = 0.75
    SIMILARITY_PARTIAL_THRESHOLD: float = 0.55
    AGREEMENT_MIN_SOURCES: int = 2

    # ---- Confidence score weights (heuristic, NOT a calibrated probability) ----
    CONFIDENCE_WEIGHT_SIMILARITY: float = 0.5
    CONFIDENCE_WEIGHT_AGREEMENT: float = 0.3
    CONFIDENCE_WEIGHT_CONSISTENCY: float = 0.2

    # ---- Database ----
    DATABASE_URL: str = "sqlite:///./hallucination_detection.db"

    # ---- Retrieval ----
    RETRIEVAL_TIMEOUT_SECONDS: int = 8
    RETRIEVAL_MAX_RESULTS_PER_SOURCE: int = 3

    # ---- CORS ----
    FRONTEND_ORIGIN: str = "http://localhost:5173"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance (env is only parsed once per process)."""
    return Settings()
