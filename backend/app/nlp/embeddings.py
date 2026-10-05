"""Sentence-Transformers wrapper for computing semantic similarity.

We lazily load the model once per process (it's a ~90MB download the
first time) and expose a simple `cosine_similarity(a, b)` helper plus a
batch `embed(texts)` helper for efficiency when comparing one claim
against several evidence snippets.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                logger.info(
                    "Loading Sentence-Transformers model: %s", settings.SENTENCE_TRANSFORMER_MODEL
                )
                _model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Return an (N, D) matrix of L2-normalized embeddings for `texts`."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return embeddings


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two pieces of text, in range [-1, 1]."""
    if not text_a or not text_b:
        return 0.0
    vectors = embed([text_a, text_b])
    a, b = vectors[0], vectors[1]
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def best_similarity(claim: str, evidence_texts: list[str]) -> tuple[float, int]:
    """Return (max similarity, index of best-matching evidence) for a claim
    against a list of evidence snippets. Returns (0.0, -1) if no evidence.
    """
    if not evidence_texts:
        return 0.0, -1

    claim_vec = embed([claim])[0]
    evidence_vecs = embed(evidence_texts)

    best_score = -1.0
    best_idx = -1
    for idx, vec in enumerate(evidence_vecs):
        denom = np.linalg.norm(claim_vec) * np.linalg.norm(vec)
        score = float(np.dot(claim_vec, vec) / denom) if denom else 0.0
        if score > best_score:
            best_score = score
            best_idx = idx
    return max(best_score, 0.0), best_idx
