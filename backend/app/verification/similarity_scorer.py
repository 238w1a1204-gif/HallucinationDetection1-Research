"""Thin wrapper around app.nlp.embeddings for the verification layer.

Kept as its own module (rather than calling embeddings directly from the
verdict engine) so that verification code depends on a "verification
concept" (per-evidence similarity scores) rather than the raw NLP layer,
matching the modular separation required by the project spec.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.nlp.embeddings import embed
import numpy as np


@dataclass
class ScoredEvidence:
    evidence_text: str
    source_name: str
    source_url: str | None
    similarity: float


def score_evidence_against_claim(
    claim: str, evidence_items: list[tuple[str, str, str | None]]
) -> list[ScoredEvidence]:
    """Score each (evidence_text, source_name, source_url) tuple against `claim`.

    Returns a list of ScoredEvidence sorted by similarity, descending.
    """
    if not evidence_items:
        return []

    texts = [item[0] for item in evidence_items]
    claim_vec = embed([claim])[0]
    evidence_vecs = embed(texts)

    scored: list[ScoredEvidence] = []
    for (text, source_name, source_url), vec in zip(evidence_items, evidence_vecs):
        denom = np.linalg.norm(claim_vec) * np.linalg.norm(vec)
        sim = float(np.dot(claim_vec, vec) / denom) if denom else 0.0
        scored.append(
            ScoredEvidence(
                evidence_text=text,
                source_name=source_name,
                source_url=source_url,
                similarity=max(sim, 0.0),
            )
        )
    scored.sort(key=lambda s: s.similarity, reverse=True)
    return scored
