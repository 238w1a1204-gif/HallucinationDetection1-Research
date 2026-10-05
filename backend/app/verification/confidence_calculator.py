"""Heuristic confidence score.

IMPORTANT: this is a heuristic weighted-sum score, NOT a calibrated
probability.  It should be read as "how much combined signal supports
this claim", on a 0–1 scale, not as "P(claim is true)".  No claim of
statistical calibration is made anywhere in this project.

Formula (changed from the original):

  confidence = (W_SIM   × normalized_similarity)
             + (W_AGREE × (agree_count / total_sources))
             + (W_CONS  × consistency_signal)
             + (W_ENT   × entity_presence_score)

Weights (all configurable via app.config, defaults below):
  W_SIM   = 0.35   (was 0.50 — reduced because similarity alone is not proof)
  W_AGREE = 0.25   (was 0.30)
  W_CONS  = 0.20   (unchanged — zero when contradiction or entity mismatch)
  W_ENT   = 0.20   (NEW — zero when no claim entities found in evidence)

entity_presence_score: fraction of named entities in the claim that
actually appear in the evidence text.  A claim about "John McCarthy" where
the evidence only mentions "Infosys" would score 0.0 here, dragging the
overall confidence down even if semantic similarity is moderate.

consistency_signal = 1.0 unless contradiction_detected OR entity_mismatch.

DOCUMENTED LIMITATION:
Weights are heuristic starting points; they have not been calibrated
against a labelled benchmark dataset.  The entity_presence_score itself
depends on spaCy NER accuracy.
"""
from __future__ import annotations

from app.config import get_settings

settings = get_settings()

TOTAL_SOURCES_QUERIED = 3  # wikipedia, wikidata, duckduckgo

# Default weights — kept in module-level constants so tests can reference them.
WEIGHT_SIMILARITY       = 0.35
WEIGHT_AGREEMENT        = 0.25
WEIGHT_CONSISTENCY      = 0.20
WEIGHT_ENTITY_PRESENCE  = 0.20


def calculate_confidence(
    similarity: float,
    agree_count: int,
    contradiction_detected: bool,
    entity_mismatch: bool,
    entity_presence_score: float = 1.0,
    total_sources_queried: int = TOTAL_SOURCES_QUERIED,
) -> float:
    """Return a heuristic confidence score in [0.0, 1.0].

    Parameters
    ----------
    similarity:
        Best cosine similarity between claim and any evidence snippet [0,1].
    agree_count:
        Number of distinct sources whose evidence cleared the partial threshold.
    contradiction_detected:
        True if the NLI model flagged a contradiction.
    entity_mismatch:
        True if a key entity in the claim was not found in the evidence.
    entity_presence_score:
        Fraction of named entities in the claim that appear in the evidence.
        Defaults to 1.0 (no penalty) when not provided (backward-compatible).
    total_sources_queried:
        Denominator for the agreement ratio (normally 3).
    """
    norm_similarity  = max(0.0, min(similarity, 1.0))
    agreement_ratio  = (agree_count / total_sources_queried) if total_sources_queried else 0.0
    consistency_sig  = 0.0 if (contradiction_detected or entity_mismatch) else 1.0
    entity_presence  = max(0.0, min(entity_presence_score, 1.0))

    confidence = (
        WEIGHT_SIMILARITY      * norm_similarity
        + WEIGHT_AGREEMENT     * agreement_ratio
        + WEIGHT_CONSISTENCY   * consistency_sig
        + WEIGHT_ENTITY_PRESENCE * entity_presence
    )
    return round(max(0.0, min(confidence, 1.0)), 4)
