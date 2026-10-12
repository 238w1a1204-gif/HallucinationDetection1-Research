"""Multi-source agreement logic.

IMPORTANT DOCUMENTED LIMITATION:
Source agreement is used as *supporting evidence*, not as proof of truth.
Two sources can independently repeat the same wrong information (e.g. if
both ultimately trace back to the same original mistake), so agreement
increases confidence but never guarantees correctness.

Changes from original:
- Sources are only counted as "agreeing" when their evidence clears BOTH
  the similarity threshold AND (if the claim has named entities) shows at
  least minimal entity overlap with the claim.
- This prevents a situation where three sources all return topically-related
  articles (high similarity to the claim topic) but none of them actually
  contain the entity named in the claim — which was the root cause of the
  "John McCarthy / Infosys" false-positive.

The entity-presence guard is intentionally lenient (threshold 0.0+) so that
claims without named entities (e.g. "water is a chemical compound") still
count sources normally.
"""
from __future__ import annotations

from app.config import get_settings
from app.verification.entity_matcher import entity_presence_score as _ep_score
from app.verification.similarity_scorer import ScoredEvidence

settings = get_settings()

# Minimum entity-presence score for a source to be counted as "agreeing".
# Set low (>0) so that claims without named entities are unaffected (they
# return presence_score=1.0 from entity_presence_score()).
_MIN_ENTITY_PRESENCE_FOR_AGREEMENT = 0.01


def count_agreeing_sources(
    scored_evidence: list[ScoredEvidence],
    claim: str = "",
) -> int:
    """Count distinct sources that both:
      1. Have similarity >= SIMILARITY_PARTIAL_THRESHOLD, AND
      2. Have at least minimal entity overlap with the claim (if the claim
         contains named entities).

    `claim` is optional for backward compatibility; when omitted the
    entity check is skipped (entity_presence=1.0 by default).
    """
    agreeing_sources: set[str] = set()
    for item in scored_evidence:
        if item.similarity < settings.SIMILARITY_PARTIAL_THRESHOLD:
            continue
        # Entity-presence check: only meaningful when a claim string is given
        if claim:
            ep = _ep_score(claim, item.evidence_text)
            if ep < _MIN_ENTITY_PRESENCE_FOR_AGREEMENT:
                continue   # evidence doesn't mention any claim entity — skip

        source_identity = (
            item.source_url.rstrip("/").split("?")[0].lower()
            if item.source_url
            else item.source_name
        )
        agreeing_sources.add(source_identity)

    return len(agreeing_sources)
