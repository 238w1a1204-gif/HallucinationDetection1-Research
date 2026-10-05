"""Aggregates claim-level verdicts into an answer-level hallucination diagnosis.

Rules (documented, applied in order):
1. If any claim is CONTRADICTED -> overall answer "contains hallucination".
2. If all claims are VERIFIED -> "no hallucination detected".
3. Otherwise (any mix involving PARTIALLY_SUPPORTED/UNSUPPORTED but no
   CONTRADICTED) -> "partially hallucinated / unverifiable".

Overall confidence = simple average of per-claim confidence scores
(each claim counts equally, since each is one factual unit).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.verification.verdict_engine import (
    VERDICT_CONTRADICTED,
    VERDICT_UNSUPPORTED,
    VERDICT_VERIFIED,
)

OVERALL_HALLUCINATED = "CONTAINS_HALLUCINATION"
OVERALL_CLEAN = "NO_HALLUCINATION_DETECTED"
OVERALL_PARTIAL = "PARTIALLY_HALLUCINATED_OR_UNVERIFIABLE"
OVERALL_NO_CLAIMS = "NO_CLAIMS_EXTRACTED"


@dataclass
class HallucinationAnalysis:
    overall_verdict: str
    overall_confidence: float
    num_claims: int
    num_verified: int
    num_partially_supported: int
    num_unsupported: int
    num_contradicted: int


def analyze(claim_verdicts: list[str], claim_confidences: list[float]) -> HallucinationAnalysis:
    num_claims = len(claim_verdicts)
    if num_claims == 0:
        return HallucinationAnalysis(
            overall_verdict=OVERALL_NO_CLAIMS,
            overall_confidence=0.0,
            num_claims=0,
            num_verified=0,
            num_partially_supported=0,
            num_unsupported=0,
            num_contradicted=0,
        )

    num_verified = claim_verdicts.count(VERDICT_VERIFIED)
    num_contradicted = claim_verdicts.count(VERDICT_CONTRADICTED)
    num_unsupported = claim_verdicts.count(VERDICT_UNSUPPORTED)
    num_partially = claim_verdicts.count("PARTIALLY_SUPPORTED")

    if num_contradicted > 0:
        overall_verdict = OVERALL_HALLUCINATED
    elif num_verified == num_claims:
        overall_verdict = OVERALL_CLEAN
    else:
        overall_verdict = OVERALL_PARTIAL

    overall_confidence = round(sum(claim_confidences) / len(claim_confidences), 4) if claim_confidences else 0.0

    return HallucinationAnalysis(
        overall_verdict=overall_verdict,
        overall_confidence=overall_confidence,
        num_claims=num_claims,
        num_verified=num_verified,
        num_partially_supported=num_partially,
        num_unsupported=num_unsupported,
        num_contradicted=num_contradicted,
    )
