"""Builds a human-readable explanation string for a claim's verdict.

The verdict engine already produces a machine-oriented `explanation`
field; this module is used by the report generator to turn a full set of
per-claim results into a readable narrative for the "Explainability"
pipeline stage and the downloadable report.
"""
from __future__ import annotations

from app.verification.verdict_engine import ClaimVerdict


def explain_claim(claim_text: str, verdict: ClaimVerdict) -> str:
    return (
        f"Claim: \"{claim_text}\"\n"
        f"Verdict: {verdict.verdict}\n"
        f"Max similarity to evidence: {verdict.max_similarity:.2f}\n"
        f"Sources agreeing: {verdict.num_sources_agreeing}\n"
        f"Contradiction detected: {verdict.contradiction_detected}\n"
        f"Entity/date mismatch: {verdict.entity_mismatch}\n"
        f"Reasoning: {verdict.explanation}"
    )
