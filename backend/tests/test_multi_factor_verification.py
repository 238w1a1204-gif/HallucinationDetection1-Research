"""Regression tests for multi-factor factual verification.

These tests verify that the system correctly handles cases where:
1. Similarity is high but entity is wrong → should NOT be VERIFIED/PARTIALLY_SUPPORTED
2. Entities match but relationship is wrong → downgraded verdict
3. Evidence directly supports the relationship → VERIFIED/PARTIALLY_SUPPORTED
4. Evidence contradicts the claim → CONTRADICTED
5. Evidence is insufficient → UNSUPPORTED

All embedding and NLI calls are mocked to make tests fast and deterministic.
Entity and relationship checks run against the actual spaCy model.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch

from app.verification.similarity_scorer import ScoredEvidence
from app.verification.verdict_engine import (
    determine_verdict,
    VERDICT_VERIFIED,
    VERDICT_PARTIALLY_SUPPORTED,
    VERDICT_UNSUPPORTED,
    VERDICT_CONTRADICTED,
)
from app.verification.confidence_calculator import calculate_confidence
from app.correction.answer_corrector import (
    ClaimCorrectionInput,
    generate_corrected_answer,
)
from app.config import get_settings

settings = get_settings()

# ── Helpers ──────────────────────────────────────────────────────────────────

def _no_contradiction():
    return patch(
        "app.verification.verdict_engine.detect_contradiction",
        return_value=(False, "neutral"),
    )

def _with_nli_contradiction():
    return patch(
        "app.verification.verdict_engine.detect_contradiction",
        return_value=(True, "contradiction"),
    )

def _ev(text: str, similarity: float, source: str = "duckduckgo") -> ScoredEvidence:
    return ScoredEvidence(
        evidence_text=text,
        source_name=source,
        source_url=None,
        similarity=similarity,
    )

STRONG = settings.SIMILARITY_STRONG_THRESHOLD
PARTIAL = settings.SIMILARITY_PARTIAL_THRESHOLD


# ── Core scenario: wrong entity, high similarity ──────────────────────────────

class TestWrongEntityHighSimilarity:
    """John McCarthy / Infosys — the original failing case."""

    # Similarity in the partial range with evidence that contains no people
    def test_wrong_person_partial_similarity_returns_not_supported(self):
        """
        Claim: 'john mccarthy was the ceo of infosys'
        Evidence: generic Infosys description with no people mentioned.
        Similarity: in the partial range (0.60).
        Expected: CONTRADICTED or UNSUPPORTED (NOT PARTIALLY_SUPPORTED).
        """
        evidence = [_ev(
            "Infosys Limited is an Indian multinational information technology "
            "company that offers business consulting, IT and outsourcing services.",
            similarity=0.60,
        )]
        with _no_contradiction():
            verdict = determine_verdict(
                "john mccarthy was the ceo of infosys", evidence
            )
        # Must NOT be PARTIALLY_SUPPORTED — entity "john mccarthy" not in evidence
        assert verdict.verdict in (VERDICT_CONTRADICTED, VERDICT_UNSUPPORTED), (
            f"Expected CONTRADICTED or UNSUPPORTED, got {verdict.verdict}. "
            f"Explanation: {verdict.explanation}"
        )

    def test_wrong_person_strong_similarity_returns_contradicted(self):
        """
        High similarity to Infosys text but wrong person → CONTRADICTED.
        """
        evidence = [_ev(
            "Infosys was co-founded by Narayana Murthy in 1981.",
            similarity=STRONG + 0.05,
        )]
        with _no_contradiction():
            verdict = determine_verdict(
                "john mccarthy was the ceo of infosys", evidence
            )
        assert verdict.verdict == VERDICT_CONTRADICTED
        assert verdict.entity_mismatch is True

    def test_confidence_low_for_wrong_entity_claim(self):
        """Confidence should be below 0.5 when entity is missing from evidence."""
        score = calculate_confidence(
            similarity=0.65,
            agree_count=1,
            contradiction_detected=False,
            entity_mismatch=True,
            entity_presence_score=0.0,
        )
        assert score < 0.5, f"Expected confidence < 0.5, got {score}"

    def test_corrected_answer_does_not_repeat_wrong_claim(self):
        """
        PARTIALLY_SUPPORTED + entity_mismatch=True → claim removed from corrected answer.
        """
        inputs = [
            ClaimCorrectionInput(
                claim_text="john mccarthy was the ceo of infosys",
                verdict=VERDICT_PARTIALLY_SUPPORTED,
                best_evidence_text="Infosys is an IT company.",
                entity_mismatch=True,
            )
        ]
        corrected = generate_corrected_answer(inputs)
        # The original wrong claim must NOT appear verbatim
        assert "john mccarthy" not in corrected.lower() or "removed" in corrected.lower() or "not verified" in corrected.lower()
        # A note about the removal should be present
        assert "corrections applied" in corrected.lower() or "could not be verified" in corrected.lower()


# ── Correct entity, evidence discusses relationship ────────────────────────────

class TestCorrectEntityRelationshipSupported:

    def test_correct_person_and_relationship_can_be_verified(self):
        """
        Claim: 'Narayana Murthy was the CEO of Infosys'
        Evidence: mentions Narayana Murthy AND the CEO relationship.
        Similarity: strong.  Two sources.
        Expected: VERIFIED.
        """
        evidence = [
            _ev(
                "Narayana Murthy is the co-founder and former CEO of Infosys.",
                similarity=STRONG + 0.05,
                source="wikipedia",
            ),
            _ev(
                "Infosys CEO Narayana Murthy led the company for two decades.",
                similarity=PARTIAL + 0.05,
                source="wikidata",
            ),
        ]
        with _no_contradiction():
            verdict = determine_verdict(
                "Narayana Murthy was the CEO of Infosys", evidence
            )
        assert verdict.verdict in (VERDICT_VERIFIED, VERDICT_PARTIALLY_SUPPORTED)
        # Must NOT be UNSUPPORTED or CONTRADICTED
        assert verdict.verdict not in (VERDICT_UNSUPPORTED, VERDICT_CONTRADICTED)

    def test_corrected_answer_keeps_verified_claims(self):
        inputs = [
            ClaimCorrectionInput(
                claim_text="Albert Einstein was born in Ulm",
                verdict=VERDICT_VERIFIED,
                best_evidence_text="Albert Einstein was born in Ulm, Germany in 1879.",
                entity_mismatch=False,
            )
        ]
        corrected = generate_corrected_answer(inputs)
        assert "albert einstein" in corrected.lower()


# ── Evidence contradicts the claim ───────────────────────────────────────────

class TestContradictedClaims:

    def test_nli_contradiction_always_wins(self):
        evidence = [_ev(
            "Albert Einstein was born in 1879, not 1990.",
            similarity=STRONG + 0.05,
        )]
        with _with_nli_contradiction():
            verdict = determine_verdict(
                "Einstein was born in 1990", evidence
            )
        assert verdict.verdict == VERDICT_CONTRADICTED
        assert verdict.contradiction_detected is True

    def test_date_mismatch_returns_contradicted(self):
        """Year mismatch: claim says 1990, evidence says 1879."""
        evidence = [_ev(
            "Albert Einstein was born in 1879.",
            similarity=STRONG + 0.05,
        )]
        with _no_contradiction():
            verdict = determine_verdict(
                "Einstein was born in 1990", evidence
            )
        assert verdict.verdict == VERDICT_CONTRADICTED
        assert verdict.entity_mismatch is True

    def test_contradicted_claim_replaced_in_corrected_answer(self):
        inputs = [
            ClaimCorrectionInput(
                claim_text="Einstein was born in 1990",
                verdict=VERDICT_CONTRADICTED,
                best_evidence_text="Albert Einstein was born in 1879.",
                entity_mismatch=True,
            )
        ]
        corrected = generate_corrected_answer(inputs)
        # Evidence text (with 1879) should appear; original claim (1990) should not
        assert "1879" in corrected
        # Note about correction should appear
        assert "corrected" in corrected.lower() or "corrections applied" in corrected.lower()


# ── Insufficient evidence ────────────────────────────────────────────────────

class TestInsufficientEvidence:

    def test_no_evidence_returns_unsupported(self):
        verdict = determine_verdict("John McCarthy invented LISP", [])
        assert verdict.verdict == VERDICT_UNSUPPORTED

    def test_very_low_similarity_returns_unsupported(self):
        evidence = [_ev("Random unrelated text.", similarity=0.10)]
        with _no_contradiction():
            verdict = determine_verdict(
                "Marie Curie discovered polonium", evidence
            )
        assert verdict.verdict == VERDICT_UNSUPPORTED

    def test_corrected_answer_for_unsupported_claim(self):
        inputs = [
            ClaimCorrectionInput(
                claim_text="john mccarthy was the ceo of infosys",
                verdict=VERDICT_UNSUPPORTED,
                best_evidence_text=None,
                entity_mismatch=False,
            )
        ]
        corrected = generate_corrected_answer(inputs)
        assert "could not be verified" in corrected.lower() or "insufficient" in corrected.lower() or "unsupported" in corrected.lower()


# ── Partially supported (no entity mismatch) ─────────────────────────────────

class TestPartiallySupportedKept:

    def test_partial_no_mismatch_kept_in_corrected_answer(self):
        """PARTIALLY_SUPPORTED + entity_mismatch=False → kept with note."""
        inputs = [
            ClaimCorrectionInput(
                claim_text="Water is composed of hydrogen and oxygen",
                verdict=VERDICT_PARTIALLY_SUPPORTED,
                best_evidence_text="H2O is a molecule of two hydrogen atoms bonded to one oxygen.",
                entity_mismatch=False,
            )
        ]
        corrected = generate_corrected_answer(inputs)
        assert "water" in corrected.lower() or "hydrogen" in corrected.lower()

    def test_partial_with_mismatch_removed_from_corrected_answer(self):
        """PARTIALLY_SUPPORTED + entity_mismatch=True → removed."""
        inputs = [
            ClaimCorrectionInput(
                claim_text="john mccarthy was the ceo of infosys",
                verdict=VERDICT_PARTIALLY_SUPPORTED,
                best_evidence_text="Infosys is an IT company.",
                entity_mismatch=True,
            )
        ]
        corrected = generate_corrected_answer(inputs)
        # Claim text should not appear in the kept sentences
        # (it may appear in the note, so we check the body before the note)
        body = corrected.split("[Corrections applied]")[0]
        assert "john mccarthy was the ceo" not in body.lower()
