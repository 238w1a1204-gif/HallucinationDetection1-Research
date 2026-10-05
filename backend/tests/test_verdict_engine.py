"""Tests for app.verification.verdict_engine.

We mock the NLI contradiction detector and entity matcher so that
tests are deterministic and do not require downloading the ~1.6 GB model.
Embedding calls are also mocked to avoid the sentence-transformers model.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from app.verification.similarity_scorer import ScoredEvidence
from app.verification.verdict_engine import (
    determine_verdict,
    ClaimVerdict,
    VERDICT_VERIFIED,
    VERDICT_PARTIALLY_SUPPORTED,
    VERDICT_UNSUPPORTED,
    VERDICT_CONTRADICTED,
)
from app.config import get_settings

settings = get_settings()

# ── helpers ──────────────────────────────────────────────────────────────────

def _ev(similarity: float, source: str = "wikipedia") -> ScoredEvidence:
    """Shorthand: build a ScoredEvidence with the given similarity."""
    return ScoredEvidence(
        evidence_text="some evidence",
        source_name=source,
        source_url=None,
        similarity=similarity,
    )


def _no_contradiction():
    return patch(
        "app.verification.verdict_engine.detect_contradiction",
        return_value=(False, "entailment"),
    )


def _no_entity_mismatch():
    from app.verification.entity_matcher import EntityCheckResult
    return patch(
        "app.verification.verdict_engine.check_entity_consistency",
        return_value=EntityCheckResult(mismatch=False, reason="ok"),
    )


def _with_contradiction():
    return patch(
        "app.verification.verdict_engine.detect_contradiction",
        return_value=(True, "contradiction"),
    )


def _with_entity_mismatch():
    from app.verification.entity_matcher import EntityCheckResult
    return patch(
        "app.verification.verdict_engine.check_entity_consistency",
        return_value=EntityCheckResult(mismatch=True, reason="year mismatch"),
    )


# ── tests ─────────────────────────────────────────────────────────────────────

class TestDetermineVerdict:
    def test_no_evidence_returns_unsupported(self):
        verdict = determine_verdict("some claim", [])
        assert verdict.verdict == VERDICT_UNSUPPORTED
        assert verdict.max_similarity == 0.0
        assert verdict.num_sources_agreeing == 0

    def test_verified_with_high_similarity_and_multi_source_agreement(self):
        # Both sources above partial threshold → agree_count = 2 ≥ AGREEMENT_MIN_SOURCES (2)
        evidence = [
            _ev(settings.SIMILARITY_STRONG_THRESHOLD + 0.05, "wikipedia"),
            _ev(settings.SIMILARITY_PARTIAL_THRESHOLD + 0.01, "wikidata"),
        ]
        with _no_contradiction(), _no_entity_mismatch():
            verdict = determine_verdict("claim", evidence)
        assert verdict.verdict == VERDICT_VERIFIED

    def test_partially_supported_high_similarity_single_source(self):
        # Strong similarity but only one source agrees
        evidence = [
            _ev(settings.SIMILARITY_STRONG_THRESHOLD + 0.05, "wikipedia"),
            _ev(0.1, "wikidata"),  # below partial threshold → does not agree
        ]
        with _no_contradiction(), _no_entity_mismatch():
            verdict = determine_verdict("claim", evidence)
        assert verdict.verdict == VERDICT_PARTIALLY_SUPPORTED

    def test_partially_supported_medium_similarity(self):
        # Similarity between partial and strong thresholds
        mid = (settings.SIMILARITY_PARTIAL_THRESHOLD + settings.SIMILARITY_STRONG_THRESHOLD) / 2
        evidence = [_ev(mid, "wikipedia")]
        with _no_contradiction(), _no_entity_mismatch():
            verdict = determine_verdict("claim", evidence)
        assert verdict.verdict == VERDICT_PARTIALLY_SUPPORTED

    def test_unsupported_low_similarity(self):
        evidence = [_ev(settings.SIMILARITY_PARTIAL_THRESHOLD - 0.1, "wikipedia")]
        with _no_contradiction(), _no_entity_mismatch():
            verdict = determine_verdict("claim", evidence)
        assert verdict.verdict == VERDICT_UNSUPPORTED

    def test_contradicted_by_nli(self):
        evidence = [_ev(settings.SIMILARITY_STRONG_THRESHOLD + 0.05)]
        with _with_contradiction(), _no_entity_mismatch():
            verdict = determine_verdict("claim", evidence)
        assert verdict.verdict == VERDICT_CONTRADICTED
        assert verdict.contradiction_detected is True

    def test_contradicted_by_entity_mismatch_on_high_similarity(self):
        # Strong similarity + entity mismatch → CONTRADICTED (not VERIFIED)
        evidence = [_ev(settings.SIMILARITY_STRONG_THRESHOLD + 0.05)]
        with _no_contradiction(), _with_entity_mismatch():
            verdict = determine_verdict("claim", evidence)
        assert verdict.verdict == VERDICT_CONTRADICTED
        assert verdict.entity_mismatch is True

    def test_entity_mismatch_on_low_similarity_does_not_upgrade_to_contradicted(self):
        # Entity mismatch only triggers CONTRADICTED when similarity is also high
        evidence = [_ev(settings.SIMILARITY_PARTIAL_THRESHOLD - 0.1)]
        with _no_contradiction(), _with_entity_mismatch():
            verdict = determine_verdict("claim", evidence)
        # Low similarity → should be UNSUPPORTED, not CONTRADICTED
        assert verdict.verdict == VERDICT_UNSUPPORTED

    def test_verdict_object_has_correct_fields(self):
        evidence = [_ev(settings.SIMILARITY_STRONG_THRESHOLD + 0.05, "wikipedia"),
                    _ev(settings.SIMILARITY_PARTIAL_THRESHOLD + 0.01, "wikidata")]
        with _no_contradiction(), _no_entity_mismatch():
            v = determine_verdict("claim", evidence)
        assert isinstance(v, ClaimVerdict)
        assert isinstance(v.max_similarity, float)
        assert isinstance(v.num_sources_agreeing, int)
        assert isinstance(v.explanation, str)
        assert len(v.explanation) > 0

    def test_contradiction_wins_over_high_similarity(self):
        # Even if similarity is perfect and multiple sources agree,
        # NLI contradiction must produce CONTRADICTED.
        evidence = [
            _ev(1.0, "wikipedia"),
            _ev(0.9, "wikidata"),
            _ev(0.8, "duckduckgo"),
        ]
        with _with_contradiction(), _no_entity_mismatch():
            verdict = determine_verdict("claim", evidence)
        assert verdict.verdict == VERDICT_CONTRADICTED
